# perceiver.py
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import whisper
import soundfile as sf  
import librosa  
import webrtcvad  
import parselmouth 
import torch  
from transformers import AutoModel, AutoFeatureExtractor
from faster_whisper import WhisperModel  # type: ignore
from funasr import AutoModel as FunASRAutoModel  # type: ignore

@dataclass
class SpeechAttributes:
    """
    语音属性
    """
    emotion: Optional[str] = None          # 情绪类别（emotion2vec 输出的 label）
    emotion_score: Optional[float] = None  # 情绪 top1 分数/置信度（emotion2vec scores 的最大值）
    certainty: Optional[float] = None      # “确定性”启发式评分 0~1
    speaking_rate: Optional[float] = None  # 语速 tokens/sec（基于转写+VAD时间估计）
    pause_ratio: Optional[float] = None    # 停顿占比（静音时间/总时间）

    # 额外统计特征
    energy_mean: Optional[float] = None    # RMS 能量均值
    energy_std: Optional[float] = None     # RMS 能量标准差
    filler_rate: Optional[float] = None    # 填充词比例（从转写统计）


@dataclass
class SpeechState:

    transcript: str
    attributes: SpeechAttributes
    prosody_embedding: Optional[List[float]] = None  # 256-d (or any length)
    meta: Optional[Dict[str, Any]] = None

# -----------------------------
# 工具函数
# -----------------------------
def _require(cond: bool, msg: str):
    
    if not cond:
        raise RuntimeError(msg)

# 读取音频 → 转单声道 → 重采样到16kHz → 归一化到[-1,1]
def _load_audio_mono(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    _require(sf is not None, "Please install soundfile: pip install soundfile")
    audio, sr = sf.read(path, dtype="float32", always_2d=False)

    # 多通道 => 取均值做 mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)

    # 重采样到 target_sr
    if sr != target_sr:
        _require(librosa is not None, "Resampling requires librosa: pip install librosa")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr

    audio = np.clip(audio, -1.0, 1.0)
    return audio, sr


# RMS（均方根）能量的平均值
def _rms_energy(audio: np.ndarray, frame_len: int = 400, hop: int = 160) -> np.ndarray:

    if len(audio) < frame_len:
        return np.array([float(np.sqrt(np.mean(audio**2) + 1e-9))], dtype=np.float32)

    # 1. 分帧
    frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop).T if librosa else None
    if frames is None:
        return np.array([float(np.sqrt(np.mean(audio**2) + 1e-9))], dtype=np.float32)

    # 2. 计算每帧的 RMS 能量
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-9)
    return rms.astype(np.float32)


# 语音活动检测VAD
# sr: 16000 (Hz) 是 sampling rate（采样率） 的缩写
# 采样率 = 每秒钟对音频信号采样的次数，单位是 Hz（赫兹）

def _vad_speech_mask(audio: np.ndarray, sr: int, aggressiveness: int = 2) -> Tuple[float, float, float]:

    total_time = len(audio) / sr
    if total_time <= 1e-6:
        return 0.0, 0.0, 0.0

    # 方法1：WebRTC VAD（优先）
    if webrtcvad is not None and sr in (8000, 16000, 32000, 48000):
        vad = webrtcvad.Vad(aggressiveness)

        # 将音频分成 30ms 的帧
        frame_ms = 30
        frame_len = int(sr * frame_ms / 1000)

        pcm16 = (np.clip(audio, -1, 1) * 32768.0).astype(np.int16).tobytes()

        # 逐帧检测是否为语音
        speech_frames = 0
        total_frames = 0
        for i in range(0, len(pcm16), frame_len * 2):
            chunk = pcm16[i: i + frame_len * 2]
            if len(chunk) < frame_len * 2:
                break
            total_frames += 1
            if vad.is_speech(chunk, sr):   # WebRTC 判断
                speech_frames += 1
        
        # 计算说话时长
        speech_time = speech_frames * (frame_ms / 1000.0)
        speech_time = min(speech_time, total_time)
        silence_time = max(total_time - speech_time, 0.0)
        pause_ratio = silence_time / total_time
        return speech_time, silence_time, pause_ratio

    # ---------- 2) 能量阈值兜底 ----------
    if librosa is None:
        return total_time, 0.0, 0.0

    rms = _rms_energy(audio)
    thr = max(np.percentile(rms, 20), 1e-4) * 1.5  # heuristic

    speech_frames = float(np.sum(rms > thr))
    total_frames = float(len(rms))

    speech_ratio = speech_frames / max(total_frames, 1.0)
    speech_time = speech_ratio * total_time
    silence_time = total_time - speech_time
    pause_ratio = silence_time / total_time
    return float(speech_time), float(silence_time), float(pause_ratio)


def _estimate_f0_parselmouth(audio: np.ndarray, sr: int) -> Tuple[Optional[float], Optional[float]]:
    """
    用 Parselmouth(Praat) 估计 mean/std of F0 (Hz)。
    """
    if parselmouth is None:
        return None, None
    try:
        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=0.01)  # 10ms
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]
        if f0.size == 0:
            return None, None
        return float(np.mean(f0)), float(np.std(f0))
    except Exception:
        return None, None

# 定义中文填充词
_CN_FILLERS = ["嗯", "呃", "额", "那个", "这个", "就是", "然后", "可能", "大概", "好像", "有点"]

# 定义英文填充词
_EN_FILLERS = ["um", "uh", "er", "like", "you know", "maybe", "kinda", "sort of", "i guess"]


# 统计填充词在转写文本中的比例
def _filler_rate_from_text(text: str) -> float:
    """
    Approximate filler rate: filler_count / max(tokens,1)
    For Chinese, use character/phrase match; for English, word-level.
    """
    if not text:
        return 0.0
    t = text.lower()

    # 统计 token 数
    en_tokens = re.findall(r"[a-z']+", t)
    token_count = len(en_tokens) if en_tokens else max(len(text), 1)

    # 统计填充词出现次数
    filler_count = 0
    for f in _CN_FILLERS:
        filler_count += text.count(f)
    for f in _EN_FILLERS:
        filler_count += t.count(f)

    # 计算比例
    return float(filler_count) / float(max(token_count, 1))


# 基于多个指标的启发式评分
def _certainty_heuristic(
    pause_ratio: float,
    speaking_rate: float,
    filler_rate: float,
) -> float:
    """
    Heuristic certainty in [0,1].
    Lower certainty when: high pause_ratio, high filler_rate, very low speaking_rate.
    """
    # 输入：
    # pause_ratio、speaking_rate、filler_rate 

    # 1. 语速归一化（映射到 0-1）
    sr_norm = (speaking_rate - 2.0) / 4.0  # 2->0, 6->1
    sr_norm = float(np.clip(sr_norm, 0.0, 1.0))

    # 2. 停顿占比归一化
    pr = float(np.clip(pause_ratio, 0.0, 1.0))

    # 3. 填充词率归一化（除以 0.2 作为基准）
    fr = float(np.clip(filler_rate / 0.2, 0.0, 1.0))

    # 4. 加权求和
    base = 0.55 * sr_norm + 0.25 * (1.0 - pr) + 0.20 * (1.0 - fr)

    return float(np.clip(base, 0.0, 1.0))


# 转写文本 token 数 ÷ 实际说话时长
def _speaking_rate_tokens_per_sec(transcript: str, speech_time: float) -> float:
    """
    speaking_rate in tokens/sec based on transcript token count and VAD speech_time.
    """
    if speech_time <= 1e-6:
        return 0.0
    
    # 1. 检测是否为英文
    en_tokens = re.findall(r"[a-zA-Z']+", transcript)
    # 2. 中文场景：统计字符数（去除空格）
    if len(en_tokens) >= 2:
        token_count = len(en_tokens)
    else:
        token_count = max(len(transcript.replace(" ", "")), 1)
    
    # 3. 返回计算语速
    return float(token_count) / float(speech_time)


class ASRWhisper:

    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        _require(whisper is not None, "Install openai-whisper: pip install -U openai-whisper")

        # device: None 时 whisper 会自己选择（一般有 CUDA 就用 CUDA）
        # 也可以强制传入 "cuda" 或 "cpu"
        if device is None:
            self.model = whisper.load_model(model_name)
        else:
            self.model = whisper.load_model(model_name, device=device)

    def transcribe(self, wav_path: str, language: Optional[str] = None) -> Tuple[str, Optional[float]]:
        # openai-whisper 的 transcribe 支持 language 提示
        # 调用 OpenAI Whisper 模型
        result = self.model.transcribe(wav_path, language=language)
        text = (result.get("text") or "").strip()
        return text


# 来源： WavLM 或其他语音模型的隐藏层输出
class ProsodyEmbedderHF:
    """
    Create an utterance-level embedding using a HuggingFace speech model.
    Requires transformers + torch.
    """
    def __init__(self, model_id: str = "microsoft/wavlm-base-plus", device: Optional[str] = None):
        _require(
            torch is not None and AutoModel is not None and AutoFeatureExtractor is not None,
            "Install torch + transformers to use HF embedder."
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # 1. 特征提取
        inputs = self.extractor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 2. 前向传播
        out = self.model(**inputs)

        # 3. 时间维度池化（平均） 4. 转为 numpy
        hs = out.last_hidden_state  # [B, T, C]
        emb = hs.mean(dim=1).squeeze(0).detach().cpu().numpy()  # [C]
        return emb.astype(np.float32)


class EmotionStub:
    """
    情绪模型占位符：默认返回 None 避免“假装准确”。
    """
    def predict(
        self,
        audio: np.ndarray,
        sr: int,
        transcript: str = ""
    ) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        return None, None, None


class Emotion2VecFunASR:
    """
    使用 FunASR 的 emotion2vec 做句级(utterance)情绪分类。
    """
    def __init__(
        self,
        model_path: str,
        output_dir: str = "./outputs",
        granularity: str = "utterance",
    ):
        self.model = FunASRAutoModel(model=model_path)
        self.output_dir = output_dir
        self.granularity = granularity

        self.last_score: Optional[float] = None
        self.last_label: Optional[str] = None
        self.last_all: Optional[Dict[str, Any]] = None

    def predict(
        self,
        audio: np.ndarray,
        sr: int,
        transcript: str = ""
    ) -> Tuple[Optional[str], Optional[float], Optional[float]]:

        # 保证 float32 mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1).astype(np.float32)
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.clip(audio, -1.0, 1.0)

        tmp_wav = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_wav = f.name
            sf.write(tmp_wav, audio, sr)

            rec_result = self.model.generate(
                input=tmp_wav,
                output_dir=self.output_dir,
                granularity=self.granularity,
                extract_embedding=False
            )

            # 兼容性检查
            if not rec_result or not isinstance(rec_result, list) or not isinstance(rec_result[0], dict):
                self.last_score, self.last_label = None, None
                self.last_all = {"raw": rec_result}
                return None, None, None

            if "labels" not in rec_result[0] or "scores" not in rec_result[0]:
                self.last_score, self.last_label = None, None
                self.last_all = {"raw": rec_result}
                return None, None, None

            labels = rec_result[0]["labels"]
            scores = rec_result[0]["scores"]

            if not labels or not scores:
                self.last_score, self.last_label = None, None
                self.last_all = {"labels": labels, "scores": scores}
                return None, None, None

            max_idx = int(np.argmax(scores))
            emotion = str(labels[max_idx])
            score = float(scores[max_idx])

            self.last_score = score
            self.last_label = emotion
            self.last_all = {"labels": labels, "scores": scores}

            return emotion

        finally:
            if tmp_wav is not None and os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass


# -----------------------------
# Main Perceiver
# -----------------------------
class SpeechPerceiver:
    def __init__(
        self,
        asr: Optional[Any] = None,
        embedder: Optional[Any] = None,
        emotion_model: Optional[Any] = None,
        target_sr: int = 16000,
    ):

        self.asr = asr
        self.embedder = embedder
        self.emotion_model = emotion_model or EmotionStub()
        self.target_sr = target_sr

    def perceive(self, wav_path: str, language: Optional[str] = None) -> SpeechState:
        # Load audio
        audio, sr = _load_audio_mono(wav_path, target_sr=self.target_sr)

        # 1) ASR
        transcript = ""
        if self.asr is not None:
            transcript = self.asr.transcribe(wav_path, language=language)

        # 2) VAD stats
        speech_time, silence_time, pause_ratio = _vad_speech_mask(audio, sr)

        # 3) speaking rate
        duration = len(audio) / sr
        speaking_rate = _speaking_rate_tokens_per_sec(
            transcript,
            speech_time if speech_time > 0 else duration
        )

        # 4) energy stats
        energy = _rms_energy(audio) if librosa is not None else np.array([float(np.sqrt(np.mean(audio**2) + 1e-9))])
        energy_mean = float(np.mean(energy))
        energy_std = float(np.std(energy))

        # 5) emotion
        emotion = self.emotion_model.predict(audio, sr, transcript=transcript)
        # emotion2vec 的 top1 score（如果 emotion_model 有保存）
        emotion_score = getattr(self.emotion_model, "last_score", None)

        # 6) filler rate from transcript
        filler_rate = _filler_rate_from_text(transcript)

        # 7) certainty (heuristic)
        certainty = _certainty_heuristic(
            pause_ratio=pause_ratio,
            speaking_rate=speaking_rate,
            filler_rate=filler_rate,
        )

        # 8) prosody embedding (optional)
        prosody_embedding = None
        if self.embedder is not None:
            emb = self.embedder.embed(audio, sr)
            prosody_embedding = emb.astype(np.float32).tolist()

        attrs = SpeechAttributes(
            emotion=emotion,
            emotion_score=emotion_score,
            certainty=certainty,
            speaking_rate=speaking_rate,
            pause_ratio=pause_ratio,
            filler_rate=filler_rate,
            energy_mean=energy_mean,
            energy_std=energy_std,
        )

        return SpeechState(
            transcript=transcript,
            attributes=attrs,
            prosody_embedding=prosody_embedding,
            meta={
                "sr": sr,
                "audio_duration_sec": float(duration),
                "speech_time_sec": float(speech_time),
                "silence_time_sec": float(silence_time),
                "emotion2vec": getattr(self.emotion_model, "last_all", None),
            },
        )


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True, help="Path to wav file")
    parser.add_argument("--lang", type=str, default=None, help="ASR language hint (e.g., 'zh')")

    parser.add_argument("--use_asr", action="store_true", help="Use faster-whisper ASR if installed")
    parser.add_argument("--asr_model", type=str, default="small")

    parser.add_argument("--use_embed", action="store_true", help="Use HF embedding if installed")
    parser.add_argument("--embed_model", type=str, default="microsoft/wavlm-base-plus")

    # emotion2vec
    parser.add_argument("--use_emotion2vec", action="store_true", help="Use FunASR emotion2vec if installed")
    parser.add_argument("--emotion2vec_model_path", type=str, default="", help="Path to emotion2vec model dir")
    parser.add_argument("--emotion2vec_output_dir", type=str, default="./outputs")

    args = parser.parse_args()

    asr = ASRWhisper(args.asr_model) if args.use_asr else None
    embedder = ProsodyEmbedderHF(args.embed_model) if args.use_embed else None

    if args.use_emotion2vec:
        _require(bool(args.emotion2vec_model_path), "启用 --use_emotion2vec 时必须提供 --emotion2vec_model_path")
        emotion_model = Emotion2VecFunASR(
            model_path=args.emotion2vec_model_path,
            output_dir=args.emotion2vec_output_dir,
            granularity="utterance",
        )
    else:
        emotion_model = EmotionStub()

    perceiver = SpeechPerceiver(asr=asr, embedder=embedder, emotion_model=emotion_model)
    state = perceiver.perceive(args.wav, language=args.lang)

    print(json.dumps({
        "transcript": state.transcript,
        "attributes": asdict(state.attributes),
        "prosody_embedding": None if state.prosody_embedding is None else {
            "dim": len(state.prosody_embedding),
            "preview": state.prosody_embedding[:8]
        },
        "meta": state.meta
    }, ensure_ascii=False, indent=2))
