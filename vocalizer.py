# Vocalizer.py
import os
import time
import torch
import numpy as np
from scipy.io import wavfile
from typing import Dict, Any, Optional
from StyleTTS2.styletts2 import StyleTTS2

class GenerationAgent:
    def __init__(self, 
                 sample_rate: int = 24000,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        print("[INFO] Initializing GenerationAgent...")
        self.tts = StyleTTS2()
        self.sample_rate = sample_rate
        self.device = device
        
        # 不再固定参数，改为动态计算

    def generate(
        self, 
        text: str, 
        reference_voice_path: str, 
        save_dir: str, 
        output_filename: str = None,
        # ===== 新增：目标韵律特征 =====
        target_emotion: str = "neutral",
        target_emotion_score: float = 0.5,
        target_speaking_rate: float = None,  # None 表示跟随参考
        target_energy: float = None,  # None 表示跟随参考
        # ===== 新增：用户韵律特征（用于自适应）=====
        user_prosody: Optional[Dict[str, Any]] = None
    ) -> str:
        
        print(f"[INFO] Generating speech for: \"{text}\"")
        print(f"[INFO] Target emotion: {target_emotion} (intensity {target_emotion_score:.2f})")
        
        # ========================================
        # 1. 根据目标情绪和用户状态计算 TTS 参数
        # ========================================
        alpha, beta, diffusion_steps, embedding_scale = self._compute_tts_parameters(
            target_emotion=target_emotion,
            target_emotion_score=target_emotion_score,
            user_prosody=user_prosody
        )
        
        print(f"[INFO] Computed parameters:")
        print(f"  alpha={alpha:.2f} (timbre), beta={beta:.2f} (prosody)")
        print(f"  diffusion_steps={diffusion_steps}, embedding_scale={embedding_scale:.2f}")
        
        # ========================================
        # 2. 文本预处理（插入韵律标记）
        # ========================================
        enhanced_text = self._enhance_text_with_prosody(
            text=text,
            target_emotion=target_emotion,
            target_emotion_score=target_emotion_score,
            user_prosody=user_prosody
        )
        
        if enhanced_text != text:
            print(f"[INFO] Enhanced text: {enhanced_text}")
        
        # ========================================
        # 3. 加载参考风格
        # ========================================
        print(f"[INFO] Loading reference from: {reference_voice_path}")
        reference_style = self.tts.compute_style(reference_voice_path)
        
        # ========================================
        # 4. 生成语音
        # ========================================
        start = time.time()
        wav = self.tts.inference(
            enhanced_text,
            reference_style,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale
        )
        rtf = (time.time() - start) / (len(wav) / self.sample_rate)
        print(f"[INFO] Synthesis complete. RTF = {rtf:.4f}")
        
        # ========================================
        # 5. 后处理（调整语速、音量）
        # ========================================
        if target_speaking_rate is not None or target_energy is not None:
            wav = self._adjust_prosody(
                wav,
                target_speaking_rate=target_speaking_rate,
                target_energy=target_energy,
                user_prosody=user_prosody
            )
        
        # ========================================
        # 6. 保存
        # ========================================
        scaled_wav = np.int16(np.clip(wav, -1.0, 1.0) * 32767)
        
        if output_filename:
            result_name = output_filename
        else:
            result_name = reference_voice_path.split('/')[-1]
        
        wav_file_path = os.path.join(save_dir, result_name)
        os.makedirs(save_dir, exist_ok=True)
        wavfile.write(wav_file_path, self.sample_rate, scaled_wav)
        print(f"[INFO] Audio saved to: {wav_file_path}\n")
        
        return wav_file_path
    
    
    def _compute_tts_parameters(
        self,
        target_emotion: str,
        target_emotion_score: float,
        user_prosody: Optional[Dict[str, Any]]
    ) -> tuple[float, float, int, float]:
        """
        根据目标情绪和用户韵律计算 TTS 参数
        
        """
        
        # ========================================
        # 基础值
        # ========================================
        alpha = 0.3  # 音色相似度
        beta = 0.7   # 韵律相似度
        diffusion_steps = 5
        embedding_scale = 1.0
        
        # ========================================
        # 根据目标情绪调整
        # ========================================
        
        # 1. 情绪强度 → beta (韵律相似度)
        if target_emotion_score > 0.7:
            # 高情绪强度 → 需要更强的韵律表现
            beta = 0.8
            diffusion_steps = 10  # 更多步数，更细腻
        elif target_emotion_score > 0.5:
            beta = 0.7
            diffusion_steps = 7
        else:
            # 低情绪强度 → 更平淡
            beta = 0.6
            diffusion_steps = 5
        
        # 2. 情绪类型 → embedding_scale
        emotion_scales = {
            "happy": 1.2,      # 开心 → 更有活力
            "excited": 1.3,    # 兴奋 → 能量最高
            "sad": 0.7,        # 悲伤 → 更温和
            "angry": 1.1,      # 生气 → 略高能量
            "fearful": 0.8,    # 恐惧 → 较低
            "neutral": 1.0,    # 中性 → 默认
            "calm": 0.9        # 平静 → 略低
        }
        embedding_scale = emotion_scales.get(target_emotion.lower(), 1.0)
        
        # ========================================
        # 根据用户韵律自适应调整
        # ========================================
        if user_prosody:
            user_emotion = user_prosody.get("emotion", "neutral")
            user_certainty = user_prosody.get("certainty", 0.5)
            user_energy = user_prosody.get("energy_mean", 0.03)
            
            # 策略1: 如果用户很犹豫，回复应该更温和、确定
            if user_certainty < 0.4:
                print(f"[INFO] User is hesitant ({user_certainty:.2f}), adjusting to be more gentle")
                beta -= 0.1  # 降低韵律强度
                embedding_scale *= 0.9  # 降低能量
            
            # 策略2: 如果用户情绪很强烈，回复应该匹配或安抚
            if user_emotion in ["sad", "angry", "fearful"]:
                user_score = user_prosody.get("emotion_score", 0.0)
                if user_score > 0.7:
                    print(f"[INFO] User has strong {user_emotion} ({user_score:.2f}), matching empathy")
                    # 匹配用户的情绪强度（共情）
                    beta = min(beta + 0.1, 1.0)
            
            # 策略3: 如果用户说话很轻，回复也适当降低
            if user_energy < 0.025:
                print(f"[INFO] User speaks softly ({user_energy:.3f}), matching volume")
                embedding_scale *= 0.85
        
        # 限制范围
        alpha = float(np.clip(alpha, 0.0, 0.5))
        beta = float(np.clip(beta, 0.5, 1.0))
        diffusion_steps = int(np.clip(diffusion_steps, 3, 15))
        embedding_scale = float(np.clip(embedding_scale, 0.5, 1.5))
        
        return alpha, beta, diffusion_steps, embedding_scale
    
    
    def _enhance_text_with_prosody(
        self,
        text: str,
        target_emotion: str,
        target_emotion_score: float,
        user_prosody: Optional[Dict[str, Any]]
    ) -> str:
        """
        在文本中插入韵律控制标记
        """
        enhanced = text
        
        # ========================================
        # 1. 根据用户停顿习惯添加停顿
        # ========================================
        if user_prosody:
            pause_ratio = user_prosody.get("pause_ratio", 0.0)
            # print(pause_ratio)
            # 如果用户停顿多，回复中也适当加停顿（表示耐心倾听）
            if pause_ratio > 0.3:
                # 在逗号和句号后加短暂停顿
                enhanced = enhanced.replace(', ', ',... ')
                enhanced = enhanced.replace('. ', '... ')
                # print("enhanced:", enhanced)
        
        # ========================================
        # 2. 根据目标情绪调整语气词
        # ========================================
        if target_emotion == "happy" and target_emotion_score > 0.7:
            # 开心 → 可以加感叹号
            if not enhanced.endswith('!'):
                enhanced = enhanced.rstrip('.') + '!'
        
        elif target_emotion == "sad":
            # 悲伤/共情 → 语气更温和
            enhanced = enhanced.replace('!', '.')

        return enhanced
    
    
    def _adjust_prosody(
        self,
        wav: np.ndarray,
        target_speaking_rate: Optional[float],
        target_energy: Optional[float],
        user_prosody: Optional[Dict[str, Any]]
    ) -> np.ndarray:

        adjusted_wav = wav.copy()
        
        # ========================================
        # 1. 调整语速（需要 librosa 或 soundfile）
        # ========================================
        if target_speaking_rate is not None and user_prosody:
            user_rate = user_prosody.get("speaking_rate", 4.0)
            
            # 计算语速比例
            # 如果用户说得很慢（2.0），回复也稍慢一些（3.0）
            # 如果用户说得很快（6.0），回复可以正常（4.5）
            
            if user_rate < 3.0:
                # 用户说得慢 → 回复也稍慢（体现耐心）
                speed_factor = 0.9  # 稍微变慢
            elif user_rate > 5.5:
                # 用户说得快 → 回复可以稍快（匹配节奏）
                speed_factor = 1.1
            else:
                speed_factor = 1.0
            
            # 应用语速调整（需要 librosa）
            try:
                import librosa
                adjusted_wav = librosa.effects.time_stretch(adjusted_wav, rate=speed_factor)
                print(f"[INFO] Adjusted speaking rate by {speed_factor:.2f}x")
            except ImportError:
                print("[WARNING] librosa not available, skipping speed adjustment")
        
        # ========================================
        # 2. 调整音量
        # ========================================
        if target_energy is not None and user_prosody:
            user_energy = user_prosody.get("energy_mean", 0.03)
            
            # 如果用户说话很轻，回复也适当降低音量
            if user_energy < 0.025:
                volume_factor = 0.85
            elif user_energy > 0.05:
                volume_factor = 1.1
            else:
                volume_factor = 1.0
            
            adjusted_wav = adjusted_wav * volume_factor
            print(f"[INFO] Adjusted volume by {volume_factor:.2f}x")
        
        return adjusted_wav