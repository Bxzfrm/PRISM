# main.py
from __future__ import annotations

import json
import os
from dataclasses import asdict

from perceiver import SpeechPerceiver, ASRWhisper, Emotion2VecFunASR, EmotionStub
from manager import SpeechDescriptionManager
from responder import Responder
from vocalizer import GenerationAgent


STANDARD_EMOTIONS = ["bright", "clear", "husky", "low_pitched", "melodious", "soft", "warm"]

tone_suggestions = {
    "angry": "husky",
    "disgusted": "husky",
    "fearful": "soft",
    "happy": "bright",
    "neutral": "clear",
    "other": "melodious",
    "sad": "warm",
    "surprised": "melodious",
    "unknown": "soft"
}


def main():
    # ===== 路径 =====
    dataset_json_path = ""    # path to test set JSON
    audio_root = ""           # root dir of audio files
    out_jsonl = ""            # output path
    reference_audio_root = "" # root dir of reference audio

    # ===== 模型路径 =====
    # perceiver emotion2vec
    emotion2vec_model_path = ""

    # responder: LLM + COMET
    responder_model_path = ""
    comet_path = ""

    # OpenAI API 配置（用于 manager）
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")  


    # ===== 初始化 perceiver / manager / responder / vocalizer =====
    # 1. Perceiver（语音分析）
    print("  - Loading ASR (Whisper)...")
    asr = ASRWhisper("base")  

    print("  - Loading Emotion2Vec...")
    emotion_model = Emotion2VecFunASR(
        model_path=emotion2vec_model_path,
        output_dir="./outputs",
        granularity="utterance",
    )

    print("  - Initializing SpeechPerceiver...")
    perceiver = SpeechPerceiver(asr=asr, embedder=None, emotion_model=emotion_model)

    # 2. Manager（生成自然语言描述）
    print("  - Initializing SpeechDescriptionManager...")
    mgr = SpeechDescriptionManager(
        api_key=openai_api_key,
        base_url=openai_base_url,
        model="gpt-3.5-turbo"
    )

    # 3. Responder（生成回复）
    responder = Responder(
        model_path=responder_model_path,
        comet_path=comet_path,
        device="cuda",
    )

    # 4. Vocalizer（生成语音）
    generator = GenerationAgent(device="cuda")

    # ===== 批量读取数据并处理 =====
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for conversation in data:
        conversation_id = conversation["conversation_id"]
        speaker_profile = conversation["speaker_profile"]
        spk_id = speaker_profile["ID"]

        # 取最后一轮 turn
        max_turn = max(conversation["turns"], key=lambda x: int(x["turn_id"]))
        dialogue_history = max_turn["dialogue_history"]

        # 取最大 index
        max_index = max(dialogue_history, key=lambda x: x["index"])
        mi = max_index["index"]

        turn_id = "dia" + conversation_id
        audio_path = f"{audio_root}/dia{conversation_id}utt{mi}_{spk_id}.wav"

        # ===== 1) perceiver 语音分析 =====
        speech_state_obj = perceiver.perceive(audio_path, language=None)
        speech_state = {
            "transcript": speech_state_obj.transcript,
            "attributes": asdict(speech_state_obj.attributes),
            "prosody_embedding": speech_state_obj.prosody_embedding,
            "meta": speech_state_obj.meta,
        }

        transcript = speech_state["transcript"]
        emotion = speech_state["attributes"].get("emotion") or "unknown"

        # ===== 2) manager 生成韵律描述 =====
        prosody_description = mgr.generate_description_v2(speech_state)
        print(f"韵律描述: {prosody_description}")

        # ===== 3) responder 生成回复 =====
        reply, tone, knowledge = responder.respond(transcript, prosody_description, dialogue_history, emotion)
        print(f"reply: {reply}")
        print(f"tone: {tone}")
        print(f"knowledge: {knowledge}")


        # ===== 3.1) manager 验证 =====
        verify_result = mgr.verify_response(
            prosody_description=prosody_description,
            reply=reply,
            tone=tone,
            user_emotion=emotion,
        )
        print(f"[VERIFY] aligned={verify_result['aligned']}, issues={verify_result['issues']}")
        print(f"[VERIFY] suggestions={verify_result['suggestions']}")

        # ===== 3.2) responder 更新回复 =====
        history_text = responder.get_dialogue_history(dialogue_history)
        reply, tone, adopted = responder.revise(
            reply=reply,
            tone=tone,
            verify_result=verify_result,
            transcript=transcript,
            prosody_description=prosody_description,
            dialogue_history=history_text,
            emotion=emotion,
        )
        print(f"[REVISE] adopted={adopted}")
        print(f"[FINAL] reply={reply}")
        print(f"[FINAL] tone={tone}")

        # ===== 4) vocalizer =====
        # # 从 tone 中提取目标情绪
        target_emotion = tone["emotion"]           
        target_emotion_score = tone["emotion_score"]   
        if target_emotion in STANDARD_EMOTIONS:
            target_emotion = target_emotion
        else:
            target_emotion = tone_suggestions.get(emotion, "clear")
        print(f"target_emotion: {target_emotion}")
        print(f"target_emotion_score: {target_emotion_score}")

        # 参考音频
        gender = speaker_profile['gender']
        reference_voice_path = os.path.join(reference_audio_root, gender, target_emotion + ".wav")
        
        audio_path = generator.generate(
            text=reply,
            reference_voice_path=reference_voice_path,  # 参考音频
            save_dir="./outputs",
            output_filename=f"{turn_id}.wav",
            # 关键：传入目标情绪和用户韵律
            target_emotion=target_emotion,
            target_emotion_score=target_emotion_score,
            user_prosody=speech_state["attributes"]  # 用户的韵律特征
        )
        
        print(f"Generated audio: {audio_path}")

        # ===== 写入结果 =====
        new_rply = {
            "turn_id": turn_id,
            "utterance": reply,
            "knowledge": knowledge,
            "tone": tone,
            "audio_path": audio_path,
            "prosody_description":  prosody_description
        }
        with open(out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(new_rply, ensure_ascii=False) + "\n")
        print("已存：", turn_id)


if __name__ == "__main__":
    # print("111")
    main()
