# responder.py
from __future__ import annotations

import json
import re
import gc
import logging
from typing import Any, List, Optional, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from funasr import AutoModel as FunASRAutoModel  
import re

# ====================== 配置日志 ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================== 常量 ======================
knowledge_origin = (
    "Don't rush to reply, I can provide the following additional knowledge to help you provide a better reply. "
    "The following are the definitions of the five commonsense relations, followed by the content of the five relations "
    "extracted from the existing conversation. You can combine them and the dialogue context generates the final reply."
)
descriptions = '''
xWant represents what they would want after the event.
xNeed represents what they need in order for the event to happen.
xEffect represents the effect of the event on the person.
xReact represents their reaction to the event.
xIntent represents their intent before the event.
'''

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

Relations = ["xWant", "xNeed", "xEffect", "xReact", "xIntent"]
STANDARD_EMOTIONS = ["bright", "clear", "husky", "low_pitched", "melodious", "soft", "warm"]


class Responder:

    def __init__(
        self,
        model_path: str,
        comet_path: str,
        device: str = "cuda",
        torch_dtype=torch.float16,
    ):
        self.device = device
        
        try:
            logger.info("开始初始化 Responder 模型...")
            
            # ========== 初始化对话模型 ==========
            logger.info(f"加载模型: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16, 
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.eval()

            # ========== 初始化 COMET ==========
            logger.info(f"加载 COMET 模型: {comet_path}")
            self.kg_tokenizer = AutoTokenizer.from_pretrained(comet_path)
            self.kg_model = AutoModelForSeq2SeqLM.from_pretrained(comet_path).to(device)
            self.kg_model.eval()
            
            logger.info("所有模型加载成功！")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise

    # ------------------ 你的原有工具函数：history ------------------
    @staticmethod
    def get_dialogue_history(dialogue_history: List[Dict[str, Any]]) -> str:
        try:
            history_text = ""
            for turn in dialogue_history:
                role = "user" if turn["role"] == "speaker" else "system"
                history_text += f"{role}: {turn['utterance']}\n"
            return history_text
        except Exception as e:
            logger.error(f"处理对话历史失败: {e}")
            return ""

    @staticmethod
    def parse_history_to_messages(history_text: str) -> List[Dict[str, str]]:
        try:
            messages = []
            for line in history_text.strip().split("\n"):
                if not line.strip():
                    continue
                if line.startswith("user:"):
                    messages.append({"role": "user", "content": line[len("user:"):].strip()})
                elif line.startswith("system:"):
                    messages.append({"role": "system", "content": line[len("system:"):].strip()})
                elif line.startswith("assistant:"):
                    messages.append({"role": "assistant", "content": line[len("assistant:"):].strip()})
                else:
                    messages.append({"role": "user", "content": line.strip()})
            return messages
        except Exception as e:
            logger.error(f"解析历史消息失败: {e}")
            return []

    # ------------------ 工具调用决策：get_model_response / get_final_response ------------------
    def get_model_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            system_prompt = """You are an assistant that can call external tools when necessary.
                If the user input requires external knowledge, respond ONLY in the following format:

                TOOL_CALL: {"name": "KnowledgeAgent.generate_knowledge", "arguments": {"prompt": "Use knowledge"}}

                Otherwise, just answer normally as an assistant.
                """
            prompt = f"System: {system_prompt}\n"
            for msg in messages:
                prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
            prompt += "Assistant:"

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_reply = output_text.split("Assistant:")[-1].strip()
            
            # 清理内存
            del inputs, outputs
            torch.cuda.empty_cache()
            
            return assistant_reply
            
        except KeyboardInterrupt:
            logger.warning("模型生成被用户中断 (Ctrl+C)")
            raise  # 重新抛出以完全停止程序
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU 内存不足！")
            torch.cuda.empty_cache()
            return ""
        except Exception as e:
            logger.error(f"模型响应生成失败: {e}")
            return ""

    def get_final_response(self, messages: List[Dict[str, str]]):
        try:
            assistant_reply = self.get_model_response(messages)
            knowledge = None
            
            if "TOOL_CALL:" in assistant_reply:
                last_user_content = ""
                for m in reversed(messages):
                    if m["role"] == "user":
                        last_user_content = m["content"]
                        break
                
                if last_user_content:
                    knowledge = self.generate_knowledge(last_user_content)
                else:
                    logger.warning("未找到用户输入，无法生成知识")
            
            return knowledge
            
        except Exception as e:
            logger.error(f"获取最终响应失败: {e}")
            return None

    # ------------------ COMET knowledge ------------------
    @staticmethod
    def dict_to_relations(knowledge: Dict[str, str]) -> List[Optional[str]]:
        try:
            order = ["xWant", "xNeed", "xEffect", "xReact", "xIntent"]
            return [knowledge.get(k) if knowledge.get(k) not in [None, "none"] else None for k in order]
        except Exception as e:
            logger.error(f"转换知识字典失败: {e}")
            return [None] * 5

    def generate_knowledge(self, prompt: str, relations=None, beams: int = 3):
        try:
            event = prompt
            if relations is None:
                relations = Relations
            results = {}
            
            for rel in relations:
                try:
                    query = f"{event} {rel} [GEN]"
                    tokenized = self.kg_tokenizer(
                        query, 
                        return_tensors="pt", 
                        truncation=True, 
                        padding="max_length"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        output_ids = self.kg_model.generate(
                            **tokenized,
                            num_beams=beams,
                            do_sample=False,
                            max_new_tokens=30,
                            early_stopping=True,
                            pad_token_id=self.kg_tokenizer.eos_token_id
                        )
                    
                    decoded = self.kg_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    results[rel] = decoded[0]
                    
                    # 清理内存
                    del tokenized, output_ids
                    
                except Exception as e:
                    logger.error(f"生成关系 {rel} 失败: {e}")
                    results[rel] = None
            
            torch.cuda.empty_cache()
            return self.dict_to_relations(results)
            
        except KeyboardInterrupt:
            logger.warning("知识生成被用户中断")
            raise
        except Exception as e:
            logger.error(f"知识生成失败: {e}")
            return [None] * 5

    @staticmethod
    def format_knowledge_for_prompt(knowledge, origin=knowledge_origin, desc=descriptions):
        try:
            relations = ["xWant", "xNeed", "xEffect", "xReact", "xIntent"]

            if knowledge is None:
                knowledge_text = "No extracted knowledge."
            elif isinstance(knowledge, list):
                lines = []
                for i, val in enumerate(knowledge):
                    rel = relations[i] if i < len(relations) else f"extra_{i - len(relations) + 1}"
                    if val is None or (isinstance(val, str) and val.strip() == ""):
                        v = "Unknown"
                    else:
                        v = str(val).strip()
                    lines.append(f"{rel}: {v}")
                knowledge_text = "\n".join(lines)
            elif isinstance(knowledge, dict):
                lines = []
                for rel in relations:
                    val = knowledge.get(rel, "Unknown")
                    lines.append(f"{rel}: {val if val is not None else 'Unknown'}")
                knowledge_text = "\n".join(lines)
            else:
                knowledge_text = str(knowledge).strip() or "No extracted knowledge."

            return f"""{origin}
                    {desc}
                    Extracted Knowledge from the conversation:
                    {knowledge_text}"""
        except Exception as e:
            logger.error(f"格式化知识失败: {e}")
            return "No extracted knowledge."

    # ------------------ prompt 构建（保持你的原样） ------------------
    @staticmethod
    def build_prompt(transcript, prosody_description, dialogue_history):
        if isinstance(dialogue_history, list):
            dialogue_history = "\n".join(dialogue_history)

        return f"""
            You are a warm, empathetic assistant who understands emotions and responds with care.
            Your goal is to make the user feel understood and supported — not just advised.

            Dialogue History:
            {dialogue_history.strip()}

            Current User Input:
            User said: "{transcript}"

            User's Speech Analysis (prosody):
            {prosody_description}

            Your Task:
                1. Generate a short, natural, emotionally aware response (≤100 words).
                2. The response should fit the user's emotional state:
                - If sad → show understanding, comfort gently
                - If angry → stay calm, validate their feelings
                - If happy → share their joy and encourage
                - If anxious → reassure and calm them down
                3. Select an appropriate emotional tone from: {STANDARD_EMOTIONS}
                4. Set score (0.0-1.0) to indicate intensity.

            OUTPUT FORMAT:
            You MUST output ONLY valid JSON in this exact format:
            {{"reply": "your empathetic response here", "emotion": "warm", "score": 0.8}}

            IMPORTANT:
            - Do NOT include any text before or after the JSON
            - Do NOT use markdown code blocks (no ```json```)
            - Just output the raw JSON object

            Example outputs:
            {{"reply": "That sounds really tough. I'm here if you need to talk.", "emotion": "neutral", "score": 0.6}}
            {{"reply": "That's wonderful news! I'm so happy for you!", "emotion": "bright", "score": 0.9}}

            Now generate your response in JSON format:""".strip()

    def build_prompt_knowledge(self, transcript, prosody_description, dialogue_history, knowledge):
        if isinstance(dialogue_history, list):
            dialogue_history = "\n".join(dialogue_history)

        knowledge_block = self.format_knowledge_for_prompt(knowledge)

        return f"""
            You are a warm, empathetic assistant who understands emotions and responds with care.
            Your goal is to make the user feel understood and supported — not just advised.

            Dialogue History:
            {dialogue_history.strip()}

            Current User Input:
            User said: "{transcript}"

            User's Speech Analysis (prosody):
            {prosody_description}

            Relevant Knowledge (optional for context):
            {knowledge_block}

            Your Task:
            1. Generate a short, natural, emotionally aware response (≤100 words).
            2. The response should fit the user's emotional state:
            - If sad → show understanding, comfort gently
            - If angry → stay calm, validate their feelings
            - If happy → share their joy and encourage
            - If anxious → reassure and calm them down
            3. Select an appropriate emotional tone from: {STANDARD_EMOTIONS}
            4. Set score (0.0-1.0) to indicate intensity.

            OUTPUT FORMAT:
            You MUST output ONLY valid JSON in this exact format:
            {{"reply": "your empathetic response here", "emotion": "warm", "score": 0.8}}

            IMPORTANT:
            - Do NOT include any text before or after the JSON
            - Do NOT use markdown code blocks (no ```json```)
            - Just output the raw JSON object

            Example outputs:
            {{"reply": "That sounds really tough. I'm here if you need to talk.", "emotion": "neutral", "score": 0.6}}
            {{"reply": "That's wonderful news! I'm so happy for you!", "emotion": "bright", "score": 0.9}}

            Now generate your response in JSON format:""".strip()
    
    # ------------------ 规则匹配情绪 ------------------
    def _rule_based_emotion(self, reply: str, input_emotion: str) -> Dict[str, Any]:
        """
        基于规则匹配情绪
        """
        logger.info("[RULE-BASED] 使用规则匹配情绪")
        
        reply_lower = reply.lower()
    
        emotion_keywords = {
            "bright": ["wonderful", "amazing", "great", "fantastic", "excellent", "awesome", 
                       "congratulations", "celebrate", "happy", "joy", "excited"],
            "warm": ["understand", "here for you", "support", "care", "comfort", "sorry to hear",
                     "with you", "feel", "empathize"],
            "soft": ["gentle", "calm", "peaceful", "relax", "breathe", "okay", "safe", "quiet"],
            "clear": ["yes", "sure", "of course", "certainly", "understand", "noted", "got it"],
            "husky": ["serious", "important", "critical", "warning", "careful", "urgent"],
            "melodious": ["interesting", "curious", "surprised", "unexpected", "wow", "really"],
            "low_pitched": ["deep", "profound", "significant", "meaningful", "thoughtful"]
        }
        
        emotion_scores = {}
        for emo, keywords in emotion_keywords.items():
            score = sum(1 for kw in keywords if kw in reply_lower)
            if score > 0:
                emotion_scores[emo] = score
        
        exclamation_count = reply.count("!")
        if exclamation_count >= 2:
            emotion_scores["bright"] = emotion_scores.get("bright", 0) + 2
        
        input_emotion_map = {
            "happy": "bright",
            "sad": "warm",
            "angry": "husky",
            "fearful": "soft",
            "neutral": "clear",
            "surprised": "melodious",
            "disgusted": "husky",
            "unknown": "soft"
        }
        
        if emotion_scores:
            matched_emotion = max(emotion_scores, key=emotion_scores.get)
            score = min(emotion_scores[matched_emotion] / 5.0 + 0.5, 1.0)
            logger.info(f"[RULE-BASED] 关键词匹配: {matched_emotion}, score: {score:.2f}")
            return {
                "emotion": matched_emotion,
                "emotion_score": round(score, 2)
            }
        
        fallback_emotion = input_emotion_map.get(str(input_emotion).lower(), "clear")
        logger.info(f"[RULE-BASED] : {fallback_emotion}")
        return {
            "emotion": fallback_emotion,
            "emotion_score": 0.6
        }
    
    # ------------------ 最终回复生成 ------------------
    def get_response(self, transcript, prosody_description, dialogue_history, knowledge, emotion):
               
        reply = None
        tone = None
        retry_count = 0
        max_retries=3

        while retry_count <= max_retries:
            try:
                if retry_count > 0:
                    logger.warning(f"[RETRY {retry_count}/{max_retries}] Reply 为空，重新生成...")
                else:
                    logger.info("单阶段生成：回复 + 情绪")

                logger.info("第一阶段：生成回复文本")
                
                if knowledge is not None:
                    user_input = self.build_prompt_knowledge(transcript, prosody_description, dialogue_history, knowledge)
                else:
                    user_input = self.build_prompt(transcript, prosody_description, dialogue_history)

                messages = [{"role": "system", "content": "You are an emotion-aware assistant."}]
                messages.append({"role": "user", "content": user_input})

                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

                # ===== 根据重试次数调整生成参数 =====
                # 第一次失败后，逐渐增加随机性
                temperature = 0.7 + (retry_count * 0.1)  # 0.7 -> 0.8 -> 0.9 -> 1.0
                temperature = min(temperature, 1.0)  # 不超过 1.0
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        num_beams=1,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.1 + (retry_count * 0.05), 
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )

                gen = outputs[0][inputs.input_ids.shape[-1]:]
                generated_text = self.tokenizer.decode(gen, skip_special_tokens=True)
                logger.info(f"[Stage 1] Generated reply: {generated_text}")

                del inputs, outputs, gen
                gc.collect()
                torch.cuda.empty_cache()

                # ===== 解析 JSON =====
                json_match = re.search(r'\{.*?\}', generated_text, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group()
                        logger.info(f"[JSON] Extracted: {json_str}")
                        
                        obj = json.loads(json_str)
                        
                        reply = obj.get("reply") or obj.get("response")
                        response_emotion = obj.get("emotion") or obj.get("emotions")
                        emotion_score = obj.get("score")
                        
                        # ===== 检查 reply 是否为空 =====
                        if reply and reply.strip():  # reply 不为空
                            if response_emotion:
                                tone = {
                                    "emotion": response_emotion,
                                    "emotion_score": float(emotion_score) if emotion_score else 0.7
                                }
                                logger.info(f"[SUCCESS] Reply: {reply[:50]}..., Tone: {tone}")
                                return reply, tone
                            else:
                                # 有 reply 但没有 emotion，使用规则匹配
                                logger.warning("[PARTIAL] 只解析到 reply，使用规则匹配")
                                tone = self._rule_based_emotion(reply, emotion)
                                return reply, tone
                        
                        else:  # reply 为空，需要重试
                            logger.warning(f"[EMPTY REPLY] Reply 为空: '{reply}'")
                            retry_count += 1
                            continue  # 重试
                    # ← 添加：JSON 解析异常处理
                    except json.JSONDecodeError as e:
                        logger.warning(f"[JSON PARSE FAILED] {e}")
                        retry_count += 1
                        continue
                    
                    except Exception as e:
                        logger.error(f"[JSON PROCESSING ERROR] {e}")
                        retry_count += 1
                        continue
                else:
                    # 没找到 JSON，但可能有纯文本
                    clean_text = re.sub(r'\{[^}]*\}', '', generated_text).strip()
                    if clean_text:
                        logger.warning("[NO JSON] 使用纯文本 + 规则匹配")
                        reply = clean_text
                        tone = self._rule_based_emotion(reply, emotion)
                        return reply, tone
                    else:
                        # 完全没有可用文本
                        logger.warning("[EMPTY OUTPUT] 生成内容为空")
                        retry_count += 1
                        continue  # 重试  
            except KeyboardInterrupt:
                logger.warning("生成被用户中断")
                raise
                
            except torch.cuda.OutOfMemoryError:
                logger.error("GPU 内存不足")
                torch.cuda.empty_cache()
                return "I'm sorry, I'm having trouble processing that right now.", {"emotion": "soft", "emotion_score": 0.5}
                
            except Exception as e:
                logger.error(f"生成失败: {e}", exc_info=True)
                retry_count += 1
                if retry_count > max_retries:
                    break
                continue


    def _map_tone_to_emotion(self, old_tone: str) -> Dict[str, Any]:
        """将旧的描述性 tone 映射到标准情绪"""
        try:
            tone_mapping = {
                "warm": "warm",
                "friendly": "bright",
                "calm": "soft",
                "excited": "melodious",
                "serious": "clear",
                "empathetic": "warm",
                "cheerful": "bright"
            }
            
            emotion = tone_mapping.get(old_tone.lower(), "clear")
            return {
                "emotion": emotion,
                "emotion_score": 0.7
            }
        except Exception as e:
            logger.error(f"映射 tone 失败: {e}")
            return {"emotion": "clear", "emotion_score": 0.5}

    def respond(
        self, 
        transcript: str, 
        prosody_description: str, 
        dialogue_history: List[Dict[str, Any]], 
        emotion: str
    ) -> Tuple[str, Dict[str, Any], Optional[List]]:
        try:
            logger.info(f"开始处理用户输入: {transcript[:50]}...")
            
            history_text = self.get_dialogue_history(dialogue_history)
            messages = self.parse_history_to_messages(history_text)

            knowledge = self.get_final_response(messages)
            reply, tone = self.get_response(transcript, prosody_description, history_text, knowledge, emotion)
            
            logger.info("处理完成")
            return reply, tone, knowledge
            
        except KeyboardInterrupt:
            logger.warning("用户中断了程序执行")
            raise  # 重新抛出以完全停止程序
            
        except Exception as e:
            logger.error(f"respond 方法发生错误: {e}", exc_info=True)
            
            # 返回降级响应
            fallback_reply = "I apologize, but I'm having some technical difficulties. Could you try again?"
            fallback_tone = {
                "emotion": tone_suggestions.get(str(emotion).lower(), "clear"),
                "emotion_score": 0.5
            }
            
            return fallback_reply, fallback_tone, None
    
    def revise(
        self,
        reply: str,
        tone: dict,
        verify_result: dict,
        transcript: str,
        prosody_description: str,
        dialogue_history: str,
        emotion: str,
    ) -> tuple:

        suggestions = verify_result.get("suggestions", {})
        issues = verify_result.get("issues", [])
        adopted = {}

        if verify_result.get("aligned", True):
            logger.info("[REVISE] Manager says aligned, no revision needed.")
            return reply, tone, adopted

        final_tone = tone.copy()

        suggested_emotion = suggestions.get("emotion_category")
        if suggested_emotion and suggested_emotion in STANDARD_EMOTIONS:
            if suggested_emotion != tone.get("emotion"):
                logger.info(f"[REVISE] Adopt emotion_category: {tone['emotion']} → {suggested_emotion}")
                final_tone["emotion"] = suggested_emotion
                adopted["emotion_category"] = suggested_emotion

        suggested_score = suggestions.get("emotion_score")
        if suggested_score is not None:
            current_score = tone.get("emotion_score", 0.7)
            if abs(float(suggested_score) - float(current_score)) > 0.2:
                logger.info(f"[REVISE] Adopt emotion_score: {current_score} → {suggested_score}")
                final_tone["emotion_score"] = float(suggested_score)
                adopted["emotion_score"] = suggested_score

        strategy_hint = suggestions.get("interaction_strategy")
        final_reply = reply
        if strategy_hint:
            strategy_mismatch = any(
                kw in " ".join(issues).lower()
                for kw in ["strategy", "advice", "comfort", "validate", "tone"]
            )
            if strategy_mismatch:
                logger.info(f"[REVISE] Adopt strategy hint, regenerating reply: {strategy_hint}")
                revised_reply, revised_tone = self._regenerate_with_hint(
                    transcript, prosody_description, dialogue_history,
                    strategy_hint, final_tone, emotion
                )
                if revised_reply:
                    final_reply = revised_reply
                    final_tone = revised_tone
                    adopted["interaction_strategy"] = strategy_hint
            else:
                logger.info(f"[REVISE] Strategy hint received but no clear mismatch, skipping regeneration.")

        return final_reply, final_tone, adopted


    def _regenerate_with_hint(
        self,
        transcript: str,
        prosody_description: str,
        dialogue_history: str,
        strategy_hint: str,
        tone: dict,
        emotion: str,
    ) -> tuple:
        """Regenerate reply with strategy hint injected into prompt."""
        prompt = f"""
    You are a warm, empathetic assistant.

    Dialogue History:
    {dialogue_history.strip()}

    User said: "{transcript}"
    User's Prosody: {prosody_description}

    Adjustment required: {strategy_hint}
    Target tone: {tone}

    Generate a short revised response (≤100 words) following the adjustment.
    Output ONLY valid JSON:
    {{"reply": "...", "emotion": "{tone.get('emotion', 'warm')}", "score": {tone.get('emotion_score', 0.7)}}}
    """.strip()

        messages = [
            {"role": "system", "content": "You are an emotion-aware assistant."},
            {"role": "user", "content": prompt},
        ]

        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            gen = outputs[0][inputs.input_ids.shape[-1]:]
            generated = self.tokenizer.decode(gen, skip_special_tokens=True)


            match = re.search(r'\{.*?\}', generated, re.DOTALL)
            if match:
                obj = json.loads(match.group())
                return obj.get("reply", ""), {
                    "emotion": obj.get("emotion", tone.get("emotion")),
                    "emotion_score": float(obj.get("score", tone.get("emotion_score", 0.7))),
                }
        except Exception as e:
            logger.error(f"[REGENERATE] failed: {e}")

        return None, tone