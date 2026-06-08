# manager.py
import json
from typing import Dict, Any
from openai import OpenAI
import re

class SpeechDescriptionManager:
    
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-4"):
        """
        Initialize the manager with OpenAI client
        
        Args:
            api_key: Your OpenAI API key
            base_url: Custom API base URL (optional, for API proxies or custom endpoints)
            model: Model name 
        """
        self.model = model
        
        # Initialize OpenAI client (new SDK v1.0+)
        if base_url:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        else:
            self.client = OpenAI(api_key=api_key)

        self.verifier = ResponseVerifier(client=self.client, model=model)
    
    def verify_response(
        self,
        prosody_description: str,
        reply: str,
        tone: dict,
        user_emotion: str,
    ) -> dict:
        """
        Post-hoc verification of Responder's output.
        Returns verification result with optional revision suggestions.
        """
        return self.verifier.verify(prosody_description, reply, tone, user_emotion)
    
    def generate_description_v2(self, perceiver_output: Dict[str, Any]) -> str:
        """Generate with examples"""
        
        interpreted = self._interpret_attributes(perceiver_output)
        
        prompt = f"""Generate a natural description of the speaker's prosody.
            Example 1:
            Input: Emotion: sad (0.82), Tone: low, Rhythm: slow (2.1/s), Pauses: frequent (42%), Volume: soft/steady, Certainty: hesitant (0.28)
            Output: The speaker sounds deeply sad (0.82), speaking slowly and hesitantly (0.28) with frequent pauses. Their voice is soft and low, suggesting they're struggling to express difficult feelings.

            Example 2:
            Input: Emotion: excited (0.91), Tone: energetic, Rhythm: fast (6.2/s), Pauses: few (8%), Volume: loud/fluctuating, Certainty: very certain (0.89)
            Output: The speaker is highly excited (0.91), speaking rapidly with barely any pauses. Their energetic, fluctuating voice conveys strong enthusiasm and certainty (0.89).

            Now generate:
            Transcript: "{interpreted['transcript']}"
            Emotion: {interpreted['emotion']}
            Tone: {interpreted['tone']}
            Rhythm: {interpreted['rhythm']}
            Pauses: {interpreted['pause']}
            Volume: {interpreted['energy']}
            Certainty: {interpreted['certainty']}

            Output (natural paragraph):"""
        
        # 3. Call GPT using new SDK
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You generate natural, flowing descriptions of speech prosody."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()

    
    def _interpret_attributes(self, perceiver_output: Dict[str, Any]) -> Dict[str, str]:
        """Convert numerical attributes to descriptive text"""
        
        attrs = perceiver_output.get("attributes", {})
        transcript = perceiver_output.get("transcript", "")
        
        # Emotion
        emotion = attrs.get("emotion", "unknown")
        emotion_score = attrs.get("emotion_score", 0.0)
        emotion_desc = f"{self._translate_emotion(emotion)} (intensity {emotion_score:.2f})"
        
        # Tone (comprehensive judgment)
        tone = self._infer_tone(attrs)
        
        # Rhythm
        speaking_rate = attrs.get("speaking_rate", 0.0)
        if speaking_rate < 2.5:
            rhythm = "slow"
        elif speaking_rate < 5.0:
            rhythm = "normal"
        else:
            rhythm = "fast"
        rhythm_desc = f"{rhythm} ({speaking_rate:.1f} words/sec)"
        
        # Pauses
        pause_ratio = attrs.get("pause_ratio", 0.0)
        if pause_ratio < 0.1:
            pause = "almost no pauses"
        elif pause_ratio < 0.25:
            pause = "few pauses"
        elif pause_ratio < 0.4:
            pause = "moderate pauses"
        else:
            pause = "frequent pauses"
        pause_desc = f"{pause} ({pause_ratio:.0%})"
        
        # Volume
        energy_mean = attrs.get("energy_mean", 0.0)
        energy_std = attrs.get("energy_std", 0.0)
        
        if energy_mean < 0.025:
            volume = "soft"
        elif energy_mean < 0.05:
            volume = "normal"
        else:
            volume = "loud"
        
        if energy_std < 0.01:
            variation = "steady"
        elif energy_std < 0.02:
            variation = "slight fluctuation"
        else:
            variation = "significant fluctuation"
        
        energy_desc = f"{volume}, {variation}"
        
        # Certainty
        certainty = attrs.get("certainty", 0.0)
        if certainty < 0.4:
            certainty_desc = "hesitant"
        elif certainty < 0.6:
            certainty_desc = "somewhat uncertain"
        elif certainty < 0.8:
            certainty_desc = "fairly certain"
        else:
            certainty_desc = "very certain"
        certainty_desc = f"{certainty_desc} ({certainty:.2f})"
        
        return {
            "transcript": transcript,
            "emotion": emotion_desc,
            "tone": tone,
            "rhythm": rhythm_desc,
            "pause": pause_desc,
            "energy": energy_desc,
            "certainty": certainty_desc
        }
    
    def _translate_emotion(self, emotion: str) -> str:
        """Emotion translation"""
        emotion_map = {
            "happy": "happy",
            "sad": "sad",
            "angry": "angry",
            "neutral": "neutral",
            "fear": "fearful",
            "disgust": "disgusted",
            "surprise": "surprised"
        }
        return emotion_map.get(emotion.lower(), emotion)
    
    def _infer_tone(self, attrs: Dict[str, Any]) -> str:
        """Infer tone (based on emotion, volume, variation)"""
        emotion = attrs.get("emotion", "neutral").lower()
        energy_mean = attrs.get("energy_mean", 0.0)
        energy_std = attrs.get("energy_std", 0.0)
        
        # Simple rule-based inference
        if emotion == "happy":
            if energy_std > 0.015:
                return "cheerful and lively"
            else:
                return "calm and pleasant"
        elif emotion == "sad":
            return "low and depressed"
        elif emotion == "angry":
            if energy_mean > 0.05:
                return "agitated and furious"
            else:
                return "suppressed anger"
        elif emotion == "neutral":
            if energy_mean < 0.03:
                return "flat"
            else:
                return "calm"
        else:
            return "normal"

class ResponseVerifier:
    """
    Post-hoc verification module.
    Only produces alignment issues and suggestions — does NOT rewrite the reply.
    The Responder decides whether to adopt the suggestions.
    """

    def __init__(self, client: OpenAI, model: str = "gpt-3.5-turbo"):
        self.client = client
        self.model = model

    def verify(
        self,
        prosody_description: str,
        reply: str,
        tone: dict,
        user_emotion: str,
    ) -> dict:
        """
        Returns:
            {
                "aligned": bool,
                "issues": [str, ...],
                "suggestions": {
                    "emotion_category": str | None,    # e.g. "switch to warm"
                    "emotion_score": float | None,     # e.g. 0.8
                    "interaction_strategy": str | None # e.g. "focus on comfort, avoid advice"
                }
            }
        """
        prompt = self._build_prompt(prosody_description, reply, tone, user_emotion)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an alignment checker for empathetic dialogue. "
                        "You detect mismatches and give minimal, targeted suggestions. "
                        "You do NOT rewrite replies — only suggest what to adjust. "
                        "Output valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()
        return self._parse(raw)

    def _build_prompt(self, prosody_description, reply, tone, user_emotion):
        return f"""
            Check if this response aligns with the user's prosodic state.

            === User Prosody ===
            {prosody_description}

            === User Emotion ===
            {user_emotion}

            === Response to Check ===
            Reply: "{reply}"
            Tone: {tone}

            === Check These 3 Dimensions ===
            1. emotion_category: Does the tone emotion fit the user's state?
            2. emotion_score: Does the intensity reflect the prosodic weight described?
            3. interaction_strategy: Is the strategy appropriate (comfort / validate / encourage)?

            Output ONLY valid JSON:
            {{
            "aligned": true/false,
            "issues": ["brief description of each issue found"],
            "suggestions": {{
                "emotion_category": "suggested emotion name or null if fine",
                "emotion_score": 0.0 to 1.0 or null if fine,
                "interaction_strategy": "one sentence strategy hint or null if fine"
            }}
            }}

            - If fully aligned: aligned=true, issues=[], all suggestions=null
            - Do NOT write a new reply, only describe what should change
            """.strip()

    def _parse(self, raw: str) -> dict:
        
        default = {
            "aligned": True,
            "issues": [],
            "suggestions": {
                "emotion_category": None,
                "emotion_score": None,
                "interaction_strategy": None,
            },
        }
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                obj = json.loads(match.group())
                suggestions = obj.get("suggestions", {})
                return {
                    "aligned": obj.get("aligned", True),
                    "issues": obj.get("issues", []),
                    "suggestions": {
                        "emotion_category": suggestions.get("emotion_category"),
                        "emotion_score": suggestions.get("emotion_score"),
                        "interaction_strategy": suggestions.get("interaction_strategy"),
                    },
                }
        except Exception:
            pass
        return default
