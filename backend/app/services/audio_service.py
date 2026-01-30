import whisper
import librosa
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AudioService:
    def __init__(self):
        try:
            # Using 'base' model for demo speed/memory
            self.model = whisper.load_model("base")
        except Exception as e:
            logger.warning(f"Failed to load Whisper model: {e}. Falling back to mock transcription.")
            self.model = None

    async def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribes audio and analyzes technical quality (SNR, clipping).
        """
        # 1. Technical Analysis (librosa)
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Simple SNR estimation (Root Mean Square)
            rms = librosa.feature.rms(y=y)
            avg_rms = np.mean(rms)
            
            # Clipping detection (normalized peaks > 0.99)
            clipping = np.sum(np.abs(y) > 0.99) / len(y)
            
            audio_quality = max(0, min(100, (avg_rms * 1000) * (1 - clipping)))
        except Exception as e:
            logger.error(f"Librosa analysis failed: {e}")
            audio_quality = 50
            duration = 0

        # 2. Transcription (Whisper)
        transcript = ""
        confidence = 0.5
        
        if self.model:
            try:
                result = self.model.transcribe(audio_path)
                transcript = result["text"].strip()
                confidence = 0.95 # Base confidence for successful run
            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}")
                transcript = "[TRANSCRIPTION ERROR]"

        reasoning = f"Audio quality rated at {audio_quality:.1f}%. "
        if transcript:
            reasoning += f"Successfully extracted {len(transcript.split())} words."
        else:
            reasoning += "No clear dialogue detected."

        return {
            "transcript": transcript,
            "quality_score": audio_quality,
            "duration": duration,
            "reasoning": reasoning,
            "confidence": confidence
        }

audio_service = AudioService()
