import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Optional ML imports
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

class AudioService:
    def __init__(self):
        self.model = None
        if WHISPER_AVAILABLE:
            try:
                self.model = whisper.load_model("base")
            except Exception as e:
                logger.warning(f"Failed to load Whisper model: {e}. Using mock transcription.")
        else:
            logger.info("Whisper not available. Using mock audio analysis.")

    async def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribes audio and analyzes technical quality (SNR, clipping).
        Returns mock data when ML dependencies are not available.
        """
        # Return mock data if librosa not available
        if not LIBROSA_AVAILABLE:
            logger.info("Librosa not available, returning mock audio data")
            return {
                "transcript": "Mock transcript: Hello, this is a test dialogue.",
                "quality_score": 85.0,
                "duration": 30.0,
                "reasoning": "Mock audio analysis (librosa not installed)",
                "confidence": 0.7
            }
        
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

