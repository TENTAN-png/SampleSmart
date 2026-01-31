import asyncio
import logging
from typing import List, Dict, Any, Callable
from app.models import database as models
from app.db.session import SessionLocal

# Import real services
from app.services.cv_service import cv_service
from app.services.audio_service import audio_service
from app.services.nlp_service import nlp_service
from app.services.scoring_service import scoring_service
from app.services.intent_embedding_service import intent_embedding_service
from app.services.semantic_search_service import semantic_search_service
from sqlalchemy.orm.attributes import flag_modified

logger = logging.getLogger(__name__)

class ProcessingStage:
    def __init__(self, name: str, func: Callable, weight: float = 1.0):
        self.name = name
        self.func = func
        self.weight = weight

class ProcessingOrchestrator:
    def __init__(self):
        self.stages: List[ProcessingStage] = [
            ProcessingStage("Frame & Data Analysis", self._run_cv_analysis, weight=2.0),
            ProcessingStage("Audio Processing", self._run_audio_analysis, weight=2.0),
            ProcessingStage("Script Alignment", self._run_nlp_alignment, weight=1.0),
            ProcessingStage("Intelligence Scoring", self._run_scoring, weight=0.5),
            ProcessingStage("Intent Indexing", self._run_intent_indexing, weight=0.5)
        ]
        self._progress: Dict[int, Dict[str, Any]] = {}

    async def get_status(self, take_id: int) -> Dict[str, Any]:
        take_id = int(take_id)
        # Return live progress if currently processing
        if take_id in self._progress:
            logger.info(f"Returning live status for take {take_id}")
            return self._progress[take_id]
        
        # Otherwise, try to load final result from database
        db = SessionLocal()
        try:
            take = db.query(models.Take).get(take_id)
            if not take:
                return {"status": "not_found", "progress": 0}
            
            # If it has metadata, it's completed
            if take.ai_metadata:
                return {
                    "status": "completed",
                    "progress": 100,
                    "stages": {
                        "Frame & Data Analysis": "completed",
                        "Audio Processing": "completed",
                        "Script Alignment": "completed",
                        "Intelligence Scoring": "completed",
                        "Intent Indexing": "completed"
                    },
                    "cv": take.ai_metadata.get("cv", {}),
                    "audio": take.ai_metadata.get("audio", {}),
                    "nlp": take.ai_metadata.get("nlp", {}),
                    "logs": [
                        "Task loaded from database storage.",
                        f"Previous analysis for Take {take_id} restored."
                    ]
                }
            
            return {"status": "pending", "progress": 0}
        finally:
            db.close()

    async def process_take(self, take_id: int):
        take_id = int(take_id)
        # Prevent parallel processing of the same take
        if take_id in self._progress and self._progress[take_id]["status"] == "processing":
            logger.warning(f"Take {take_id} is already being processed. Skipping.")
            return

        self._progress[take_id] = {
            "status": "processing",
            "progress": 0,
            "current_stage": None,
            "stages": {s.name: "pending" for s in self.stages},
            "logs": [f"Initiating processing for Take {take_id}"]
        }
        
        db = SessionLocal()
        take = db.query(models.Take).get(take_id)
        if not take:
            self._progress[take_id]["status"] = "error"
            return

        # Context for script
        target_script = "I told you we shouldn't have come here, Marcus. The perimeter is compromised."

        try:
            total_weight = sum(s.weight for s in self.stages)
            current_weight = 0
            
            context = {}

            for stage in self.stages:
                self._progress[take_id]["current_stage"] = stage.name
                self._progress[take_id]["stages"][stage.name] = "running"
                self._progress[take_id]["logs"].append(f"[{take_id}] Starting {stage.name}...")
                
                # Execute stage logic
                if stage.name == "Script Alignment":
                    result = await stage.func(take, db, context.get("transcript", ""), target_script)
                elif stage.name == "Intelligence Scoring":
                    result = await stage.func(take, db, context)
                elif stage.name == "Intent Indexing":
                    result = await stage.func(take, db, context)
                else:
                    result = await stage.func(take, db)
                
                context.update(result if isinstance(result, dict) else {})

                self._progress[take_id]["stages"][stage.name] = "completed"
                current_weight += stage.weight
                self._progress[take_id]["progress"] = int((current_weight / total_weight) * 100)
                
                db.commit()

            self._progress[take_id]["status"] = "completed"
            self._progress[take_id]["progress"] = 100
            self._progress[take_id]["current_stage"] = None

        except Exception as e:
            logger.error(f"Error processing take {take_id}: {str(e)}")
            self._progress[take_id]["status"] = "error"
            self._progress[take_id]["logs"].append(f"ERROR: {str(e)}")
        finally:
            db.close()

    async def _run_cv_analysis(self, take: models.Take, db):
        res = await cv_service.analyze_video(take.file_path)
        take.duration = res["duration"]
        take.ai_metadata["cv"] = res
        take.ai_reasoning["cv"] = res["reasoning"]
        return res

    async def _run_audio_analysis(self, take: models.Take, db):
        res = await audio_service.analyze_audio(take.file_path)
        take.ai_metadata["audio"] = res
        take.ai_reasoning["audio"] = res["reasoning"]
        return res

    async def _run_nlp_alignment(self, take: models.Take, db, transcript, script):
        res = await nlp_service.align_script(transcript, script)
        take.ai_metadata["nlp"] = res
        take.ai_reasoning["nlp"] = res["reasoning"]
        return res

    async def _run_scoring(self, take: models.Take, db, context):
        res = scoring_service.compute_take_score(
            context.get("cv", {}),
            context.get("audio", {}),
            context.get("nlp", {})
        )
        
        # Finalize results
        take.confidence_score = res.get("total_score", 0)
        take.ai_reasoning["summary"] = res.get("summary", "")
        take.ai_reasoning["breakdown"] = res.get("breakdown", {})
        take.ai_metadata["score_breakdown"] = res.get("breakdown", {})
        
        # Ensure SQLAlchemy detects the change in JSON fields
        flag_modified(take, "ai_metadata")
        flag_modified(take, "ai_reasoning")
        
        db.commit()
        return res

    async def _run_intent_indexing(self, take: models.Take, db, context):
        """Generate intent embeddings for semantic search."""
        self._progress[take.id]["logs"].append(f"[{take.id}] Starting Intent Indexing...")
        try:
            # Extract data from context
            transcript = context.get("transcript", "")
            cv_data = context.get("cv", {})
            audio_data = context.get("audio", {})
            
            self._progress[take.id]["logs"].append(f"[{take.id}] Building multimodal context description...")
            
            # Determine emotion from CV analysis
            emotion_label = "neutral"
            if cv_data.get("objects_detected"):
                # Simple emotion detection based on context
                emotion_label = "thoughtful"
            
            # Build audio features for timing
            audio_features = {
                "has_pause_before": False,
                "pause_before_duration": 0,
                "speech_rate": audio_data.get("quality", {}).get("speech_rate", 150)
            }
            
            # Timing patterns
            timing_data = {
                "pattern": "normal",
                "reaction_delay": 0
            }
            
            self._progress[take.id]["logs"].append(f"[{take.id}] Detected primary intent: {emotion_label}")
            self._progress[take.id]["logs"].append(f"[{take.id}] Generating semantic embedding vectors...")
            
            # Generate embedding for the entire take as a single moment
            embedding = intent_embedding_service.generate_moment_embedding(
                transcript_snippet=transcript[:200] if transcript else "",
                emotion_data={"primary_emotion": emotion_label, "intensity": 60},
                audio_features=audio_features,
                timing_data=timing_data,
                script_context=""
            )
            
            self._progress[take.id]["logs"].append(f"[{take.id}] Moment embedding generated successfully.")
            self._progress[take.id]["logs"].append(f"[{take.id}] Adding moment to FAISS similarity index...")
            
            # Add to search index
            moment_id = take.id * 1000  # Simple moment ID
            semantic_search_service.index_moment(
                moment_id=moment_id,
                take_id=take.id,
                start_time=0,
                end_time=take.duration or 10,
                embedding=embedding,
                transcript_snippet=transcript[:200] if transcript else "",
                emotion_label=emotion_label,
                audio_features=audio_features,
                timing_data=timing_data
            )
            
            self._progress[take.id]["logs"].append(f"[{take.id}] Saving FAISS index to persistent storage...")
            # Save index
            semantic_search_service.save_index()
            
            self._progress[take.id]["logs"].append(f"[{take.id}] Intent indexing and search integration complete!")
            logger.info(f"Indexed take {take.id} for semantic search")
            return {"indexed": True, "moment_id": moment_id}
            
        except Exception as e:
            msg = f"Intent indexing failed: {str(e)}"
            self._progress[take.id]["logs"].append(f"ERROR: {msg}")
            logger.warning(f"Intent indexing failed for take {take.id}: {e}")
            return {"indexed": False, "error": msg}

orchestrator = ProcessingOrchestrator()
