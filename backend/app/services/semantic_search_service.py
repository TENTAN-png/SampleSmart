"""
Semantic Search Service for SmartCut AI
FAISS-based vector indexing and intent-based retrieval.
"""
import numpy as np
import logging
import pickle
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from app.services.intent_embedding_service import intent_embedding_service

logger = logging.getLogger(__name__)

# Optional ML imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Using mock vector search.")


@dataclass
class SearchResult:
    """A single semantic search result with explainability."""
    result_id: int
    take_id: int
    moment_id: int
    start_time: float
    end_time: float
    confidence: float
    transcript_snippet: str
    emotion_label: str
    reasoning: Dict[str, Any]


class SemanticSearchService:
    """
    Vector-based semantic search using FAISS (or mock if unavailable).
    Indexes video moments and retrieves by intent similarity.
    """
    
    INDEX_PATH = "./storage/faiss_index.bin"
    METADATA_PATH = "./storage/faiss_metadata.pkl"
    
    def __init__(self):
        self.dimension = intent_embedding_service.EMBEDDING_DIM
        self.index: Any = None  # faiss.IndexFlatIP or MockIndex
        self.metadata: List[Dict] = []  # Parallel list of moment metadata
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        os.makedirs("./storage", exist_ok=True)
        
        if FAISS_AVAILABLE and os.path.exists(self.INDEX_PATH) and os.path.exists(self.METADATA_PATH):
            try:
                self.index = faiss.read_index(self.INDEX_PATH)
                with open(self.METADATA_PATH, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}. Creating new.")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index (or mock)."""
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            # Mock index structure associated with metadata list
            self.index = "MOCK_INDEX"
            
        self.metadata = []
        logger.info(f"Created new {'FAISS' if FAISS_AVAILABLE else 'MOCK'} index")
    
    def save_index(self):
        """Persist index to disk."""
        os.makedirs("./storage", exist_ok=True)
        if FAISS_AVAILABLE and self.index != "MOCK_INDEX":
            faiss.write_index(self.index, self.INDEX_PATH)
        
        with open(self.METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved index metadata with {len(self.metadata)} items")
    
    def index_moment(
        self,
        moment_id: int,
        take_id: int,
        start_time: float,
        end_time: float,
        embedding: np.ndarray,
        transcript_snippet: str = "",
        emotion_label: str = "neutral",
        audio_features: Dict = None,
        timing_data: Dict = None
    ):
        """
        Add a moment's embedding to the index.
        """
        # Ensure normalized for cosine similarity
        embedding = embedding.astype(np.float32)
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        # Add to FAISS or Mock
        if FAISS_AVAILABLE and self.index != "MOCK_INDEX":
            self.index.add(embedding.reshape(1, -1))
        
        # Store metadata (acts as source of truth for mock mode too)
        self.metadata.append({
            "moment_id": moment_id,
            "take_id": take_id,
            "start_time": start_time,
            "end_time": end_time,
            "transcript_snippet": transcript_snippet,
            "emotion_label": emotion_label,
            "audio_features": audio_features or {},
            "timing_data": timing_data or {}
        })
    
    def search_by_intent(
        self,
        query: str,
        top_k: int = 10,
        filters: Dict = None
    ) -> List[SearchResult]:
        """
        Search for moments matching the editor's intent query.
        """
        if not self.metadata:
            return []
            
        if FAISS_AVAILABLE and self.index != "MOCK_INDEX" and self.index.ntotal > 0:
            # Generate query embedding
            query_embedding = intent_embedding_service.embed_query(query)
            query_embedding = query_embedding.reshape(1, -1)
            
             # Search FAISS
            k = min(top_k * 3, self.index.ntotal)  # Over-fetch for filtering
            scores, indices = self.index.search(query_embedding, k)
            current_scores = scores[0]
            current_indices = indices[0]
        else:
            # Mock results using metadata directly
            import random
            current_indices = list(range(len(self.metadata)))
            random.shuffle(current_indices)
            current_indices = current_indices[:min(top_k * 3, len(self.metadata))]
            current_scores = [0.85 - (i * 0.05) for i in range(len(current_indices))] # Fake scores

        # Parse query intent for filtering and explainability
        parsed_intent = intent_embedding_service.parse_query_intent(query)
        
        results = []
        for i, idx in enumerate(current_indices):
            idx = int(idx)
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            score = float(current_scores[i])
            meta = self.metadata[idx]
            
            # Apply filters
            if filters:
                if filters.get("take_id") and meta["take_id"] != filters["take_id"]:
                    continue
                if filters.get("emotion") and meta["emotion_label"] != filters["emotion"]:
                    continue
            
            # Generate reasoning
            reasoning = self._generate_reasoning(query, parsed_intent, meta, score)
            
            results.append(SearchResult(
                result_id=idx,
                take_id=meta["take_id"],
                moment_id=meta["moment_id"],
                start_time=meta["start_time"],
                end_time=meta["end_time"],
                confidence=score,
                transcript_snippet=meta["transcript_snippet"],
                emotion_label=meta["emotion_label"],
                reasoning=reasoning
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def _generate_reasoning(
        self,
        query: str,
        parsed_intent: Dict,
        metadata: Dict,
        score: float
    ) -> Dict[str, Any]:
        """
        Generate human-readable reasoning for why a result matched.
        """
        matched_because = []
        
        # Emotion match
        emotion = metadata.get("emotion_label", "neutral")
        if parsed_intent["emotions"]:
            if emotion in parsed_intent["emotions"]:
                matched_because.append(f"Emotion matches: {emotion}")
            else:
                matched_because.append(f"Detected emotion: {emotion}")
        else:
            matched_because.append(f"Detected emotion: {emotion}")
        
        # Timing patterns
        timing = metadata.get("timing_data", {})
        audio = metadata.get("audio_features", {})
        
        if timing.get("pattern"):
            pattern = timing["pattern"].replace("_", " ")
            matched_because.append(f"Timing pattern: {pattern}")
        
        if audio.get("pause_before_duration", 0) > 0.5:
            duration = audio["pause_before_duration"]
            matched_because.append(f"{duration:.1f}s pause before speaking")
        
        if audio.get("pause_after_duration", 0) > 0.5:
            duration = audio["pause_after_duration"]
            matched_because.append(f"{duration:.1f}s pause after speaking")
        
        # Transcript relevance
        transcript = metadata.get("transcript_snippet", "")
        if transcript:
            matched_because.append(f"Dialogue: \"{transcript[:50]}...\"" if len(transcript) > 50 else f"Dialogue: \"{transcript}\"")
        else:
            matched_because.append("Silent moment / non-verbal reaction")
        
        return {
            "matched_because": matched_because,
            "emotion_detected": emotion,
            "timing_pattern": timing.get("pattern", "normal"),
            "confidence_score": round(score * 100, 1),
            "query_intent": parsed_intent
        }
    
    def get_suggestions(self, partial_query: str) -> List[str]:
        """
        Get query suggestions for autocomplete.
        """
        suggestions = [
            "hesitant reaction before answering",
            "tense pause before dialogue",
            "awkward silence after confession",
            "relieved smile after conflict",
            "angry interruption mid-sentence",
            "thoughtful pause while listening",
            "surprised reaction to news",
            "nervous laughter",
            "confident delivery",
            "emotional breakdown",
            "subtle facial reaction",
            "dramatic silence"
        ]
        
        partial_lower = partial_query.lower()
        return [s for s in suggestions if partial_lower in s.lower()][:5]
    
    def clear_index(self):
        """Clear all indexed data."""
        self._create_new_index()
        self.save_index()


# Singleton instance
semantic_search_service = SemanticSearchService()
