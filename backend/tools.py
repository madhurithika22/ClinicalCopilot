from typing import Dict, Any, List

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import whisper
from pathlib import Path
import json
from datetime import datetime
# ---------------------------
# Qdrant & Embedding Setup
# ---------------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "clinical_guidelines"

# Create global clients (loaded once)
_qdrant_client = QdrantClient(url=QDRANT_URL)
_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_whisper_model = whisper.load_model("base")

def rag_query_tool(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieves guideline-based suggestions from Qdrant.
    If Qdrant or the client is misconfigured, it fails gracefully and returns [].
    """
    try:
        vec = _embed_model.encode(query)

        # Not all qdrant-client versions have the same API.
        # Safest: only call search() if it exists.
        if hasattr(_qdrant_client, "search"):
            results = _qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vec,
                limit=top_k,
            )
        else:
            # Older client: skip actual search to avoid crashes in your demo
            print("⚠️ QdrantClient.search() not available in this version. Returning [].")
            return []

        formatted: List[Dict[str, Any]] = []
        for r in results:
            payload = r.payload
            formatted.append(
                {
                    "title": payload.get("title"),
                    "text": payload.get("text"),
                    "tags": payload.get("tags"),
                    "score": r.score,
                }
            )
        return formatted

    except Exception as e:
        # Do NOT crash the whole API – just log and return no hits.
        print("🔥 RAG / Qdrant error:", repr(e))
        return []


# ---------------------------
# Mock Action Tools
# ---------------------------

def tool_order_test(test_name: str) -> Dict[str, Any]:
    """
    Mock test ordering tool.
    In real deployment, this would call a lab system via MCP/FHIR.
    """
    return {
        "action": "order_test",
        "test_name": test_name,
        "status": "ordered_mock",
        "order_id": f"LAB-MOCK-{test_name.upper()}",
    }

EMR_STORE_PATH = Path(__file__).parent / "emr_store.json"
def tool_update_emr(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mock EMR update tool.
    Now also PERSISTS records to a local JSON file (emr_store.json)
    so you can prove that data is stored.
    """
    record = {
        "emr_record_id": f"EMR-{int(datetime.utcnow().timestamp())}",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        **payload,
    }

    # Load existing EMR records if file exists
    records: list[Dict[str, Any]] = []
    if EMR_STORE_PATH.exists():
        try:
            with EMR_STORE_PATH.open("r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    # Append new record
    records.append(record)

    # Save back to file
    with EMR_STORE_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return {
        "action": "update_emr",
        "status": "success_mock",
        "emr_record_id": record["emr_record_id"],
    }

try:
    import whisper
    _whisper_model = whisper.load_model("tiny")  # you can try "base" later if laptop can handle it
    print("✅ Whisper STT model loaded (tiny).")
except Exception as e:
    _whisper_model = None
    print("⚠️ Whisper not available. STT will only return a placeholder. Error:", repr(e))


def tool_transcribe_voice(path: str) -> str:
    """
    Speech-to-text tool.
    - If Whisper works: return the actual transcript.
    - If Whisper fails: return an explicit error sentence.
    """
    if _whisper_model is None:
        print("⚠️ Whisper model not loaded.")
        return "STT is not available (Whisper model not loaded)."

    try:
        result = _whisper_model.transcribe(path)
        text = (result.get("text") or "").strip()
        print("🗣️ Whisper transcription:", text)
        if not text:
            return "Transcription empty (no speech or decoding issue)."
        return text
    except Exception as e:
        print("🔥 Whisper STT error:", repr(e))
        return "Transcription failed due to an internal error."