from ..state import AgentState
from ..tools import rag_query_tool


def planner_node(state: AgentState) -> AgentState:
    """
    Uses RAG over guidelines (Qdrant) to suggest tests.
    """
    note = state.note_summary or state.raw_transcript or "no symptoms text"
    symptoms_text = ", ".join(state.symptoms) if state.symptoms else "unspecified symptoms"
    query = f"Suggest initial basic tests for patient with: {symptoms_text}. Note: {note}"

    hits = rag_query_tool(query, top_k=3)

    suggested_tests: list[str] = []
    for h in hits:
        text = h["text"].lower()
        for candidate in ["cbc", "ecg", "urine", "lipid"]:
            if candidate in text and candidate.upper() not in suggested_tests:
                suggested_tests.append(candidate.upper())

    state.guideline_hits = hits
    state.suggested_tests = suggested_tests
    state.audit_log.append(f"Planner node: suggested tests {suggested_tests}.")
    return state
