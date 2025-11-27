# backend/nodes/symptom_node.py

from ..state import AgentState

# Very simple rule-based symptom extractor (demo friendly)
SYMPTOM_KEYWORDS = {
    "chest pain": "chest pain",
    "tightness in chest": "chest tightness",
    "breathless": "shortness of breath",
    "shortness of breath": "shortness of breath",
    "fever": "fever",
    "high temperature": "fever",
    "cough": "cough",
    "headache": "headache",
    "vomiting": "vomiting",
    "nausea": "nausea",
    "dizziness": "dizziness",
    "fatigue": "fatigue",
    "tired": "fatigue"
}


def symptom_node(state: AgentState) -> AgentState:
    """
    Extracts a simple list of symptoms from the note text.
    This runs AFTER scribe_node.
    """
    text = (state.note_summary or state.raw_transcript or "").lower()
    found: list[str] = []

    for phrase, label in SYMPTOM_KEYWORDS.items():
        if phrase in text:
            found.append(label)

    # merge with existing, keep unique order
    merged = list(dict.fromkeys(state.symptoms + found))
    state.symptoms = merged

    state.audit_log.append(f"Symptom node: extracted symptoms {state.symptoms}.")
    return state
