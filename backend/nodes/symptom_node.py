# backend/nodes/symptom_node.py

from ..state import AgentState

"""
Symptom Node

Goal:
- Read free-text note (from scribe_node / transcript)
- Map natural language phrases into a clean, canonical
  symptom list that the planner can use.

The canonical symptoms MUST match the keys in STANDARD_TESTS
in planner_node.py:
    - "chest pain"
    - "shortness of breath"
    - "fever"
    - "cough"
    - "headache"
    - "vomiting"
    - "abdominal pain"
    - "diabetes"
    - "hypertension"
"""

# Map of phrases (what doctor/patient might say) -> canonical symptom key
SYMPTOM_KEYWORDS = {
    # Chest pain cluster
    "chest pain": "chest pain",
    "heaviness in chest": "chest pain",
    "tightness in chest": "chest pain",
    "pressure in chest": "chest pain",

    # Shortness of breath cluster
    "shortness of breath": "shortness of breath",
    "breathlessness": "shortness of breath",
    "breathless": "shortness of breath",
    "difficulty breathing": "shortness of breath",

    # Fever
    "fever": "fever",
    "high temperature": "fever",
    "raised temperature": "fever",

    # Cough
    "cough": "cough",
    "coughing": "cough",

    # Headache
    "headache": "headache",
    "pain in head": "headache",
    "migraine": "headache",

    # Vomiting / nausea
    "vomiting": "vomiting",
    "vomit": "vomiting",
    "threw up": "vomiting",
    "nausea": "vomiting",
    "feeling like vomiting": "vomiting",

    # Abdominal pain
    "abdominal pain": "abdominal pain",
    "stomach pain": "abdominal pain",
    "pain in stomach": "abdominal pain",
    "tummy pain": "abdominal pain",
    "gastric pain": "abdominal pain",

    # Diabetes (history / known case)
    "diabetes": "diabetes",
    "type 2 diabetes": "diabetes",
    "type ii diabetes": "diabetes",
    "high blood sugar": "diabetes",
    "sugar patient": "diabetes",

    # Hypertension
    "hypertension": "hypertension",
    "high blood pressure": "hypertension",
    "bp is high": "hypertension",
}


def symptom_node(state: AgentState) -> AgentState:
    """
    Extracts a canonical list of symptoms from the note text.

    Input:
        - state.note_summary (preferred)
        - or state.raw_transcript

    Output:
        - state.symptoms: List[str] with canonical labels matching STANDARD_TESTS keys
        - audit_log updated
    """
    text = (state.note_summary or state.raw_transcript or "").lower()
    found: list[str] = []

    for phrase, canonical in SYMPTOM_KEYWORDS.items():
        if phrase in text:
            found.append(canonical)

    # Merge with any existing symptoms, keep unique / ordered
    merged = list(dict.fromkeys(state.symptoms + found))
    state.symptoms = merged

    state.audit_log.append(f"Symptom node: extracted symptoms {state.symptoms}.")
    return state
