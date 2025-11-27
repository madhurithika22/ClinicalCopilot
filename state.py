from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph

# --- Core FHIR/ABDM IDs ---
class AccessData(TypedDict):
    """IDs and tokens received from the Biometric/ABDM trigger."""
    patient_abha_id: Optional[str]
    hospital_facility_id: Optional[str]
    consent_artifact: Optional[str]

# --- Structured Clinical Outputs ---
class ClinicalData(TypedDict):
    """The main clinical payload updated by the agents."""
    
    # Updated by Agent 2 (Scribe)
    raw_transcript: Optional[str]
    soap_draft: Optional[str]
    extracted_symptoms: Optional[List[str]] 
    
    # Pulled by Agent 1 (Data Fetch)
    fhir_allergies: Optional[List[str]] # The reference data for Agent 5 (Safety)
    fhir_medications: Optional[List[str]]
    
    # Updated by Agent 4 (Prescribing)
    rx_draft_fhir_json: Optional[str]  # FHIR MedicationRequest JSON
    
    # Updated by Agent 3 (Planner/CDS)
    suggested_tests_rag: Optional[str] # RAG context/suggestion for doctor
    
    # Updated by Agent 5 (Safety Check)
    safety_alert: Optional[str] # Set to "ALERT" or None
    
# --- The Central State Object for LangGraph ---
class AgentState(TypedDict):
    """
    The state dictionary passed between all nodes in the LangGraph.
    """
    # 1. Access and Control Context
    access_context: AccessData
    
    # 2. Clinical Data Payload
    clinical_data: ClinicalData
    
    # 3. Control flow (used by conditional edges)
    workflow_status: str # e.g., "NEEDS_PRESCRIPTION", "REVIEW_REQUIRED", "PENDING_UPDATE"