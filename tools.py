from langchain.tools import tool
import json
from datetime import datetime
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from typing import List, Dict, Any

# --- Helper function to initialize/load the DB (Connecting to Qdrant Docker) ---
def initialize_guidelines_db():
    # 1. Use Ollama for local embeddings
    # NOTE: Ensure Ollama is running and accessible on port 11434
    embeddings = OllamaEmbeddings(model="llama3:8b", base_url="http://host.docker.internal:11434")
    
    # 2. Initialize Qdrant client to connect to the Docker container
    # NOTE: Qdrant Docker must be running and accessible on port 6333
    client = QdrantClient(url="http://host.docker.internal:6333") 
    COLLECTION_NAME = "clinical_guidelines"
    
    # 3. Mock Data: Simulate complex clinical guidelines
    guideline_text = [
        "Asthma Protocol: For new cough and wheezing in adults, first suggest a chest X-ray and spirometry. Prescribe Albuterol PRN. Do not prescribe beta-blockers.",
        "Diabetes Type 2: If blood sugar > 200 mg/dL, the primary suggestion is Metformin 500mg BID. Order HbA1c test and LFT.",
        "Common Cold: Symptoms lasting less than 5 days do not require antibiotics. Suggest Vitamin C and rest. Consider a viral panel only if fever is recurrent.",
    ]
    
    # 4. Create or reuse the Vector Store
    qdrant = Qdrant(client=client, embeddings=embeddings, collection_name=COLLECTION_NAME)
    
    # Check if the collection needs data population
    try:
        count = client.count(collection_name=COLLECTION_NAME, exact=True).count
        if count == 0:
             print("[RAG Tool] Populating Qdrant with mock guidelines...")
             qdrant.add_texts(guideline_text)
             print(f"[RAG Tool] Qdrant population complete. {len(guideline_text)} points added.")
    except Exception as e:
        # Handle case where collection doesn't exist yet (Qdrant creates it on first add)
        print(f"[RAG Tool] Initializing collection {COLLECTION_NAME}...")
        qdrant.add_texts(guideline_text)
        
    return qdrant

# --- Tool 1: Data Fetch (Read Access) ---
@tool
def tool_fetch_fhir(patient_abha_id: str, consent_artifact: str) -> str:
    """
    Simulates fetching a bundle of patient data (Allergies, Meds) via the
    ABDM HIE using a valid Consent Artifact. Returns a string summary.
    """
    print(f"\n[Tool Call] Securely fetching FHIR data for ABHA ID: {patient_abha_id[:5]}...")
    
    # Mock data relevant to the Safety Agent (Agent 5)
    mock_allergies = ["Penicillin", "Sulfonamides (SFA)", "Aspirin"]
    mock_medications = ["Metformin", "Lisinopril"]
    
    return json.dumps({
        "patient_id": patient_abha_id,
        "allergies": mock_allergies,
        "current_medications": mock_medications
    })

# --- Tool 2: RAG Query Tool (Guideline Access) ---
@tool
def tool_query_guidelines(query: str) -> str:
    """
    Queries the local Clinical Guidelines Vector Database (Qdrant) for evidence
    to support diagnostic tests or treatment suggestions.
    """
    qdrant_db = initialize_guidelines_db()
    
    # Search the vector store
    results = qdrant_db.similarity_search(query, k=2)
    
    # Format results for the LLM to consume
    context = "\n".join([doc.page_content for doc in results])
    
    print(f"\n[Tool Call] RAG Query for '{query[:20]}...' returned context.")
    return context

# --- Tool 3: Order Test (Write Access) ---
@tool
def tool_order_test(test_name: str, justification: str) -> str:
    """Simulates sending a FHIR ServiceRequest to the hospital's LIS/MCP system."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[Tool Call] Order Test Success at {timestamp}: {test_name}")
    print(f"   -> Justification: {justification}")
    return f"SUCCESS: Test '{test_name}' ordered via MCP."

# --- Tool 4: EMR Update (Final Write Access) ---
@tool
def tool_update_emr(fhir_payload_json: str, resource_type: str) -> str:
    """Simulates a secure FHIR POST (WRITE) to the hospital's EMR/HIP Gateway."""
    
    print("\n[Tool Call] EMR Update Initiated via Secure FHIR POST...")
    print(f"   -> Resource Type: {resource_type}")
    print(f"   -> Status: Successfully committed final record to HMIS/ABDM at {datetime.now().strftime('%H:%M:%S')}")
    
    return f"SUCCESS: FHIR {resource_type} written to EMR."