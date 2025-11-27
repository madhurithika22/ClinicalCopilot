from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from pathlib import Path

QDRANT_PATH = Path("qdrant_local")
client = QdrantClient(path=str(QDRANT_PATH))

GUIDELINE_COLLECTION = "clinical_guidelines"

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
GUIDELINES = [
    {
        "id": 1,
        "text": "Acute chest pain requires urgent ECG, serial cardiac troponins, chest X-ray, basic metabolic panel and CBC to rule out acute coronary syndrome and other causes.",
        "source": "Cardiology guideline",
    },
    {
        "id": 2,
        "text": "Shortness of breath with suspected heart failure should prompt chest X-ray, NT-proBNP, ECG, echocardiography and assessment of renal function and electrolytes.",
        "source": "Heart failure guideline",
    },
    {
        "id": 3,
        "text": "Patients with fever of unknown origin generally need CBC with differential, ESR, CRP, urine analysis, chest X-ray and targeted cultures based on suspected source.",
        "source": "Infectious disease guideline",
    },
    {
        "id": 4,
        "text": "For acute cough with systemic symptoms, consider CBC, chest X-ray and inflammatory markers such as CRP to differentiate bacterial from viral infection.",
        "source": "Respiratory infection guideline",
    },
    {
        "id": 5,
        "text": "Chronic cough may require chest X-ray, pulmonary function tests including spirometry, and sputum AFB smear in tuberculosis-endemic regions.",
        "source": "Chronic cough guideline",
    },
    {
        "id": 6,
        "text": "Chronic diabetes follow-up should include HbA1c every 3 to 6 months, fasting plasma glucose, lipid profile, renal function test and urine microalbuminuria.",
        "source": "Diabetes follow-up guideline",
    },
    {
        "id": 7,
        "text": "Hypertensive patients require regular monitoring of renal function, electrolytes, fasting lipid profile, ECG and periodic echocardiography if end-organ damage suspected.",
        "source": "Hypertension guideline",
    },
    {
        "id": 8,
        "text": "Uncomplicated urinary tract infection is typically diagnosed with urine routine and microscopy; urine culture is recommended in recurrent or complicated cases.",
        "source": "UTI guideline",
    },
    {
        "id": 9,
        "text": "Patients with syncope should undergo ECG, blood glucose, orthostatic blood pressure measurement and further cardiac evaluation including echocardiogram where indicated.",
        "source": "Syncope guideline",
    },
    {
        "id": 10,
        "text": "Evaluation of anemia includes complete blood count with peripheral smear, iron studies, vitamin B12 and folate levels and stool occult blood when gastrointestinal blood loss is suspected.",
        "source": "Anemia guideline",
    },
    {
        "id": 11,
        "text": "Stroke-like symptoms such as sudden focal weakness or slurred speech require immediate CT brain without contrast, blood glucose check, electrolytes and ECG as part of acute stroke protocol.",
        "source": "Stroke guideline",
    },
    {
        "id": 12,
        "text": "Persistent headache with red flag features such as vomiting or neurological deficit warrants neuroimaging with CT or MRI brain and basic laboratory work-up.",
        "source": "Headache guideline",
    },
    {
        "id": 13,
        "text": "Acute abdominal pain is evaluated using CBC, serum amylase and lipase, liver function tests, serum electrolytes and ultrasound abdomen, with CT imaging if serious pathology suspected.",
        "source": "Acute abdomen guideline",
    },
    {
        "id": 14,
        "text": "Pregnant women require routine antenatal work-up including CBC, blood group and Rh typing, urine analysis and obstetric ultrasound to assess fetal wellbeing.",
        "source": "Antenatal care guideline",
    },
]

# Create collection
client.recreate_collection(
    collection_name=GUIDELINE_COLLECTION,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

# Insert embeddings
points = []
for g in GUIDELINES:
    vec = embedder.encode(g["text"]).tolist()
    points.append(
        PointStruct(
            id=g["id"],
            vector=vec,
            payload={"text": g["text"], "source": g["source"]},
        )
    )

client.upsert(
    collection_name=GUIDELINE_COLLECTION,
    points=points,
)

print("Done! Ingested guidelines into local Qdrant.")
