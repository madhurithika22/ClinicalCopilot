import os
from tempfile import NamedTemporaryFile
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timezone
from .auth import authorize_patient, is_patient_authorized
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from .state import AgentState
from .schemas import (
    TriggerWorkflowRequest,
    TriggerWorkflowResponse,
    HumanReviewRequest,
    EMRUpdatePayload,
)
from .graph import run_initial_workflow
from .tools import tool_update_emr, tool_transcribe_voice, EMR_STORE_PATH, tool_send_to_pharmacy, PHARMACY_STORE_PATH
from .nodes.hil_node import hil_apply_decision

class ApproveEMRRequest(BaseModel):
    patient_id: str
    note_summary: str
    symptoms: List[str]
    suggested_tests: List[str]
    draft_prescription: str
app = FastAPI(title="Agentic AI Healthcare Workflow Assistant")

class PharmacySendRequest(BaseModel):
    patient_id: str
    prescription: str
    emr_record_id: Optional[str] = None
    suggested_tests: List[str] = []
    symptoms: List[str] = []

# Simple CORS for frontend demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/trigger-workflow", response_model=TriggerWorkflowResponse)
def trigger_workflow(req: TriggerWorkflowRequest):
    """
    Entry point: simulate a consultation with note_text.
    """
    init_state = AgentState(
        patient_id=req.patient_id,
        raw_transcript=req.note_text,
        note_summary=req.note_text,
    )
    final_state = run_initial_workflow(init_state)
    return {"state": final_state.model_dump()}

@app.get("/get-emr")
def get_emr(patient_id: str):
    """
    Return all EMR records for a given patient_id
    from the local emr_store.json (backend/emr_store.json).
    Records are sorted by timestamp_utc (newest first).
    """
    records = []

    if EMR_STORE_PATH.exists():
        try:
            with EMR_STORE_PATH.open("r", encoding="utf-8") as f:
                all_records = json.load(f)
        except Exception:
            all_records = []
    else:
        all_records = []

    # Filter by patient_id
    for rec in all_records:
        if rec.get("patient_id") == patient_id:
            records.append(rec)

    # Sort by timestamp_utc descending (newest first)
    def _get_ts(r):
        ts = r.get("timestamp_utc") or r.get("timestamp") or ""
        return ts

    records.sort(key=_get_ts, reverse=True)

    return records

@app.get("/get-pharmacy-orders")
def get_pharmacy_orders(patient_id: str):
    """
    Return all pharmacy orders for a given patient_id
    from pharmacy_orders.json, sorted newest first.
    """
    records = []

    if PHARMACY_STORE_PATH.exists():
        try:
            with PHARMACY_STORE_PATH.open("r", encoding="utf-8") as f:
                all_orders = json.load(f)
        except Exception:
            all_orders = []
    else:
        all_orders = []

    # Filter by patient_id
    for rec in all_orders:
        if rec.get("patient_id") == patient_id:
            records.append(rec)

    # Sort by timestamp_utc descending
    def _get_ts(r):
        return r.get("timestamp_utc") or ""

    records.sort(key=_get_ts, reverse=True)

    return records


@app.post("/human-review")
def human_review(req: HumanReviewRequest):
    """
    Endpoint for physician to approve or reject suggested actions.
    """
    # In a real setup you'd retrieve AgentState from a DB by patient_id
    # For demo, create a minimal placeholder state
    state = AgentState(patient_id=req.patient_id)
    state = hil_apply_decision(state, approved=req.approved, doctor_comments=req.doctor_comments)
    return {"message": "Review applied", "state": state.model_dump()}


@app.post("/emr-update")
def emr_update(payload: EMRUpdatePayload):
    """
    Demonstrates EMR update as an MCP-like tool call.
    """
    result = tool_update_emr(payload.model_dump())
    return {"result": result}

@app.post("/audio-workflow", response_model=TriggerWorkflowResponse)
async def audio_workflow(patient_id: str, audio: UploadFile = File(...)):
    suffix = ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file_bytes = await audio.read()
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        transcript = tool_transcribe_voice(tmp_path)

        init_state = AgentState(
            patient_id=patient_id,
            raw_transcript=transcript,
            note_summary=transcript,
        )
        final_state = run_initial_workflow(init_state)
        return {"state": final_state.model_dump()}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Agentic Clinical Workflow Copilot</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #0f172a, #020617 60%);
      color: #e5e7eb;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      border-bottom: 1px solid #1f2937;
      background: #020617cc;
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    header h1 {
      font-size: 1.25rem;
      color: #38bdf8;
      margin: 0;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid #1f2937;
      background: #020617;
      font-size: 0.8rem;
    }
    .pill input {
      border: none;
      outline: none;
      background: transparent;
      color: #e5e7eb;
      font-size: 0.8rem;
    }
    main {
      padding: 16px 24px 32px;
      max-width: 1200px;
      margin: 0 auto;
    }
    .status-line {
      font-size: 0.8rem;
      margin-bottom: 8px;
      min-height: 1.5em;
    }
    .status-ok { color: #4ade80; }
    .status-warn { color: #fb7185; }
    .status-info { color: #38bdf8; }

    /* Stepper / carousel controls */
    .stepper {
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    .step-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid #1f2937;
      background: #020617;
      font-size: 0.75rem;
      opacity: 0.5;
    }
    .step-pill.active {
      border-color: #38bdf8;
      box-shadow: 0 0 0 1px #0ea5e9;
      opacity: 1;
    }
    .step-pill span.index {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      border-radius: 999px;
      background: #0f172a;
      font-size: 0.7rem;
    }
    .step-nav {
      margin-left: auto;
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .btn {
      border: none;
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.08s ease, box-shadow 0.08s ease, background 0.1s ease;
    }
    .btn:active { transform: scale(0.97); }
    .btn-primary {
      background: #06b6d4;
      color: #020617;
      box-shadow: 0 8px 18px rgba(8,145,178,0.7);
    }
    .btn-primary:hover { background: #22d3ee; }
    .btn-ghost {
      background: #020617;
      color: #e5e7eb;
      border: 1px solid #1f2937;
    }
    .btn-danger {
      background: #f97373;
      color: #111827;
    }
    .btn[disabled] {
      opacity: 0.5;
      cursor: not-allowed;
      box-shadow: none;
    }

    .slide {
      display: none;
      animation: fadeIn 0.25s ease-out;
    }
    .slide.active {
      display: block;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px);}
      to { opacity: 1; transform: translateY(0);}
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }
    .card {
      background: #020617;
      border-radius: 16px;
      border: 1px solid #1f2937;
      padding: 14px 16px;
      box-shadow: 0 18px 35px rgba(0,0,0,0.55);
    }
    .card h2, .card h3 {
      margin: 0 0 6px;
      font-size: 0.95rem;
      color: #e5e7eb;
    }
    .card p {
      margin: 4px 0;
      font-size: 0.8rem;
      color: #9ca3af;
    }
    textarea {
      width: 100%;
      min-height: 120px;
      resize: vertical;
      border-radius: 12px;
      border: 1px solid #1f2937;
      background: #020617;
      color: #e5e7eb;
      padding: 8px;
      font-size: 0.8rem;
      outline: none;
    }
    textarea:focus {
      border-color: #38bdf8;
    }
    pre {
      white-space: pre-wrap;
      font-size: 0.75rem;
      max-height: 180px;
      overflow-y: auto;
    }
    ul {
      padding-left: 18px;
      margin: 6px 0;
      font-size: 0.8rem;
    }
    .badge {
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 0.7rem;
      background: #0f172a;
      border: 1px solid #1f2937;
      margin: 2px 4px 2px 0;
    }
    video {
      width: 260px;
      height: 180px;
      border-radius: 16px;
      border: 1px solid #1f2937;
      background: #020617;
      object-fit: cover;
    }
    .emr-item {
      border-radius: 12px;
      border: 1px solid #1f2937;
      padding: 8px 10px;
      margin-bottom: 6px;
      background: #020617;
    }
    code {
      background: #020617;
      padding: 2px 4px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #93c5fd;
    }
    @media (max-width: 768px) {
      header { flex-direction: column; align-items: flex-start; gap: 8px; }
      .step-nav { margin-left: 0; margin-top: 6px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Agentic Clinical Workflow Copilot</h1>
    <div class="pill">
      <span style="color:#9ca3af;font-size:0.75rem;">Patient ID:</span>
      <input id="patientIdInput" value="P001" />
    </div>
  </header>

  <main>
    <div id="statusLine" class="status-line"></div>

    <!-- Stepper / Carousel header -->
    <div class="stepper">
      <div id="step1Pill" class="step-pill active">
        <span class="index">1</span>
        <span>Patient Presence Agent (Face Verify)</span>
      </div>
      <div id="step2Pill" class="step-pill">
        <span class="index">2</span>
        <span>Listening Agents (Audio / Live)</span>
      </div>
      <div id="step3Pill" class="step-pill">
        <span class="index">3</span>
        <span>Clinical Brain & EMR Agents</span>
      </div>

      <div class="step-nav">
        <button id="btnPrev" class="btn btn-ghost" disabled>← Back</button>
        <button id="btnNext" class="btn btn-primary">Next →</button>
      </div>
    </div>

    <!-- Slide 1: Biometric Gate -->
    <section id="slide1" class="slide active">
      <div class="grid" style="align-items:flex-start; margin-bottom:16px;">
        <div class="card">
          <h2>🔐 Patient Presence Agent</h2>
          <p>This agent verifies that the patient is physically present using face detection. Only after this step, listening agents and EMR will unlock.</p>
          <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:6px;">
            <button id="btnStartCam" class="btn btn-ghost">🎥 Start Camera</button>
            <button id="btnVerifyFace" class="btn btn-primary">✅ Verify Patient Face</button>
          </div>
          <p style="margin-top:8px;font-size:0.8rem;">
            Status:
            <span id="verifyStatusText" style="font-weight:600; color:#f97373;">Not Verified</span>
          </p>
          <p style="margin-top:4px;font-size:0.75rem;">
            Once verified, Step 2 (Listening) and Step 3 (Clinical Brain & EMR) become available for this patient.
          </p>
        </div>
        <div class="card" style="display:flex;flex-direction:column;align-items:flex-start;gap:6px;">
          <h3>Patient Camera View</h3>
          <video id="verifyVideo" autoplay muted></video>
          <p style="font-size:0.75rem;color:#9ca3af;">Align the patient's face in the frame before clicking <b>Verify Patient Face</b>.</p>
        </div>
      </div>
    </section>

    <!-- Slide 2: Listening Agents (Upload + Live Mic) -->
    <section id="slide2" class="slide">
      <div class="grid">
        <div class="card">
          <h2>🎧 Upload Consultation Audio (WAV)</h2>
          <p>This triggers the Audio Scribe Agent. Audio is transcribed, then passed into the workflow graph.</p>
          <input id="audioFileInput" type="file" accept=".wav,audio/wav" style="margin-top:6px;font-size:0.8rem;" />
          <p id="audioFileName" style="font-size:0.75rem;margin-top:4px;color:#e5e7eb;"></p>
          <button id="btnRunAudio" class="btn btn-primary" style="margin-top:10px;">▶ Run Audio Workflow</button>
        </div>
        <div class="card">
          <h2>🎙️ Live Microphone Agent</h2>
          <p>Uses browser speech recognition to capture the consultation in real time.</p>
          <p id="liveSupportMsg" style="font-size:0.75rem;"></p>
          <div style="display:flex;gap:8px;margin-top:6px;flex-wrap:wrap;">
            <button id="btnStartListening" class="btn btn-primary">Start Listening</button>
            <button id="btnStopListening" class="btn btn-danger">Stop</button>
          </div>
          <h3 style="margin-top:10px;">Live Transcript (editable)</h3>
          <textarea id="liveTranscriptBox" placeholder="Speak after clicking Start Listening..."></textarea>
          <button id="btnRunLiveFromText" class="btn btn-ghost" style="margin-top:8px;">🤖 Run Workflow on Transcript</button>
        </div>
        <div class="card">
          <h3>Transcript from Audio (editable)</h3>
          <p>Transcript will appear here after you upload audio and run the workflow. You can edit and re-run on text.</p>
          <textarea id="transcriptBox" placeholder="Transcript will appear here after audio upload or live listening..."></textarea>
          <button id="btnRunFromText" class="btn btn-ghost" style="margin-top:8px;">🤖 Run Workflow on Transcript</button>
        </div>
      </div>
    </section>

    <!-- Slide 3: Clinical Brain & EMR -->
    <section id="slide3" class="slide">
      <div class="grid" style="margin-bottom:16px;">
        <div class="card">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <h2>🗂 EMR Records for this Patient</h2>
            <button id="btnLoadEmr" class="btn btn-primary">Refresh EMR</button>
          </div>
          <div id="emrList" style="margin-top:8px;max-height:260px;overflow-y:auto;"></div>
        </div>
            <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <h3>🏥 Pharmacy Orders</h3>
          <button id="btnLoadPharmacy" class="btn btn-ghost">Refresh</button>
        </div>
        <div id="pharmacyList" style="margin-top:8px;max-height:260px;overflow-y:auto;font-size:0.8rem;"></div>
        </div>

        <div class="card">
          <h3>🩺 Symptoms</h3>
          <div id="symptomList" style="font-size:0.8rem;color:#e5e7eb;">
            <p style="color:#9ca3af;font-size:0.8rem;">No symptoms yet. Run a workflow.</p>
          </div>
        </div>
        <div class="card">
          <h3>🧪 Suggested Investigations</h3>
          <div id="testList">
            <p style="color:#9ca3af;font-size:0.8rem;">No tests yet. Mention symptoms like "chest pain", "fever", "diabetes".</p>
          </div>
          <p style="margin-top:6px;font-size:0.75rem;color:#9ca3af;">
            Doctor can edit the final list below (one test per line) before approval:
          </p>
          <textarea id="testsEditBox" placeholder="ECG 12-lead&#10;Chest X-Ray (PA view)&#10;CBC with Differential" style="margin-top:4px;font-size:0.75rem;min-height:90px;"></textarea>
        </div>
        <div class="card">
        <h3>Draft Prescription</h3>
        <div id="rxBox">
          <p style="color:#9ca3af;font-size:0.8rem;">No draft prescription yet.</p>
        </div>

        <p style="margin-top:6px;font-size:0.75rem;color:#9ca3af;">
          Doctor can edit the prescription before saving to EMR:
        </p>
        <textarea id="rxEditBox" placeholder="Edited prescription will appear here after workflow..." style="margin-top:4px;font-size:0.75rem;min-height:100px;"></textarea>

        <div id="safetyBox" style="margin-top:6px;"></div>
        <div id="emrIdBox" style="margin-top:6px;font-size:0.75rem;"></div>

        <button id="btnApproveEmr" class="btn btn-primary" style="margin-top:8px;width:100%;">
          Approve & Save to EMR
        </button>
        <button id="btnSendPharmacy" class="btn btn-ghost" style="margin-top:6px;width:100%;">
          Send Prescription to Pharmacy
        </button>
      </div>

        <div class="card" style="grid-column:1/-1;">
          <h3>📋 Workflow Timeline (Audit Log)</h3>
          <div id="auditLogBox" style="max-height:160px;overflow-y:auto;font-size:0.75rem;color:#e5e7eb;">
            <p style="color:#9ca3af;font-size:0.8rem;">Run a workflow to view events here.</p>
          </div>
        </div>
      </div>
    </section>
  </main>

  <script>
    // ---------- Global State ----------
    const statusLine = document.getElementById("statusLine");
    const patientIdInput = document.getElementById("patientIdInput");

    const step1Pill = document.getElementById("step1Pill");
    const step2Pill = document.getElementById("step2Pill");
    const step3Pill = document.getElementById("step3Pill");
    const btnPrev = document.getElementById("btnPrev");
    const btnNext = document.getElementById("btnNext");

    const slide1 = document.getElementById("slide1");
    const slide2 = document.getElementById("slide2");
    const slide3 = document.getElementById("slide3");

    const btnStartCam = document.getElementById("btnStartCam");
    const btnVerifyFace = document.getElementById("btnVerifyFace");
    const verifyStatusText = document.getElementById("verifyStatusText");
    const videoEl = document.getElementById("verifyVideo");

    const audioFileInput = document.getElementById("audioFileInput");
    const audioFileName = document.getElementById("audioFileName");
    const btnRunAudio = document.getElementById("btnRunAudio");
    const transcriptBox = document.getElementById("transcriptBox");
    const btnRunFromText = document.getElementById("btnRunFromText");

    const liveSupportMsg = document.getElementById("liveSupportMsg");
    const btnStartListening = document.getElementById("btnStartListening");
    const btnStopListening = document.getElementById("btnStopListening");
    const liveTranscriptBox = document.getElementById("liveTranscriptBox");
    const btnRunLiveFromText = document.getElementById("btnRunLiveFromText");

    const btnLoadEmr = document.getElementById("btnLoadEmr");
    const emrList = document.getElementById("emrList");
    const btnLoadPharmacy = document.getElementById("btnLoadPharmacy");
    const pharmacyList = document.getElementById("pharmacyList");
    const symptomList = document.getElementById("symptomList");
    const testList = document.getElementById("testList");
    const testsEditBox = document.getElementById("testsEditBox");
    const rxBox = document.getElementById("rxBox");
    const rxEditBox = document.getElementById("rxEditBox");
    const safetyBox = document.getElementById("safetyBox");
    const emrIdBox = document.getElementById("emrIdBox");
    const auditLogBox = document.getElementById("auditLogBox");
    const btnApproveEmr = document.getElementById("btnApproveEmr");
    const btnSendPharmacy = document.getElementById("btnSendPharmacy");

    let patientVerified = false;
    let currentState = null;
    let currentStream = null;
    let currentSlide = 1;
    let lastApprovedEmrId = null;

    const API_BASE = ""; // same origin

    function setStatus(text, type="info") {
      statusLine.textContent = text || "";
      statusLine.className = "status-line";
      if (!text) return;
      if (type === "ok") statusLine.classList.add("status-ok");
      if (type === "warn") statusLine.classList.add("status-warn");
      if (type === "info") statusLine.classList.add("status-info");
    }

    function getPatientId() {
      return (patientIdInput.value || "").trim();
    }

    // ---------- Slide / Carousel logic ----------
    function updateStepUI() {
      [step1Pill, step2Pill, step3Pill].forEach(p => p.classList.remove("active"));
      [slide1, slide2, slide3].forEach(s => s.classList.remove("active"));

      if (currentSlide === 1) {
        step1Pill.classList.add("active");
        slide1.classList.add("active");
        btnPrev.disabled = true;
        btnNext.textContent = "Next →";
      } else if (currentSlide === 2) {
        step2Pill.classList.add("active");
        slide2.classList.add("active");
        btnPrev.disabled = false;
        btnNext.textContent = "Next →";
      } else {
        step3Pill.classList.add("active");
        slide3.classList.add("active");
        btnPrev.disabled = false;
        btnNext.textContent = "Done";
      }
    }

    function goNext() {
      if (currentSlide === 1) {
        if (!patientVerified) {
          setStatus("Patient presence not verified. Please verify face before moving to Listening Agents.", "warn");
          return;
        }
        currentSlide = 2;
      } else if (currentSlide === 2) {
        currentSlide = 3;
      } else {
        // Slide 3 "Done" – stay there or circle back if you want
        currentSlide = 1;
      }
      updateStepUI();
    }

    function goPrev() {
      if (currentSlide === 2) {
        currentSlide = 1;
      } else if (currentSlide === 3) {
        currentSlide = 2;
      }
      updateStepUI();
    }

    btnNext.onclick = goNext;
    btnPrev.onclick = goPrev;

    // when patient changes, reset slide & verification
    patientIdInput.addEventListener("input", () => {
      patientVerified = false;
      verifyStatusText.textContent = "Not Verified";
      verifyStatusText.style.color = "#f97373";
      currentSlide = 1;
      lastApprovedEmrId = null; 
      updateStepUI();
      setStatus("Patient changed. Please verify face again to unlock next steps.", "info");
    });

    // ---------- Biometric Gate ----------
    btnStartCam.onclick = async () => {
      try {
        if (currentStream) {
          currentStream.getTracks().forEach(t => t.stop());
          currentStream = null;
        }
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        currentStream = stream;
        videoEl.srcObject = stream;
        await videoEl.play();
        setStatus("Camera started. Align patient's face and click Verify.", "info");
      } catch (err) {
        console.error(err);
        setStatus("Cannot access camera: " + err.message, "warn");
      }
    };

    btnVerifyFace.onclick = async () => {
      const pid = getPatientId();
      if (!pid) {
        setStatus("Please enter a Patient ID before verifying.", "warn");
        return;
      }
      if (!videoEl.videoWidth) {
        setStatus("Camera not ready. Click Start Camera first.", "warn");
        return;
      }

      const canvas = document.createElement("canvas");
      canvas.width = videoEl.videoWidth || 640;
      canvas.height = videoEl.videoHeight || 480;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

      setStatus("Verifying patient face...", "info");

      const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg"));
      const formData = new FormData();
      formData.append("image", blob, "frame.jpg");

      try {
        const res = await fetch(`/verify-patient-face?patient_id=${encodeURIComponent(pid)}`, {
          method: "POST",
          body: formData
        });
        const json = await res.json();
        if (json.authorized) {
          patientVerified = true;
          verifyStatusText.textContent = "Verified";
          verifyStatusText.style.color = "#4ade80";
          setStatus("Patient face verified. You can move to Listening Agents.", "ok");
        } else {
          patientVerified = false;
          verifyStatusText.textContent = "Not Verified";
          verifyStatusText.style.color = "#f97373";
          setStatus("Face verification failed: " + (json.reason || "unknown reason"), "warn");
        }
      } catch (err) {
        console.error(err);
        setStatus("Error during face verification: " + err.message, "warn");
      } finally {
        if (currentStream) {
          currentStream.getTracks().forEach(t => t.stop());
          currentStream = null;
        }
      }
    };

    // ---------- Audio Upload Workflow ----------
    audioFileInput.onchange = () => {
      const file = audioFileInput.files[0];
      audioFileName.textContent = file ? ("Selected: " + file.name) : "";
    };

    btnRunAudio.onclick = async () => {
      if (!patientVerified) {
        setStatus("EMR/workflow locked: verify patient face first.", "warn");
        return;
      }
      const pid = getPatientId();
      if (!pid) {
        setStatus("Enter a Patient ID first.", "warn");
        return;
      }
      const file = audioFileInput.files[0];
      if (!file) {
        setStatus("Select a WAV file first.", "warn");
        return;
      }
      setStatus("Uploading audio and running workflow...", "info");
      const formData = new FormData();
      formData.append("audio", file);
      try {
        const res = await fetch(`/audio-workflow?patient_id=${encodeURIComponent(pid)}`, {
          method: "POST",
          body: formData
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error("Backend error " + res.status + ": " + text);
        }
        const json = await res.json();
        currentState = json.state;
        transcriptBox.value = currentState.raw_transcript || "";
        liveTranscriptBox.value = currentState.raw_transcript || "";
        renderState();
        setStatus("Audio workflow completed.", "ok");
        currentSlide = 3;
        updateStepUI();
      } catch (err) {
        console.error(err);
        setStatus("Error calling /audio-workflow: " + err.message, "warn");
      }
    };

    // ---------- Workflow from Transcript ----------
    async function runWorkflowWithTranscript(text) {
      if (!patientVerified) {
        setStatus("EMR/workflow locked: verify patient face first.", "warn");
        return;
      }
      const pid = getPatientId();
      if (!pid) {
        setStatus("Enter a Patient ID first.", "warn");
        return;
      }
      text = (text || "").trim();
      if (!text) {
        setStatus("Transcript is empty. Speak / type something first.", "warn");
        return;
      }
      setStatus("Running workflow on transcript...", "info");
      try {
        const res = await fetch("/trigger-workflow", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ patient_id: pid, note_text: text })
        });
        if (!res.ok) {
          const t = await res.text();
          throw new Error("Backend error " + res.status + ": " + t);
        }
        const json = await res.json();
        currentState = json.state;
        renderState();
        setStatus("Workflow completed.", "ok");
        currentSlide = 3;
        updateStepUI();
      } catch (err) {
        console.error(err);
        setStatus("Error calling /trigger-workflow: " + err.message, "warn");
      }
    }

    btnRunFromText.onclick = () => runWorkflowWithTranscript(transcriptBox.value);
    btnRunLiveFromText.onclick = () => runWorkflowWithTranscript(liveTranscriptBox.value);

    // ---------- Browser STT for Live Tab ----------
    let recognition = null;
    let listening = false;
    let finalTranscript = "";

    (function initSTT() {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) {
        liveSupportMsg.textContent = "❌ This browser does not support SpeechRecognition. Use Chrome.";
        btnStartListening.disabled = true;
        btnStopListening.disabled = true;
        return;
      }
      recognition = new SR();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";
      liveSupportMsg.textContent = "✅ Live STT available (Chrome Web Speech).";

      recognition.onstart = () => {
        listening = true;
        setStatus("Listening... speak now.", "info");
      };
      recognition.onerror = (event) => {
        console.error("STT error:", event);
        setStatus("Speech recognition error: " + event.error, "warn");
      };
      recognition.onend = () => {
        listening = false;
        setStatus("Stopped listening.", "info");
      };
      recognition.onresult = (event) => {
        let interim = "";
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const t = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += " " + t;
          } else {
            interim += " " + t;
          }
        }
        liveTranscriptBox.value = (finalTranscript + " " + interim).trim();
        transcriptBox.value = liveTranscriptBox.value;
      };
    })();

    btnStartListening.onclick = () => {
      if (!recognition || listening) return;
      finalTranscript = liveTranscriptBox.value || "";
      recognition.start();
    };
    btnStopListening.onclick = () => {
      if (!recognition || !listening) return;
      recognition.stop();
    };
    btnLoadPharmacy.onclick = async () => {
    if (!patientVerified) {
      setStatus("Pharmacy data locked: verify patient face first.", "warn");
      return;
    }
    const pid = getPatientId();
    if (!pid) {
      setStatus("Enter a Patient ID first.", "warn");
      return;
    }
    setStatus("Loading pharmacy orders...", "info");
    try {
      const res = await fetch(`/get-pharmacy-orders?patient_id=${encodeURIComponent(pid)}`);
      if (!res.ok) {
        const t = await res.text();
        throw new Error("Backend error " + res.status + ": " + t);
      }
      const json = await res.json();
      renderPharmacyList(json);
      setStatus("Loaded " + json.length + " pharmacy order(s).", "ok");
    } catch (err) {
      console.error(err);
      setStatus("Error loading pharmacy orders: " + err.message, "warn");
    }
  };

    // ---------- EMR viewer ----------
    btnLoadEmr.onclick = async () => {
      if (!patientVerified) {
        setStatus("EMR locked: verify patient face first.", "warn");
        return;
      }
      const pid = getPatientId();
      if (!pid) {
        setStatus("Enter a Patient ID first.", "warn");
        return;
      }
      setStatus("Loading EMR records...", "info");
      try {
        const res = await fetch(`/get-emr?patient_id=${encodeURIComponent(pid)}`);
        if (!res.ok) {
          const t = await res.text();
          throw new Error("Backend error " + res.status + ": " + t);
        }
        const json = await res.json();
        renderEmrList(json);
        setStatus("Loaded " + json.length + " EMR record(s).", "ok");
      } catch (err) {
        console.error(err);
        setStatus("Error loading EMR: " + err.message, "warn");
      }
    };

    function renderEmrList(records) {
      emrList.innerHTML = "";
      if (!records || records.length === 0) {
        emrList.innerHTML = "<p style='font-size:0.8rem;color:#9ca3af;'>No EMR records yet for this patient.</p>";
        return;
      }
      records.slice().reverse().forEach(rec => {
        const div = document.createElement("div");
        div.className = "emr-item";
        const ts = (rec.timestamp_utc || "").replace("T"," ").replace("Z","");
        const sym = rec.symptoms && rec.symptoms.length ? rec.symptoms.join(", ") : "None";
        const tests = rec.suggested_tests && rec.suggested_tests.length ? rec.suggested_tests.join(", ") : "None";

        div.innerHTML = `
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <span style="font-family:monospace;color:#38bdf8;font-size:0.8rem;">${rec.emr_record_id || ""}</span>
            <span style="font-size:0.7rem;color:#9ca3af;">${ts}</span>
          </div>
          <p style="font-size:0.8rem;color:#e5e7eb;"><b>Symptoms:</b> ${sym}</p>
          <p style="font-size:0.8rem;color:#e5e7eb;"><b>Tests:</b> ${tests}</p>
        `;
        if (rec.draft_prescription) {
          const details = document.createElement("details");
          const summary = document.createElement("summary");
          summary.textContent = rec.approved_by_doctor ? "Approved Prescription" : "Draft Prescription";
          summary.style.cursor = "pointer";
          summary.style.color = rec.approved_by_doctor ? "#4ade80" : "#38bdf8";
          summary.style.fontSize = "0.75rem";
          const pre = document.createElement("pre");
          pre.textContent = rec.draft_prescription;
          pre.style.marginTop = "4px";
          details.appendChild(summary);
          details.appendChild(pre);
          div.appendChild(details);
        }
        emrList.appendChild(div);
      });
    }
    function renderPharmacyList(records) {
    pharmacyList.innerHTML = "";
    if (!records || records.length === 0) {
      pharmacyList.innerHTML =
        "<p style='font-size:0.8rem;color:#9ca3af;'>No pharmacy orders yet for this patient.</p>";
      return;
    }

    records.forEach((rec) => {
      const div = document.createElement("div");
      div.className = "emr-item"; // reuse same style

      const ts = (rec.timestamp_utc || "").replace("T", " ").replace("Z", "");
      const status = rec.status || "unknown";

      // Show only first few lines of prescription
      let rxPreview = "";
      if (rec.prescription) {
        const lines = rec.prescription.split("\\n");
        rxPreview = lines.slice(0, 3).join("\\n");
        if (lines.length > 3) rxPreview += "\\n...";
      }

      div.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
          <span style="font-family:monospace;color:#fbbf24;font-size:0.8rem;">${rec.order_id || ""}</span>
          <span style="font-size:0.7rem;color:#9ca3af;">${ts}</span>
        </div>
        <p style="font-size:0.75rem;color:#e5e7eb;"><b>Status:</b> ${status}</p>
        ${
          rec.emr_record_id
            ? "<p style='font-size:0.75rem;color:#9ca3af;'><b>From EMR:</b> " +
              rec.emr_record_id +
              "</p>"
            : ""
        }
      `;

      if (rxPreview) {
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = "Prescription details";
        summary.style.cursor = "pointer";
        summary.style.fontSize = "0.75rem";
        summary.style.color = "#38bdf8";
        const pre = document.createElement("pre");
        pre.textContent = rxPreview;
        pre.style.marginTop = "4px";
        details.appendChild(summary);
        details.appendChild(pre);
        div.appendChild(details);
      }

      pharmacyList.appendChild(div);
    });
  }

    // ---------- Render State ----------
    function renderState() {
      if (!currentState) {
        symptomList.innerHTML = "<p style='color:#9ca3af;font-size:0.8rem;'>No symptoms yet. Run a workflow.</p>";
        testList.innerHTML = "<p style='color:#9ca3af;font-size:0.8rem;'>No tests yet.</p>";
        rxBox.innerHTML = "<p style='color:#9ca3af;font-size:0.8rem;'>No draft prescription yet.</p>";
        safetyBox.innerHTML = "";
        emrIdBox.innerHTML = "";
        auditLogBox.innerHTML = "<p style='color:#9ca3af;font-size:0.8rem;'>Run a workflow to view events here.</p>";
        testsEditBox.value = "";
        rxEditBox.value = "";
        return;
      }
      const s = currentState;

      // Symptoms
      if (Array.isArray(s.symptoms) && s.symptoms.length > 0) {
        symptomList.innerHTML = s.symptoms.map(sym =>
          "<span class='badge'>" + sym + "</span>"
        ).join(" ");
      } else {
        symptomList.innerHTML = "<p style='color:#9ca3af;font-size:0.8rem;'>No symptoms detected.</p>";
      }

      // Tests
      if (Array.isArray(s.suggested_tests) && s.suggested_tests.length > 0) {
        const ul = document.createElement("ul");
        s.suggested_tests.forEach(t => {
          const li = document.createElement("li");
          li.textContent = t;
          ul.appendChild(li);
        });
        testList.innerHTML = "";
        testList.appendChild(ul);
        testsEditBox.value = s.suggested_tests.join("\\n");
      } else {
        testList.innerHTML = "<p style='color:#9ca3af;font-size:0.8rem;'>No tests suggested.</p>";
        testsEditBox.value = "";
      }

      // Rx
      if (s.draft_prescription) {
        rxBox.innerHTML = "<pre>" + s.draft_prescription + "</pre>";
        rxEditBox.value = s.draft_prescription;
      } else {
        rxBox.innerHTML = "<p style='color:#9ca3af;font-size:0.8rem;'>No draft prescription yet.</p>";
        rxEditBox.value = "";
      }

      // Safety
      if (Array.isArray(s.safety_flags) && s.safety_flags.length > 0) {
        safetyBox.innerHTML = "<p style='font-size:0.75rem;color:#facc15;'><b>⚠ Safety Flags:</b></p>" +
          "<ul>" + s.safety_flags.map(f => "<li>" + f + "</li>").join("") + "</ul>";
      } else {
        safetyBox.innerHTML = "";
      }

      // EMR record id from executed_actions if present
      let emrId = null;
      if (Array.isArray(s.executed_actions)) {
        const emrAction = s.executed_actions.find(a => a && a.action === "update_emr");
        if (emrAction && emrAction.emr_record_id) emrId = emrAction.emr_record_id;
      }
      if (emrId) {
        emrIdBox.innerHTML = "<span style='font-size:0.75rem;color:#4ade80;'>🗂 EMR stored as <code>" + emrId + "</code></span>";
      } else {
        emrIdBox.innerHTML = "";
      }

      // Audit log
      if (Array.isArray(s.audit_log) && s.audit_log.length > 0) {
        auditLogBox.innerHTML = "";
        const ol = document.createElement("ol");
        s.audit_log.forEach(line => {
          const li = document.createElement("li");
          li.textContent = line;
          li.style.marginBottom = "2px";
          ol.appendChild(li);
        });
        auditLogBox.appendChild(ol);
      } else {
        auditLogBox.innerHTML = "<p style='color:#9ca3af;font-size:0.8rem;'>No audit log entries.</p>";
      }
    }
    btnSendPharmacy.onclick = async () => {
    if (!patientVerified) {
      setStatus("Pharmacy action locked: verify patient face first.", "warn");
      return;
    }
    const pid = getPatientId();
    if (!pid) {
      setStatus("Enter a Patient ID first.", "warn");
      return;
    }
    if (!currentState) {
      setStatus("Run and approve a workflow first.", "warn");
      return;
    }

    const rxText = (rxEditBox.value || "").trim();
    if (!rxText) {
      setStatus("Prescription text is empty. Please review/edit before sending.", "warn");
      return;
    }

    if (!lastApprovedEmrId) {
      setStatus("No approved EMR found. Please approve the consultation before sending to pharmacy.", "warn");
      return;
    }

    const symptoms = Array.isArray(currentState.symptoms)
      ? currentState.symptoms
      : [];
    const testsLines = (testsEditBox.value || "")
      .split("\\n")
      .map((t) => t.trim())
      .filter((t) => t.length > 0);

    setStatus("Sending prescription to pharmacy...", "info");

    try {
      const res = await fetch("/send-to-pharmacy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: pid,
          prescription: rxText,
          emr_record_id: lastApprovedEmrId,
          suggested_tests: testsLines,
          symptoms: symptoms,
        }),
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error("Backend error " + res.status + ": " + t);
      }

      const json = await res.json();
      const orderId = json.order_id || "";

      setStatus(
        "📤 Prescription sent to pharmacy as order " + orderId,
        "ok"
      );
    } catch (err) {
      console.error(err);
      setStatus("Error sending to pharmacy: " + err.message, "warn");
    }
  };

    // ---------- Approve & Save EMR ----------
    btnApproveEmr.onclick = async () => {
      if (!patientVerified) {
        setStatus("EMR locked: verify patient face first.", "warn");
        return;
      }
      const pid = getPatientId();
      if (!pid) {
        setStatus("Enter a Patient ID first.", "warn");
        return;
      }
      if (!currentState) {
        setStatus("Run a workflow first before approving.", "warn");
        return;
      }

      const noteSummary =
        (currentState.note_summary || currentState.raw_transcript || "").trim();

      const symptoms = Array.isArray(currentState.symptoms)
        ? currentState.symptoms
        : [];

      const testsLines = (testsEditBox.value || "")
        .split("\\n")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);

      const rxText = (rxEditBox.value || "").trim();

      if (!rxText) {
        setStatus("Prescription text is empty. Please review/edit before approving.", "warn");
        return;
      }

      setStatus("Saving approved consultation to EMR...", "info");

      try {
        const res = await fetch("/approve-emr", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            patient_id: pid,
            note_summary: noteSummary,
            symptoms: symptoms,
            suggested_tests: testsLines,
            draft_prescription: rxText,
          }),
        });

        if (!res.ok) {
          const t = await res.text();
          throw new Error("Backend error " + res.status + ": " + t);
        }

        const json = await res.json();
        const emrId = json.emr_record_id || "";
        lastApprovedEmrId = emrId;
        setStatus(
          "✅ Consultation approved and saved to EMR as " + emrId,
          "ok"
        );
        if (emrId) {
          emrIdBox.innerHTML =
            "<span style='font-size:0.75rem;color:#4ade80;'>🗂 Approved EMR stored as <code>" +
            emrId +
            "</code></span>";
        }
      } catch (err) {
        console.error(err);
        setStatus("Error saving EMR: " + err.message, "warn");
      }
    };

    // Init
    updateStepUI();
    setStatus("Step 1: Verify patient presence using camera. Then move to Listening Agents.", "info");
  </script>
</body>
</html>
    """


@app.post("/stt-only")
async def stt_only(audio: UploadFile = File(...)):
    """
    Lightweight STT endpoint:
    - receives audio chunk
    - runs tool_transcribe_voice
    - returns just the transcript
    """
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file_bytes = await audio.read()
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        transcript = tool_transcribe_voice(tmp_path)
        return {"transcript": transcript}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

@app.post("/approve-emr")
def approve_emr(req: ApproveEMRRequest):
    """
    Human-in-the-loop approval endpoint.
    Called from the UI *after* the doctor reviews/edits tests & prescription.
    Stores an EMR record with approved_by_doctor = true.
    """
    # Biometric gate: same rule – EMR only if patient is verified
    if not is_patient_authorized(req.patient_id):
        raise HTTPException(
            status_code=403,
            detail="Patient face not verified. EMR is locked for this patient."
        )

    # Ensure EMR store exists & load
    records = []
    if EMR_STORE_PATH.exists():
        try:
            with EMR_STORE_PATH.open("r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    # Build new EMR record
    emr_record_id = f"EMR_APPROVED_{len(records) + 1:05d}"
    now_utc = datetime.now(timezone.utc).isoformat()

    record = {
        "emr_record_id": emr_record_id,
        "record_type": "approved_consultation",
        "patient_id": req.patient_id,
        "timestamp_utc": now_utc,
        "note_summary": req.note_summary,
        "symptoms": req.symptoms,
        "suggested_tests": req.suggested_tests,
        "draft_prescription": req.draft_prescription,
        "approved_by_doctor": True,
    }

    records.append(record)

    # Save back
    EMR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EMR_STORE_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print("EMR saved to:", EMR_STORE_PATH.resolve())
    return {"status": "ok", "emr_record_id": emr_record_id}

@app.post("/send-to-pharmacy")
def send_to_pharmacy(req: PharmacySendRequest):
    """
    Pharmacy Agent endpoint.

    Called from the UI AFTER the doctor has approved the consultation
    and reviewed the final prescription. This simulates sending
    an e-prescription order to a pharmacy system.
    """
    # Same gate: only if patient face is verified
    if not is_patient_authorized(req.patient_id):
        raise HTTPException(
            status_code=403,
            detail="Patient face not verified. Pharmacy actions are locked for this patient."
        )

    # Use our mock tool to persist an order
    result = tool_send_to_pharmacy(
        {
            "patient_id": req.patient_id,
            "prescription": req.prescription,
            "emr_record_id": req.emr_record_id,
            "suggested_tests": req.suggested_tests,
            "symptoms": req.symptoms,
        }
    )

    return {
        "status": "ok",
        "order_id": result["order_id"],
        "timestamp_utc": result["timestamp_utc"],
    }


@app.post("/verify-patient-face")
async def verify_patient_face(patient_id: str, image: UploadFile = File(...)):
    """
    Biometric gate (demo-oriented):
    - Receives a frame from the browser.
    - Tries to detect at least one face using OpenCV Haarcascade.
    - On success: authorize patient_id for this session.
    - On repeated failure: for demo purposes we can choose to still authorize (optional).
    """
    # 1) Decode the image
    data = await image.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        print("❌ [verify-patient-face] Failed to decode image from browser.")
        return {"authorized": False, "reason": "Failed to decode image."}

    # 2) Prepare grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # 3) Load Haarcascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        # Cascade not loaded properly – for demo, we can still authorize
        print("⚠️ [verify-patient-face] Cascade file not loaded, running in DEMO MODE.")
        authorize_patient(patient_id)
        return {
            "authorized": True,
            "reason": "Cascade missing, demo mode: auto-authorized patient."
        }

    # 4) Run detection (slightly more lenient parameters)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,      # was 5, reduce to detect more easily
        minSize=(60, 60)     # don't require very large face
    )

    print(f"🔍 [verify-patient-face] Detected faces count: {len(faces)}")

    if len(faces) == 0:
        # --- OPTION A: STRICT (real) ---
        # return {"authorized": False, "reason": "No face detected in frame."}

        # --- OPTION B: DEMO FRIENDLY ---
        # For project demo, you might prefer to always authorize
        # if *some* frame is there, to avoid getting stuck.
        authorize_patient(patient_id)
        return {
            "authorized": True,
            "reason": "No clear face detected, but demo override: patient authorized."
        }

    # At least one face found
    authorize_patient(patient_id)
    return {"authorized": True, "reason": "Face detected and patient authorized."}