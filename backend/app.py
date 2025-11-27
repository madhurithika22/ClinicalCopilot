import os
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .state import AgentState
from .schemas import (
    TriggerWorkflowRequest,
    TriggerWorkflowResponse,
    HumanReviewRequest,
    EMRUpdatePayload,
)
from .graph import run_initial_workflow
from .tools import tool_update_emr,tool_transcribe_voice
from .nodes.hil_node import hil_apply_decision

app = FastAPI(title="Agentic AI Healthcare Workflow Assistant")

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
    """
    End-to-end flow:
    1. Receive audio file for a given patient.
    2. Transcribe audio using STT tool (Whisper if available).
    3. Run the same agentic workflow as /trigger-workflow.
    """

    # 1) Save uploaded audio to a temporary file
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file_bytes = await audio.read()
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # 2) Transcribe via tool layer
        transcript = tool_transcribe_voice(tmp_path)

        # 3) Run agentic workflow
        init_state = AgentState(
            patient_id=patient_id,
            raw_transcript=transcript,
            note_summary=transcript,
        )
        final_state = run_initial_workflow(init_state)
        return {"state": final_state.model_dump()}
    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

from fastapi.responses import HTMLResponse

@app.get("/live-audio", response_class=HTMLResponse)
def live_audio_page():
    """
    Web UI for a 'perfect' listening agent using browser STT:
    - Uses Web Speech API (Chrome) for highly accurate live transcription
    - Sends final transcript to /trigger-workflow
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Listening Agent – Browser STT</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #020617;
                color: #e5e7eb;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 24px;
            }
            h1 { color: #38bdf8; }
            .row {
                display: flex;
                gap: 16px;
                margin-top: 16px;
                flex-wrap: wrap;
                max-width: 1000px;
            }
            .card {
                background: #020617;
                border-radius: 16px;
                padding: 16px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.5);
                flex: 1 1 320px;
                min-width: 300px;
            }
            button {
                margin: 4px;
                padding: 8px 14px;
                border-radius: 9999px;
                border: none;
                cursor: pointer;
                font-weight: 600;
            }
            #startBtn { background: #22c55e; color: #022c22; }
            #stopBtn { background: #ef4444; color: #fee2e2; }
            #runWorkflowBtn { background: #3b82f6; color: #dbeafe; }
            button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            label { font-size: 14px; }
            input {
                padding: 6px 10px;
                border-radius: 9999px;
                border: 1px solid #334155;
                background: #020617;
                color: #e5e7eb;
                margin-left: 6px;
            }
            textarea {
                width: 100%;
                min-height: 150px;
                background: #020617;
                border-radius: 12px;
                border: 1px solid #1f2937;
                color: #e5e7eb;
                padding: 10px;
                resize: vertical;
            }
            pre {
                white-space: pre-wrap;
                background: #020617;
                padding: 10px;
                border-radius: 8px;
                overflow-x: auto;
                max-height: 350px;
            }
            #status { font-size: 13px; margin-top: 6px; }
        </style>
    </head>
    <body>
        <h1>Agentic Healthcare – Listening Agent</h1>
        <div class="row">
            <div class="card">
                <div>
                    <label for="patientId">Patient ID:</label>
                    <input id="patientId" type="text" value="P_LISTEN_001" />
                </div>
                <p style="margin-top: 10px;">
                    1. Click <b>Start Listening</b> and speak during the consultation.<br>
                    2. Click <b>Stop</b> – you can edit the transcript.<br>
                    3. Click <b>Run Workflow</b> to let the agents process the conversation.
                </p>
                <button id="startBtn">🎙️ Start Listening</button>
                <button id="stopBtn" disabled>🛑 Stop</button>
                <div id="status"></div>
            </div>

            <div class="card">
                <h3>Live Transcript (editable)</h3>
                <textarea id="transcriptBox" placeholder="Transcript will appear here..."></textarea>
                <button id="runWorkflowBtn">🤖 Run Workflow</button>
            </div>
        </div>

        <div class="row">
            <div class="card">
                <h3>Workflow Response (state)</h3>
                <pre id="responseBox">{}</pre>
            </div>
        </div>

        <script>
            // ---- Browser Speech Recognition (Web Speech API) ----
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            let recognition = null;
            let listening = false;

            const startBtn = document.getElementById("startBtn");
            const stopBtn = document.getElementById("stopBtn");
            const runWorkflowBtn = document.getElementById("runWorkflowBtn");
            const statusDiv = document.getElementById("status");
            const transcriptBox = document.getElementById("transcriptBox");
            const responseBox = document.getElementById("responseBox");
            const patientIdInput = document.getElementById("patientId");

            if (!SpeechRecognition) {
                statusDiv.textContent = "❌ This browser does not support SpeechRecognition. Please use Chrome.";
                startBtn.disabled = true;
                stopBtn.disabled = true;
            } else {
                recognition = new SpeechRecognition();
                recognition.continuous = true;
                recognition.interimResults = true;
                recognition.lang = "en-US";

                let finalTranscript = "";

                recognition.onstart = () => {
                    listening = true;
                    statusDiv.textContent = "Listening... speak now.";
                };

                recognition.onerror = (event) => {
                    console.error("STT error:", event);
                    statusDiv.textContent = "Speech recognition error: " + event.error;
                };

                recognition.onend = () => {
                    listening = false;
                    statusDiv.textContent = "Stopped listening.";
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                };

                recognition.onresult = (event) => {
                    let interim = "";
                    for (let i = event.resultIndex; i < event.results.length; ++i) {
                        const transcript = event.results[i][0].transcript;
                        if (event.results[i].isFinal) {
                            finalTranscript += " " + transcript;
                        } else {
                            interim += " " + transcript;
                        }
                    }
                    transcriptBox.value = (finalTranscript + " " + interim).trim();
                };

                startBtn.onclick = () => {
                    if (listening) return;
                    finalTranscript = transcriptBox.value || "";
                    transcriptBox.value = finalTranscript;
                    recognition.start();
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    statusDiv.textContent = "Starting listening...";
                };

                stopBtn.onclick = () => {
                    if (!listening) return;
                    recognition.stop();
                    statusDiv.textContent = "Stopping...";
                };
            }

            // ---- Run workflow on current transcript ----
            runWorkflowBtn.onclick = async () => {
                const patientId = patientIdInput.value || "P_LISTEN_001";
                const noteText = transcriptBox.value.trim();

                if (!noteText) {
                    alert("Transcript is empty. Speak or type something first.");
                    return;
                }

                statusDiv.textContent = "Running workflow...";
                responseBox.textContent = "{}";

                try {
                    const res = await fetch("/trigger-workflow", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            patient_id: patientId,
                            note_text: noteText
                        })
                    });

                    const json = await res.json();
                    responseBox.textContent = JSON.stringify(json, null, 2);
                    statusDiv.textContent = "✅ Workflow completed. See state below.";
                } catch (err) {
                    console.error(err);
                    statusDiv.textContent = "❌ Error calling /trigger-workflow: " + err;
                }
            };
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
