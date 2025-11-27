import cv2
import requests

BACKEND_URL = "http://127.0.0.1:8000/trigger-workflow"


def trigger_consultation():
    print("📡 Triggering workflow...")
    try:
        response = requests.post(
            BACKEND_URL,
            json={
                "patient_id": "AUTO001",
                "note_text": "Patient with chest pain for 2 days, non-radiating, no shortness of breath."
            }
        )
        print("✅ Backend response:", response.status_code)
        print(response.json())
    except Exception as e:
        print("❌ Error calling backend:", e)


def main():
    print("🔍 Starting biometric trigger (face detection)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open camera. Is it used by another app?")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            triggered = True

        cv2.imshow("Biometric Trigger - Press Q to quit", frame)

        if triggered:
            print("😊 Face detected — closing camera and triggering consultation...")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("👋 Exiting without triggering.")
            cap.release()
            cv2.destroyAllWindows()
            return

    # Stop camera and close window BEFORE calling backend
    cap.release()
    cv2.destroyAllWindows()

    if triggered:
        trigger_consultation()


if __name__ == "__main__":
    main()
