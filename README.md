# Face Recognition-Based Attendance System

Automated, web-based attendance system using face recognition via webcam  
Built with Python, Flask, OpenCV, scikit-learn (KNN), and Pandas

---

## 🚀 Project Overview

This project enables hands-free, automated attendance marking using facial recognition through a live webcam stream. Designed for classrooms, small offices, or as a portfolio demonstration, it allows admin users to register individuals, mark attendance, and review records—all via a simple web interface.

---

## ✨ Features

- **Web-based registration:** Add new users with name and unique ID; captures multiple face images per user.
- **Silently captures faces:** Webcam collects face images in the background, no pop-up windows.
- **Machine Learning powered:** Uses a K-Nearest Neighbors (KNN) classifier for face recognition.
- **Automatic attendance logging:** Detects and identifies faces, logs attendance only once per person per day.
- **Easy CSV management:** Saves daily attendance logs in CSV format for easy review and export.
- **Clean, simple UI:** All usage via browser; no command-line interaction needed once running.

---

## 🧑💻 Technology Stack

- Python 3
- Flask – Web framework and server
- OpenCV – Webcam access, face detection & image processing
- Haar Cascade Classifier – Pretrained face detector model for OpenCV
- scikit-learn (KNN) – Machine learning classifier for recognition
- Joblib – Model serialization & loading (saves trained ML model)
- Pandas – For managing daily CSV attendance logs
- HTML/Jinja Templates – For web UI

---

## 🗂️ Project Structure
face-recognition-attendance-system/
├── app.py
├── haarcascade_frontalface_default.xml
├── static/
│ ├── faces/
│ └── face_recognition_model.pkl
├── Attendance/
│ └── Attendance-<date>.csv
├── templates/
│ └── home.html
├── README.md

---

## ⚙️ Setup & Installation

1. **Clone this repository:**
    ```
    git clone https://github.com/your-username/face-recognition-attendance-system.git
    cd face-recognition-attendance-system
    ```

2. **(Recommended) Create and activate a virtual environment:**
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install dependencies:**
    ```
    python3 -m pip install flask opencv-python scikit-learn pandas joblib
    ```

4. **Run the server:**
    ```
    python app.py
    ```

5. **Open your browser to:**
    ```
    http://127.0.0.1:5000/
    ```

---

## 🛠️ Usage Guide

### Register New User
- Fill in name and ID in the web form.
- The webcam silently captures 10 face images.
- Images are saved under `static/faces/username_id/`, and the ML model is retrained automatically.

### Take Attendance
- Click "Take Attendance" in the web interface.
- The webcam scans faces for ~100 frames. Detected registered users are marked present in the CSV log for the day.
- Attendance is displayed below the form, with time, name, and ID.

### Review Attendance
- All marked attendance records are viewable from the home page.
- Data is stored in `Attendance/Attendance-<date>.csv`.

---

## 🖼️ Screenshots

> Paste screenshots here after running your app: registration form, attendance table before and after, key code section, etc.

| Registration Form | Before Attendance | After Attendance |
|:-----------------:|:----------------:|:----------------:|
| (screenshot)      | (screenshot)     | (screenshot)     |

---

## 💡 How It Works (In Brief)

- **Registration:** User is registered with name/ID, their face images are captured and stored.
- **Model Training:** ML model (KNN) is trained on all stored faces.
- **Attendance:** On attendance, webcam frames are processed, faces detected and classified; present users are marked in the daily CSV.
- **Records Shown:** Attendance is visible instantly on the web dashboard.

---

## 📈 Demo Presentation

See `project_presentation.pptx` for a complete project explanation, technical diagrams, and talking points for interviews.

---

## 🛑 Limitations & Future Enhancements

- No login/authentication (open web UI)
- Not robust for very large datasets or spoofing (uses simple KNN, not deep learning)
- No anti-spoofing or liveness checks
- **Next steps:** Browser-based webcam (for true platform independence), authentication, deep learning model for better accuracy

---

## 📬 Contact

**Your Name**  
Email: gogulnath.work@gmail.com
Feel free to open issues or submit pull requests!
