# Vision_2_Text"

# 🧠 VisionNarrate – AI Image Caption Generator

VisionNarrate is a full-stack **AI-powered image captioning web application** that generates meaningful natural language descriptions for images using **Deep Learning (CNN + LSTM)**.  
The project demonstrates strong skills in **Machine Learning, Backend API development, and Frontend UI integration**.

---

## 🚀 Why This Project Matters (Recruiter Perspective)

✔ Solves a real-world AI problem (visual understanding)  
✔ Demonstrates end-to-end ML pipeline  
✔ Combines AI + Full-Stack Development  
✔ Shows production-level debugging and stability fixes  
✔ Suitable for AI / ML / Software / Full-Stack roles  

---

## 🖼️ Application Demo

**User Flow:**  
Upload Image → AI Processes Image → Caption Generated

**Sample Output:**  
> *“a dog is running through the grass”*

---

## 🏗️ System Architecture

Frontend (HTML, CSS, JavaScript)
|
v
Flask REST API (Python)
|
v
CNN (VGG16) → Feature Extraction
|
v
LSTM Decoder → Caption Generation
|
v
Response Sent to UI

yaml
Copy code

---

## 🧠 Technical Approach

### 1️⃣ Image Feature Extraction
- Pretrained **VGG16 (CNN)** model
- Extracts **4096-dimensional feature vectors**
- Uses transfer learning (ImageNet weights)

### 2️⃣ Caption Generation
- **LSTM-based sequence model**
- Trained on **Flickr8k dataset**
- Uses `<startseq>` and `<endseq>` tokens
- Predicts captions word-by-word

### 3️⃣ Inference Pipeline
- User uploads image
- Backend preprocesses image
- CNN extracts visual features
- LSTM generates caption
- Caption returned via REST API

---

## 🛠️ Tech Stack Used

| Layer | Technologies |
|------|-------------|
| Frontend | HTML5, CSS3, JavaScript |
| Backend | Python, Flask, Flask-CORS |
| AI / ML | TensorFlow, Keras |
| Models | CNN (VGG16), LSTM |
| Dataset | Flickr8k |
| Tools | Git, GitHub, VS Code |

**ATS-Friendly Stack (Single Line):**  
Python, Flask, TensorFlow, Keras, CNN (VGG16), LSTM, HTML, CSS, JavaScript, Flickr8k

---



## 📂 Project Structure

```text
VisionNarrate/
│
├── backend/                    # Backend (Flask + AI Inference)
│   ├── __init__.py
│   ├── app.py                  # Flask API entry point
│   │
│   ├── utils/                  # Utility modules
│   │   ├── __init__.py
│   │   └── caption_generator.py   # Image caption generation logic
│   │
│   ├── model/                  # Trained model artifacts
│   │   ├── caption_model.h5
│   │   └── tokenizer.pkl
│   │
│   └── uploads/                # Temporarily stores uploaded images
│
├── frontend/                   # Frontend (UI Layer)
│   ├── index.html              # Main UI page
│   ├── style.css               # Styling & animations
│   └── script.js               # Client-side logic & API calls
│
├── dataset/                    # Dataset (not pushed to GitHub)
│   └── Flickr8k/
│       ├── Images/
│       └── captions.txt
│
├── screenshots/                # UI & output screenshots
│   └── ui-demo.png
│
├── .gitignore                  # Git ignored files
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation


## ⚙️ Setup & Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/VisionNarrate-AI-Image-Caption-Generator.git
cd VisionNarrate-AI-Image-Caption-Generator
2️⃣ Create Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install Dependencies
bash
Copy code
pip install flask flask-cors tensorflow keras numpy pillow
▶️ Running the Application
Start Backend
bash
Copy code
python -m backend.app
Server runs at:

cpp
Copy code
http://127.0.0.1:5000
Start Frontend
Open frontend/index.html

Or use VS Code Live Server

🔌 API Endpoint
POST /generate-caption
Request:

yaml
Copy code
form-data:
image : <image_file>
Response:

json
Copy code
{
  "caption": "a dog running through the grass"
}
✨ Key Features
✔ AI-powered image understanding
✔ Clean, animated, user-friendly UI
✔ Stable backend inference (no reload issues)
✔ Multiple image uploads supported
✔ Modular & scalable codebase

📈 Future Enhancements
Beam search for improved captions
Cloud deployment (Render / Hugging Face)
Mobile-first responsive UI
Multilingual caption support


👤 Author
Ayush Vibhute
GitHub: https://github.com/AyushVibhute555

#output
https://github.com/AyushVibhute555/Vision_2_Text/blob/main/UI.png

