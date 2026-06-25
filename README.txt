# 🏥 MedIntel AI

**An AI-powered Medical Report Understanding and Clinical Decision Support Platform**

MedIntel AI helps users understand medical reports, analyze symptoms, locate nearby healthcare services, and maintain medical history using Artificial Intelligence.

> ⚠️ **Disclaimer**
> This application is for educational and supportive purposes only.
> It does **NOT** replace professional medical advice, diagnosis, or treatment.

---

# 🌟 Features

## 📄 Medical Report Analysis

- Upload PDF reports
- Upload lab reports
- Upload prescriptions
- OCR-based text extraction
- Medical report understanding
- Lab value interpretation
- Risk analysis
- Easy-to-understand report explanation

Example

```
Glucose: 260 mg/dL

↓

High Blood Sugar

↓

Possible Diabetes

↓

Recommendation:
Consult your physician and monitor HbA1c.
```

---

## 🩺 Symptom Checker

Describe symptoms naturally.

Example

```
Fever for 3 days
Body pain
Vomiting
```

Possible causes

- Viral Fever
- Dengue
- Food Poisoning

Recommended tests

- CBC
- Hydration
- Doctor consultation

---

## 🚨 Emergency Detection

The AI detects emergency situations such as

- Chest Pain
- Left Arm Pain
- Sweating
- Difficulty Breathing
- Stroke Symptoms

Example

```
Possible Heart Attack Risk

Seek Emergency Medical Care Immediately
```

---

## 💊 Medicine Safety

The system checks

- Age
- Weight
- Gender
- Allergies
- Pregnancy
- Medical History

Example

```
Child
Age : 5 years

↓

Suggests Paracetamol Syrup

↓

Weight-based dosage

↓

Doctor consultation recommended
```

---

## 🥗 Smart Diet Recommendation

Example

### Diabetes

Avoid

- Sugar
- Soft Drinks
- White Rice

Eat

- Oats
- Salad
- Brown Rice
- Vegetables

---

### Kidney Disease

Recommended

- Low Potassium Diet
- Low Sodium Diet

---

## 🏥 Nearby Healthcare

Find nearby

- Hospitals
- Doctors
- Pharmacies
- Ambulance
- ICU
- Pathology Labs

Includes

- Live Distance
- Ratings
- Navigation

---

## 🎤 Voice Assistant

Supports

- Speech to Text
- Text to Speech

Languages

- English
- Hindi
- Bengali
- Tamil
- Kannada

---

## 🤖 AI Chatbot

The chatbot can

- Explain reports
- Explain diseases
- Explain medicines
- Explain precautions
- Answer diet questions
- Remember previous conversations

Example

```
Can I eat rice if I have diabetes?

↓

Yes

Prefer Brown Rice

Eat in controlled quantity.
```

---

## 📊 Medical History

Track

- Blood Sugar
- Blood Pressure
- Medicines
- Symptoms

View trends for

- 3 Months
- 6 Months
- 12 Months

---

# 🧠 AI Technologies

The platform is designed to support

- OCR
- Medical NER
- PDF Structure Extraction
- BioBERT
- ClinicalBERT
- Llama 3
- Mistral
- LangChain
- Retrieval Augmented Generation (RAG)

---

# 🛠️ Tech Stack

## Frontend

- HTML
- CSS
- JavaScript
- Leaflet Maps
- Chart.js

## Backend

- FastAPI
- Python

## Database

- PostgreSQL
- MongoDB

## AI

- BioBERT
- ClinicalBERT
- Llama 3
- Mistral

---

# 📁 Project Structure

```
MedIntel-AI/
│
├── frontend/
│   └── index.html
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── uploads/
│
├── README.md
└── LICENSE
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/MedIntel-AI.git
```

Move into the project

```bash
cd MedIntel-AI/backend
```

Install packages

```bash
pip install -r requirements.txt
```

Run FastAPI

```bash
uvicorn app:app --reload
```

Open

```
http://127.0.0.1:8000/docs
```

to test the backend.

Open

```
frontend/index.html
```

to use the web application.

---

# 🌐 Deployment

Frontend

- Cloudflare Pages

Backend

- Render

Database

- PostgreSQL (Supabase)

---

# 📱 Android App

The website can be packaged as an Android application using Android Studio WebView.

---

# 🔒 Privacy

- User data should be securely stored.
- Medical reports should not be shared without consent.
- Authentication should be enabled in production.

---

# 📌 Future Improvements

- Doctor Dashboard
- Family Dashboard
- ECG Analysis
- X-Ray Analysis
- MRI Analysis
- CT Scan Analysis
- Wearable Device Integration
- WhatsApp Notifications
- AI Follow-up Monitoring

---

# 👨‍💻 Author

**Durva Pawar**

PhD Research Scholar

Bioinformatics | Artificial Intelligence | Medical AI

---

# 📄 License

This project is intended for educational and research purposes.

Medical decisions should always be made by qualified healthcare professionals.
