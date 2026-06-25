from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import JSON, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME = "MedIntel AI Backend"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medintel.db")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en-IN")

# ---------------------------------------------------------------------------
# CORS — FIX 1: list your exact Cloudflare Pages domain(s) here.
# Never use allow_origins=["*"] with allow_credentials=True; browsers block it.
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS: List[str] = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5500,http://127.0.0.1:5500",
    ).split(",")
    if origin.strip()
]

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(120), unique=True, index=True, nullable=False)
    display_name = Column(String(200), nullable=True)
    email = Column(String(255), unique=True, index=True, nullable=True)
    role = Column(String(50), default="patient")
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ReportRecord(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String(200), index=True, nullable=True)
    report_type = Column(String(80), nullable=True)
    extracted_text = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    severity = Column(String(40), nullable=True)
    payload = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ConversationRecord(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(120), index=True, nullable=True)
    role = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title=APP_NAME, version="1.0.0")

# FIX 1 applied here — separate origins list, credentials only when needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    profile: Dict[str, Any] = Field(default_factory=dict)
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    language: Optional[str] = None


class TriageRequest(BaseModel):
    symptoms: str
    age: Optional[int] = None
    gender: Optional[str] = None
    duration: Optional[str] = None
    temperature: Optional[str] = None
    weight: Optional[float] = None
    pregnancy: Optional[str] = None
    allergy: Optional[str] = None
    medical_history: Optional[str] = None
    language: Optional[str] = None


class SOSRequest(BaseModel):
    lat: Optional[float] = None
    lng: Optional[float] = None
    contact_name: Optional[str] = None
    emergency_type: Optional[str] = None
    notes: Optional[str] = None


class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = "patient"


class LoginRequest(BaseModel):
    username: str
    password: str


class PrescriptionSafetyRequest(BaseModel):
    medicines: List[str] = Field(default_factory=list)
    age: Optional[int] = None
    weight: Optional[float] = None
    gender: Optional[str] = None
    pregnancy: Optional[str] = None
    allergy: Optional[str] = None
    medical_history: Optional[str] = None
    symptoms: Optional[str] = None
    language: Optional[str] = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha_password(password: str) -> str:
    salt = os.getenv("PASSWORD_SALT", "medintel-salt")
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def safe_lower(text: Optional[str]) -> str:
    return (text or "").lower()


# FIX 2: Use Optional[UploadFile] instead of UploadFile | None (Python 3.9 compat)
def _read_text_from_file(upload: UploadFile) -> str:
    filename = (upload.filename or "").lower()
    content = upload.file.read()
    if not content:
        return ""

    if filename.endswith((".txt", ".csv", ".json", ".html", ".htm", ".xml", ".md")):
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return content.decode("latin-1", errors="ignore")

    if filename.endswith(".pdf"):
        text = _extract_pdf_text(content)
        if text.strip():
            return text
        return _ocr_pdf_bytes(content)

    if filename.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")):
        return _ocr_image_bytes(content)

    try:
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        return ""


def _ocr_image_bytes(image_bytes: bytes) -> str:
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def _ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        images = convert_from_bytes(pdf_bytes, dpi=220)
        return "\n".join(pytesseract.image_to_string(img.convert("L")) for img in images[:12]).strip()
    except Exception:
        return ""


def extract_sections(text: str) -> List[str]:
    patterns = [
        ("Patient details", r"\b(patient|name|age|sex|gender|id)\b"),
        ("Vitals", r"\b(bp|blood pressure|pulse|hr|spo2|oxygen|temperature|temp)\b"),
        ("CBC / hematology", r"\b(hemoglobin|haemoglobin|hb|wbc|rbc|platelet|platelets|mcv|mch|mchc)\b"),
        ("Biochemistry", r"\b(glucose|blood sugar|hba1c|creatinine|urea|sodium|potassium|alt|ast|bilirubin|cholesterol|triglyceride)\b"),
        ("Prescription", r"\b(rx|prescription|tablet|tab\.|capsule|cap\.|syrup|ointment|dose|dosage|take)\b"),
        ("Imaging / pathology", r"\b(x-ray|xray|ct|mri|ultrasound|scan|biopsy|pathology)\b"),
    ]
    found = [name for name, pat in patterns if re.search(pat, text or "", re.I)]
    return found or ["No clear section markers found"]


def parse_labs(text: str) -> List[Dict[str, Any]]:
    t = text or ""
    labs: List[Dict[str, Any]] = []

    def add(test: str, value: Any, status: str, note: str) -> None:
        labs.append({"test": test, "value": value, "status": status, "note": note})

    m = re.search(r"(?:blood\s*sugar|glucose|bs)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\b", t, re.I)
    if m:
        v = float(m.group(1))
        add("Glucose", v, "High" if v >= 200 else "Borderline" if v >= 140 else "Normal",
            "Suggests diabetes-risk range or uncontrolled sugar" if v >= 200 else "Monitor")

    m = re.search(r"(?:hba1c|a1c)\s*[:=]?\s*(\d+(?:\.\d+)?)\b", t, re.I)
    if m:
        v = float(m.group(1))
        add("HbA1c", v, "High" if v >= 6.5 else "Borderline" if v >= 5.7 else "Normal",
            "Diabetes-range glycemic marker" if v >= 6.5 else "Monitor")

    m = re.search(r"(?:haemoglobin|hemoglobin|hb)\s*[:=]?\s*(\d+(?:\.\d+)?)\b", t, re.I)
    if m:
        v = float(m.group(1))
        add("Hemoglobin", v, "Low" if v < 10 else "Borderline" if v < 12 else "Normal",
            "Anemia risk" if v < 10 else "Monitor")

    m = re.search(r"(?:bp|blood pressure)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})", t, re.I)
    if m:
        sys, dia = int(m.group(1)), int(m.group(2))
        status = "High" if (sys >= 140 or dia >= 90) else "Borderline" if (sys >= 130 or dia >= 80) else "Normal"
        add("Blood Pressure", f"{sys}/{dia}", status, "Hypertension concern" if status == "High" else "Monitor")

    m = re.search(r"(?:creatinine|creat)\s*[:=]?\s*(\d+(?:\.\d+)?)\b", t, re.I)
    if m:
        v = float(m.group(1))
        add("Creatinine", v, "High" if v > 1.3 else "Normal",
            "Kidney function review suggested" if v > 1.3 else "Monitor")

    m = re.search(r"(?:platelets?|plt)\s*[:=]?\s*(\d{2,5})\b", t, re.I)
    if m:
        v = int(m.group(1))
        add("Platelets", v, "Low" if v < 150000 else "Normal" if v <= 450000 else "High", "CBC review suggested")

    m = re.search(r"(?:wbc|white blood cell(?: count)?)\s*[:=]?\s*(\d+(?:\.\d+)?)\b", t, re.I)
    if m:
        v = float(m.group(1))
        add("WBC", v, "High" if v > 11 else "Low" if v < 4 else "Normal",
            "Possible infection / inflammation" if v > 11 else "Monitor")

    m = re.search(r"(?:sodium|na)\s*[:=]?\s*(\d+(?:\.\d+)?)\b", t, re.I)
    if m:
        v = float(m.group(1))
        add("Sodium", v, "Low" if v < 135 else "High" if v > 145 else "Normal", "Electrolyte imbalance review")

    m = re.search(r"(?:potassium|k)\s*[:=]?\s*(\d+(?:\.\d+)?)\b", t, re.I)
    if m:
        v = float(m.group(1))
        add("Potassium", v, "Low" if v < 3.5 else "High" if v > 5.1 else "Normal", "Electrolyte review")

    return labs


def detect_emergency(text: str) -> Tuple[bool, List[str]]:
    patterns = [
        (r"chest pain|chest pressure", "Chest pain"),
        (r"left arm numbness|arm numbness|jaw pain", "Heart-attack pattern"),
        (r"shortness of breath|breathing trouble|difficulty breathing", "Breathing difficulty"),
        (r"sweating|cold sweat", "Sweating"),
        (r"stroke|slurred speech|facial droop|one side weak|weakness on one side", "Stroke-like signs"),
        (r"unconscious|faint|loss of consciousness|seizure", "Loss of consciousness / seizure"),
        (r"severe bleeding|vomit blood|black stool", "Severe bleeding"),
    ]
    hits = [label for pat, label in patterns if re.search(pat, text, re.I)]
    return bool(hits), hits


def suggest_diet(text: str, labs: List[Dict[str, Any]]) -> List[str]:
    t = safe_lower(text)
    advice: List[str] = []
    glucose = next((x for x in labs if x["test"] == "Glucose"), None)
    hb = next((x for x in labs if x["test"] == "Hemoglobin"), None)
    creat = next((x for x in labs if x["test"] == "Creatinine"), None)
    if glucose and glucose["status"] in {"High", "Borderline"}:
        advice.extend([
            "Diabetes-friendly: avoid sweets, soda, fruit juice, and large rice portions.",
            "Prefer oats, salad, legumes, vegetables, protein, and controlled carbohydrate portions.",
        ])
    if hb and hb["status"] == "Low":
        advice.append("Anemia support: iron-rich diet, vitamin C with meals, and clinician review for cause.")
    if creat and creat["status"] == "High":
        advice.append("Kidney-friendly: review protein, sodium, and potassium intake with a doctor or dietitian.")
    if "vomit" in t or "food poisoning" in t:
        advice.append("Hydration first: oral rehydration solution, small sips, monitor dehydration.")
    if "fever" in t:
        advice.append("Fever: fluids, rest, watch for warning signs; consider CBC if symptoms persist.")
    return advice or ["Balanced diet, hydration, and follow the clinician's advice."]


def safe_medicines(profile: Dict[str, Any], findings: List[Dict[str, Any]], symptoms: str) -> List[Dict[str, Any]]:
    age_str = str(profile.get("age") or "")
    age = int(age_str) if age_str.isdigit() else 0
    weight = profile.get("weight")
    allergy = safe_lower(profile.get("allergy"))
    pregnancy = safe_lower(profile.get("pregnancy"))
    history = safe_lower(profile.get("medicalHistory") or profile.get("medical_history") or "")
    symp = safe_lower(symptoms)
    has_fever = any(k in symp for k in ["fever", "temperature"])
    glucose = next((x for x in findings if x["test"] == "Glucose"), None)
    meds: List[Dict[str, Any]] = []

    if has_fever:
        if 0 < age <= 12:
            meds.append({
                "name": "Paracetamol syrup (supportive only)",
                "usage": "For fever/discomfort when a clinician agrees.",
                "dosage": f"Weight-based dosing required — weight: {weight or 'unknown'} kg. Confirm with pediatrician.",
                "caution": "Do not guess dose. Check label concentration.",
            })
        else:
            meds.append({
                "name": "Paracetamol (supportive only)",
                "usage": "General fever/discomfort relief when medically appropriate.",
                "dosage": "Follow product label. Do not exceed max daily dose.",
                "caution": "Avoid in severe liver disease or allergy.",
            })

    if (glucose and glucose["status"] in {"High", "Borderline"}) or "diabetes" in history:
        meds.append({
            "name": "Sugar-free formulations preferred",
            "usage": "Choose sugar-free syrups/medicines when possible.",
            "dosage": "Confirm with pharmacist.",
            "caution": "Avoid sugar-heavy syrups unless prescribed.",
        })

    if pregnancy and any(x in pregnancy for x in ["yes", "pregnant"]):
        meds.append({
            "name": "Pregnancy safety check",
            "usage": "Review every medicine with a clinician before use.",
            "dosage": "Do not self-start new medicines.",
            "caution": "Some medicines are unsafe in pregnancy.",
        })

    if allergy:
        meds.append({
            "name": "Allergy check",
            "usage": "Verify active ingredients against allergy history.",
            "dosage": "Not applicable.",
            "caution": f"Reported allergy: {profile.get('allergy') or 'unspecified'}.",
        })

    if not meds:
        meds.append({
            "name": "No direct medicine suggestion",
            "usage": "Use only supportive care until a doctor reviews the report.",
            "dosage": "Not applicable.",
            "caution": "Do not replace doctor-prescribed treatment.",
        })

    return meds[:6]


def build_summary(labs: List[Dict[str, Any]], emergency: bool, severity: str) -> str:
    parts: List[str] = []
    if emergency:
        parts.append("Emergency red flags detected.")
    for lab in labs[:4]:
        if lab["test"] == "Glucose" and lab["status"] == "High":
            parts.append(f"Glucose {lab['value']} is high — needs diabetes evaluation.")
        if lab["test"] == "Hemoglobin" and lab["status"] == "Low":
            parts.append(f"Hemoglobin {lab['value']} suggests anemia risk.")
        if lab["test"] == "Blood Pressure" and lab["status"] == "High":
            parts.append(f"Blood pressure {lab['value']} suggests hypertension concern.")
    if not parts:
        parts.append("No major red-flag pattern recognized from the provided text.")
    parts.append(f"Overall severity: {severity}.")
    return " ".join(parts)


def reason_on_report(text: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    text = text or ""
    labs = parse_labs(text)
    emergency, emergency_hits = detect_emergency(text)
    conditions: List[str] = []
    risks: List[str] = []
    actions: List[str] = []

    for lab in labs:
        if lab["test"] == "Glucose" and lab["status"] == "High":
            conditions.extend(["Diabetes risk", "Hyperglycemia concern"])
            risks.append("High sugar may need clinical follow-up")
            actions.append("Check fasting glucose / HbA1c and seek clinician review")
        if lab["test"] == "Hemoglobin" and lab["status"] == "Low":
            conditions.append("Anemia risk")
            risks.append("Low hemoglobin can cause fatigue and weakness")
            actions.append("CBC, ferritin / iron studies, and doctor review")
        if lab["test"] == "Blood Pressure" and lab["status"] == "High":
            conditions.append("Hypertension concern")
            risks.append("Persistently high BP needs medical review")
            actions.append("Repeat BP and review lifestyle / medication with clinician")
        if lab["test"] == "Creatinine" and lab["status"] == "High":
            conditions.append("Kidney function concern")
            risks.append("Raised creatinine may reflect kidney impairment")
            actions.append("eGFR, urine tests, and nephrology/physician review")
        if lab["test"] in {"WBC"} and lab["status"] == "High":
            conditions.append("Infection / inflammation possible")

    low = safe_lower(text)
    if re.search(r"dengue|platelet|viral fever", low):
        conditions.append("Viral / dengue-like illness")
        actions.append("CBC trend and hydration")
    if re.search(r"vomit|vomiting|diarrhea|diarrhoea", low):
        conditions.append("Gastroenteritis / food poisoning possible")
        actions.append("Hydration assessment")

    severity = ("Emergency" if emergency
                else "High" if any(x["status"] == "High" for x in labs)
                else "Moderate" if any(x["status"] == "Borderline" for x in labs)
                else "Low")

    if emergency:
        actions = ["Seek emergency care immediately", "Use SOS / ambulance", "Do not delay for home treatment"] + actions

    if not conditions:
        conditions.append("No strong diagnosis pattern recognized from available text")

    return {
        "extracted_text": text,
        "summary": build_summary(labs, emergency, severity),
        "severity": severity,
        "emergency": emergency,
        "emergency_hits": emergency_hits,
        "conditions": list(dict.fromkeys(conditions))[:8],
        "lab_findings": labs,
        "diet_advice": suggest_diet(text, labs),
        "suggested_medicines": safe_medicines(profile, labs, text),
        "risks": list(dict.fromkeys(risks))[:8],
        "actions": list(dict.fromkeys(actions))[:8],
        "next_steps": list(dict.fromkeys(actions))[:8],
        "structure_sections": extract_sections(text),
        "medical_disclaimer": "Supportive AI only — not a substitute for a doctor.",
        "safety_checks": {
            "age_checked": bool(profile.get("age")),
            "weight_checked": bool(profile.get("weight")),
            "allergy_checked": bool(profile.get("allergy")),
            "pregnancy_checked": bool(profile.get("pregnancy")),
            "history_checked": bool(profile.get("medicalHistory") or profile.get("medical_history")),
        },
    }


def language_hint(text: str, language: Optional[str]) -> str:
    lang = (language or DEFAULT_LANGUAGE).lower()
    if lang.startswith("hi"):
        return "यह केवल सहायक जानकारी है। डॉक्टर की सलाह ज़रूर लें।"
    if lang.startswith("bn"):
        return "এটি শুধু সহায়ক তথ্য। ডাক্তারের পরামর্শ নিন।"
    if lang.startswith("ta"):
        return "இது உதவி தகவல் மட்டும். மருத்துவர் ஆலோசனையைப் பெறவும்."
    if lang.startswith("kn"):
        return "ಇದು ಸಹಾಯಕ ಮಾಹಿತಿ ಮಾತ್ರ. ವೈದ್ಯರ ಸಲಹೆ ಪಡೆಯಿರಿ."
    return text


def maybe_store_report(db: Session, patient_name: Optional[str], report_type: Optional[str], payload: Dict[str, Any]) -> None:
    try:
        from sqlalchemy.orm import Session as _S
        rec = ReportRecord(
            patient_name=patient_name,
            report_type=report_type,
            extracted_text=payload.get("extracted_text"),
            summary=payload.get("summary"),
            severity=payload.get("severity"),
            payload=payload,
        )
        db.add(rec)
        db.commit()
    except Exception:
        db.rollback()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "name": APP_NAME, "time": now_iso()}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "healthy", "service": APP_NAME, "time": now_iso()}


# FIX 2: Optional[UploadFile] instead of UploadFile | None (works on Python 3.9)
@app.post("/analyze-report")
async def analyze_report(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(default=None),
    patient_name: str = Form(default=""),
    ocr_text: str = Form(default=""),
    report_type: str = Form(default=""),
    ocr_mode: str = Form(default=""),
    analysis_focus: str = Form(default=""),
    age: str = Form(default=""),
    gender: str = Form(default=""),
    weight: str = Form(default=""),
    allergy: str = Form(default=""),
    pregnancy: str = Form(default=""),
    language: str = Form(default=DEFAULT_LANGUAGE),
    medical_history: str = Form(default=""),
    symptoms: str = Form(default=""),
    latitude: str = Form(default=""),
    longitude: str = Form(default=""),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    file_text = ""
    if file is not None:
        try:
            file_text = _read_text_from_file(file)
        except Exception:
            file_text = ""

    extracted_text = (ocr_text or file_text or "").strip() or "No OCR text provided."
    profile = {
        "patientName": patient_name, "age": age, "gender": gender, "weight": weight,
        "allergy": allergy, "pregnancy": pregnancy, "lang": language,
        "medicalHistory": medical_history, "summarySymptoms": symptoms,
    }

    payload = reason_on_report(extracted_text, profile)
    payload.update({
        "patient_name": patient_name, "report_type": report_type,
        "ocr_mode": ocr_mode, "analysis_focus": analysis_focus,
        "language": language, "timestamp": now_iso(),
        "location": {"latitude": latitude or None, "longitude": longitude or None},
    })
    payload["language_hint"] = language_hint(payload["summary"], language)
    maybe_store_report(db, patient_name or None, report_type or None, payload)
    return payload


# Alias so the frontend's fallback endpoint also works
@app.post("/analyze")
async def analyze_alias(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(default=None),
    patient_name: str = Form(default=""),
    ocr_text: str = Form(default=""),
    report_type: str = Form(default=""),
    ocr_mode: str = Form(default=""),
    analysis_focus: str = Form(default=""),
    age: str = Form(default=""),
    gender: str = Form(default=""),
    weight: str = Form(default=""),
    allergy: str = Form(default=""),
    pregnancy: str = Form(default=""),
    language: str = Form(default=DEFAULT_LANGUAGE),
    medical_history: str = Form(default=""),
    symptoms: str = Form(default=""),
    latitude: str = Form(default=""),
    longitude: str = Form(default=""),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    return await analyze_report(
        background_tasks=background_tasks, file=file, patient_name=patient_name,
        ocr_text=ocr_text, report_type=report_type, ocr_mode=ocr_mode,
        analysis_focus=analysis_focus, age=age, gender=gender, weight=weight,
        allergy=allergy, pregnancy=pregnancy, language=language,
        medical_history=medical_history, symptoms=symptoms,
        latitude=latitude, longitude=longitude, db=db,
    )


@app.post("/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    q = safe_lower(req.query)
    profile = req.profile or {}

    if re.search(r"\brice\b", q) and "diabetes" in q:
        response = "Yes, in controlled quantity. Prefer brown rice or smaller portions. Pair with protein and fiber, and follow the doctor's sugar plan."
    elif "fever" in q and "medicine" in q:
        response = "Before suggesting any medicine, check age, weight, allergy, pregnancy, and medical history. For a child, dosing should be weight-based and reviewed by a pediatric clinician."
    elif "anemia" in q or "hemoglobin" in q:
        response = "Hemoglobin around 8 can suggest anemia risk and needs CBC review and clinician evaluation. Iron-rich food alone may not be enough."
    elif "diabetes" in q and "eat" in q:
        response = "Choose controlled portions, prefer high-fiber foods, and avoid sweets and soda. Ask for a diabetes diet plan based on your report and medicines."
    elif "emergency" in q or "chest pain" in q:
        response = "Possible heart attack risk. Seek emergency care immediately and use SOS / ambulance support."
    else:
        response = "I can explain reports, medicines, symptoms, diet, emergency signs, and nearby care. Ask in English or Hindi."

    try:
        db.add(ConversationRecord(username=profile.get("patientName"), role="user", message=req.query, metadata_json={"profile": profile}))
        db.add(ConversationRecord(username=profile.get("patientName"), role="ai", message=response, metadata_json={}))
        db.commit()
    except Exception:
        db.rollback()

    return {"response": response, "language": req.language or profile.get("lang") or DEFAULT_LANGUAGE}


@app.post("/triage")
def triage(req: TriageRequest) -> Dict[str, Any]:
    raw = " ".join(filter(None, [req.symptoms, req.duration, req.temperature, req.medical_history]))
    low = safe_lower(raw)
    emergency, hits = detect_emergency(low)
    severity = "Low"
    possibilities: List[str] = []
    tests: List[str] = []
    advice: List[str] = []

    if emergency:
        severity = "Emergency"
        possibilities = ["Possible heart attack risk", "Possible stroke / severe acute event"]
        tests = ["Call emergency services now"]
        advice = ["Seek emergency care immediately", "Use SOS / ambulance", "Do not drive alone"]
    else:
        if any(x in low for x in ["fever", "temperature", "body pain", "vomit"]):
            possibilities += ["Viral fever", "Dengue", "Food poisoning", "Infection / inflammation"]
            tests += ["CBC", "Hydration assessment"]
            advice += ["Drink fluids and monitor for warning signs"]
            severity = "Moderate" if "vomit" in low else "Low"
        if any(x in low for x in ["cough", "sore throat", "cold"]):
            possibilities += ["Upper respiratory infection"]
            tests += ["Clinical evaluation"]

    if not possibilities:
        possibilities = ["Non-specific symptoms"]
        advice = ["Provide more details or upload the report"]

    return {
        "severity": severity,
        "possibilities": list(dict.fromkeys(possibilities))[:8],
        "tests": list(dict.fromkeys(tests))[:8],
        "advice": list(dict.fromkeys(advice))[:8],
        "emergency": emergency,
        "emergency_hits": hits,
        "summary": f"Triage severity: {severity}",
        "language": req.language or DEFAULT_LANGUAGE,
    }


@app.post("/sos")
def sos(req: SOSRequest) -> Dict[str, Any]:
    map_url = None
    if req.lat is not None and req.lng is not None:
        map_url = f"https://www.google.com/maps/search/?api=1&query={req.lat},{req.lng}"
    return {
        "status": "emergency_support_ready",
        "message": "Use emergency services immediately.",
        "emergency_contacts": ["Local ambulance / emergency number", "Nearest ICU", "Nearest emergency hospital"],
        "one_tap_navigation": map_url,
        "timestamp": now_iso(),
    }


# FIX 3: Accept JSON body correctly; was receiving dict but endpoint was fragile
@app.post("/nearby-care")
def nearby_care(payload: Dict[str, Any]) -> Dict[str, Any]:
    lat = payload.get("lat") or payload.get("latitude")
    lng = payload.get("lng") or payload.get("longitude")
    radius = int(payload.get("radius") or 5000)
    care_type = safe_lower(payload.get("type") or "hospital")
    focus = payload.get("focus") or payload.get("disease_focus")

    if lat is None or lng is None:
        raise HTTPException(status_code=400, detail="lat and lng are required")

    selector_map = {
        "hospital": '"amenity"="hospital"',
        "clinic": '"amenity"="clinic"',
        "doctor": '"amenity"="doctors"',
        "pharmacy": '"amenity"="pharmacy"',
        "dentist": '"amenity"="dentist"',
        "cardiologist": '"healthcare:speciality"="cardiology"',
        "neurologist": '"healthcare:speciality"="neurology"',
        "pathology": '"healthcare:speciality"="pathology"',
        "ambulance": '"emergency"="ambulance_station"',
        "icu": '"amenity"="hospital"',
    }
    selector = selector_map.get(care_type, '"amenity"="hospital"')
    overpass_query = (
        f"[out:json][timeout:25];"
        f"(node[{selector}](around:{radius},{lat},{lng});"
        f"way[{selector}](around:{radius},{lat},{lng}););"
        f"out center 12;"
    )

    live: List[Dict[str, Any]] = []
    try:
        resp = requests.post("https://overpass-api.de/api/interpreter", data=overpass_query, timeout=25)
        resp.raise_for_status()
        live = (resp.json().get("elements") or [])[:12]
    except Exception:
        live = []

    results = []
    for el in live:
        center = el.get("center") or {}
        results.append({
            "name": el.get("tags", {}).get("name") or care_type.title(),
            "lat": el.get("lat") or center.get("lat"),
            "lng": el.get("lon") or center.get("lon"),
            "address": ", ".join(filter(None, [
                el.get("tags", {}).get("addr:full"),
                el.get("tags", {}).get("addr:street"),
                el.get("tags", {}).get("addr:city"),
            ])) or "Address not available",
            "focus": focus,
        })

    return {
        "results": results,
        "navigation": {
            "google_maps": f"https://www.google.com/maps/search/?api=1&query={lat},{lng}",
            "openstreetmap": f"https://www.openstreetmap.org/?mlat={lat}&mlon={lng}#map=18/{lat}/{lng}",
        },
        "note": "If Overpass results are empty, the frontend falls back to demo locations.",
    }


@app.post("/auth/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    user = User(
        username=req.username, display_name=req.display_name, email=req.email,
        role=req.role or "patient", password_hash=sha_password(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "Registration successful", "user": {"id": user.id, "username": user.username, "role": user.role}}


@app.post("/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    user = db.query(User).filter(User.username == req.username).first()
    if not user or user.password_hash != sha_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = hashlib.sha256((JWT_SECRET + req.username + now_iso()).encode()).hexdigest()
    return {"message": "Login successful", "token": token, "user": {"id": user.id, "username": user.username, "role": user.role}}


@app.get("/history")
def history(patient_name: Optional[str] = None, db: Session = Depends(get_db)) -> Dict[str, Any]:
    q = db.query(ReportRecord)
    if patient_name:
        q = q.filter(ReportRecord.patient_name == patient_name)
    reports = q.order_by(ReportRecord.created_at.desc()).limit(50).all()
    return {
        "items": [
            {"id": r.id, "patient_name": r.patient_name, "report_type": r.report_type,
             "summary": r.summary, "severity": r.severity,
             "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in reports
        ],
        "count": len(reports),
    }


@app.post("/history")
def add_history(item: Dict[str, Any], db: Session = Depends(get_db)) -> Dict[str, Any]:
    rec = ReportRecord(
        patient_name=item.get("patient_name"), report_type=item.get("report_type"),
        extracted_text=item.get("extracted_text"), summary=item.get("summary"),
        severity=item.get("severity"), payload=item,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return {"message": "History saved", "id": rec.id}


@app.get("/export")
def export_history(patient_name: Optional[str] = None, format: str = "json", db: Session = Depends(get_db)):
    q = db.query(ReportRecord)
    if patient_name:
        q = q.filter(ReportRecord.patient_name == patient_name)
    reports = q.order_by(ReportRecord.created_at.desc()).all()
    rows = [
        {"id": r.id, "patient_name": r.patient_name, "report_type": r.report_type,
         "summary": r.summary, "severity": r.severity,
         "created_at": r.created_at.isoformat() if r.created_at else None}
        for r in reports
    ]
    if format.lower() == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else ["id", "patient_name", "summary", "severity", "created_at"])
        writer.writeheader()
        writer.writerows(rows)
        return StreamingResponse(iter([buf.getvalue().encode("utf-8")]), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=medintel_history.csv"})
    return JSONResponse({"items": rows, "count": len(rows)})


@app.post("/prescription-safety")
def prescription_safety(req: PrescriptionSafetyRequest) -> Dict[str, Any]:
    age_str = str(req.age or "")
    age = int(age_str) if age_str.isdigit() else None
    history = safe_lower(req.medical_history)
    allergy = safe_lower(req.allergy)
    pregnancy = safe_lower(req.pregnancy)
    findings = []

    for med in req.medicines:
        m = safe_lower(med)
        flags: List[str] = []
        advice: List[str] = []
        if any(x in m for x in ["ibuprofen", "diclofenac", "naproxen"]) and "kidney" in history:
            flags.append("NSAID caution in kidney disease")
            advice.append("Ask a doctor before use")
        if "syrup" in m and "diabetes" in history:
            flags.append("Check sugar content in syrup")
            advice.append("Prefer sugar-free formulations")
        if any(x in m for x in ["paracetamol", "acetaminophen"]) and age is not None and age <= 12:
            flags.append("Pediatric dose must be weight-based")
            advice.append(f"Weight-based dosing required; weight={req.weight or 'unknown'} kg")
        if pregnancy and pregnancy not in {"no", "not pregnant", "false"}:
            flags.append("Pregnancy safety review needed")
            advice.append("Check with obstetric clinician")
        if not flags:
            flags.append("No immediate red flag detected")
            advice.append("Still verify with doctor/pharmacist")
        findings.append({"medicine": med, "flags": flags, "advice": advice})

    return {
        "disclaimer": "Not a substitute for doctor. Do not self-medicate.",
        "items": findings,
        "overall": "Always verify dose, interactions, allergies, pregnancy, and age/weight.",
    }


@app.post("/voice-transcribe")
async def voice_transcribe(file: UploadFile = File(...)) -> Dict[str, Any]:
    tmp = Path("./_upload_audio.bin")
    content = await file.read()
    tmp.write_bytes(content)
    try:
        try:
            import whisper  # type: ignore
            model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))
            result = model.transcribe(str(tmp))
            return {"text": result.get("text", "").strip(), "language": result.get("language"), "engine": "whisper"}
        except Exception:
            return {"text": "Speech transcription engine not installed.", "engine": "fallback"}
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/demo/ping")
def demo_ping() -> Dict[str, Any]:
    return {"message": "MedIntel backend is ready", "allowed_origins": ALLOWED_ORIGINS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
