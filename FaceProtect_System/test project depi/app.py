
"""
FaceProtect B2B API — FastAPI Backend
Phase 5 : API + Recommendation Logic  + Trust Lists
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import mlflow


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("FaceProtectAPI")

# Paths

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "outputs/embeddings/embeddings.npy")
METADATA_PATH   = os.path.join(BASE_DIR, "outputs/embeddings/metadata.csv")
INDEX_DIR       = os.path.join(BASE_DIR, "outputs/embeddings/faiss_index")
REPORT_PATH     = os.path.join(BASE_DIR, "outputs/reports/evaluation_report.json")
TRUST_DB_PATH   = os.path.join(BASE_DIR, "outputs/trust_lists.json")
TEMP_DIR        = os.path.join(BASE_DIR, "temp_api")

os.makedirs(TEMP_DIR, exist_ok=True)

# PRIVACY ACTIONS — تنفيذ فعلي على الصورة

def _detect_faces(image: np.ndarray):
    """Haar Cascade — سريع للـ API."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces if len(faces) > 0 else []


def apply_gaussian_blur(image: np.ndarray, strength: int = 31) -> np.ndarray:
    """
    Gaussian Blur على الوجوه.
    strength لازم تكون فردية (odd) — بنتأكد جوه.
    """
    result = image.copy()
    k = strength if strength % 2 == 1 else strength + 1
    for (x, y, w, h) in _detect_faces(image):
        roi            = result[y:y+h, x:x+w]
        result[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (k, k), 0)
    return result


def apply_pixelation(image: np.ndarray, block_size: int = 15,**kwargs) -> np.ndarray:
    """
    Pixelation (mosaic) — بيقسّم الوجه لمربعات صغيرة.
    كل مربع بياخد متوسط لونه → تأثير الفسيفساء.
    """
    result = image.copy()
    for (x, y, w, h) in _detect_faces(image):
        roi = result[y:y+h, x:x+w]
        # تصغير ثم تكبير → pixelation
        small  = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)),
                            interpolation=cv2.INTER_LINEAR)
        result[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return result


def apply_solid_redact(image: np.ndarray, color: tuple = (0, 0, 0)) -> np.ndarray:
    """
    Solid color redaction — استبدال الوجه بمستطيل لون صريح.
    الافتراضي أسود (0,0,0). ممكن تحدد (255,255,255) أبيض أو أي لون.
    """
    result = image.copy()
    for (x, y, w, h) in _detect_faces(image):
        result[y:y+h, x:x+w] = color
    return result


# خريطة action_id → دالة التنفيذ
ACTION_EXECUTORS = {
    "gaussian_blur": apply_gaussian_blur,
    "pixelation":    apply_pixelation,
    "solid_redact":  apply_solid_redact,
}


# =================
# TRUST LIST
# =================

class TrustListDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._data: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def add(self, person_id: str, label: str, trusted: bool,
            owner: str = "", notes: str = "") -> Dict:
        record = dict(person_id=person_id, label=label,
                      trusted=trusted, owner=owner, notes=notes)
        self._data[person_id] = record
        self._save()
        return record

    def remove(self, person_id: str) -> bool:
        if person_id in self._data:
            del self._data[person_id]
            self._save()
            return True
        return False

    def get(self, person_id: str) -> Optional[Dict]:
        return self._data.get(person_id)

    def is_trusted(self, person_id: str) -> Optional[bool]:
        r = self._data.get(person_id)
        return r["trusted"] if r else None

    def list_all(self, owner: Optional[str] = None) -> List[Dict]:
        records = list(self._data.values())
        if owner:
            records = [r for r in records if r.get("owner") == owner]
        return records


# ============================
# RECOMMENDATION ENGINE — قرارات + تنفيذ فعلي
# ============================

class RecommendationEngine:
    """
    ثلاث طبقات للقرار:
      1. نص توصية (reason + ranked actions)
      2. primary_action_id — الأكشن التلقائي المقترح
      3. execute(image) — تنفيذ الأكشن على الصورة فعلاً وإرجاع numpy array
    """

    WEAK_SIM   = 0.50
    STRONG_SIM = 0.90

    # --- قرار نصي ---
    def decide(
        self,
        decision:     str,
        similarity:   float,
        trust_status: Optional[bool],
        person_id:    Optional[str],
    ) -> Dict[str, Any]:

        # محظور صريح
        if trust_status is False:
            return self._build(
                primary="pixelation",
                reason="Person is on the blocked list — face anonymised.",
                actions=[
                    ("pixelation",    "Mosaic/pixelate the face",          1),
                    ("gaussian_blur", "Gaussian blur as alternative",       2),
                    ("solid_redact",  "Solid black redaction",              3),
                ],
            )

        # قبول + موثوق
        if decision == "ACCEPT" and trust_status is True:
            return self._build(
                primary="allow",
                reason="Person is trusted and recognised — no action needed.",
                actions=[("allow", "Publish image as-is", 1)],
            )

        # قبول + مجهول في القوائم
        if decision == "ACCEPT" and trust_status is None:
            return self._build(
                primary="allow_with_warning",
                reason="Recognised but not in any trust list — consent required.",
                actions=[
                    ("allow_with_warning", "Allow with privacy notice",          1),
                    ("gaussian_blur",      "Blur until consent is obtained",      2),
                    ("add_to_trust_list",  "Add to trust/block list",             3),
                ],
            )

        # رفض — تشابه ضعيف (هوية غامضة)
        if decision == "REJECT" and similarity >= self.WEAK_SIM:
            return self._build(
                primary="gaussian_blur",
                reason="Low similarity — identity uncertain, precautionary blur applied.",
                actions=[
                    ("gaussian_blur",          "Blur face (precautionary)",      1),
                    ("pixelation",             "Pixelate instead of blur",        2),
                    ("retry_with_better_image","Ask user for a clearer photo",    3),
                ],
            )

        # رفض كامل — وجه مجهول
        return self._build(
            primary="pixelation",
            reason="Unknown identity — full anonymisation applied.",
            actions=[
                ("pixelation",    "Pixelate/mosaic the face",              1),
                ("gaussian_blur", "Gaussian blur as alternative",           2),
                ("solid_redact",  "Solid black redaction",                  3),
            ],
        )

    # --- تنفيذ الأكشن على الصورة ---
    def execute(self, action_id: str, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        ينفّذ الأكشن الفعلي على الـ numpy array ويرجع الصورة المعدّلة.
        الأكشنات غير التنفيذية (allow, allow_with_warning, …) بترجع الصورة كما هي.
        """
        executor = ACTION_EXECUTORS.get(action_id)
        if executor is None:
            return image           # allow / allow_with_warning / غير معروف → بلا تغيير
        return executor(image, **kwargs)

    @staticmethod
    def _build(primary: str, reason: str, actions: list) -> Dict[str, Any]:
        return {
            "primary_action": primary,
            "reason": reason,
            "actions": [
                {"action_id": a, "description": d, "priority": p}
                for a, d, p in actions
            ],
        }


# ============================
# APP STATE
# ============================

class AppState:
    faiss_index        = None
    faiss_metadata     = None
    face_model         = None
    evaluation_threshold: float = 0.70
    trust_db:  TrustListDB        = None
    recommender: RecommendationEngine = None

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 FaceProtect API starting up...")

    state.trust_db   = TrustListDB(TRUST_DB_PATH)
    state.recommender = RecommendationEngine()

    # threshold
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH) as f:
            state.evaluation_threshold = float(json.load(f).get("threshold", 0.70))
        logger.info("✅ Threshold loaded: %.4f", state.evaluation_threshold)

    # FAISS
    try:
        from embeddings.faiss_index import get_or_build_faiss_index
        state.faiss_index, state.faiss_metadata = get_or_build_faiss_index(
            emb_path=EMBEDDINGS_PATH, meta_path=METADATA_PATH,
            index_dir=INDEX_DIR, rebuild=False, normalize_for_cosine=True,
        )
        logger.info("✅ FAISS ready — %d vectors", state.faiss_index.ntotal)
    except Exception as e:
        logger.warning("⚠️  FAISS not loaded: %s", e)

    # Model
    try:
        import yaml
        with open(os.path.join(BASE_DIR, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        from models.model_loader import load_model
        state.face_model = load_model(cfg)
        logger.info("✅ Embedding model ready.")
    except Exception as e:
        logger.warning("⚠️  Model not loaded: %s", e)

    yield
    logger.info("👋 Shutting down.")


# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="FaceProtect B2B API",
    description="AI Privacy Guard — Face Recognition, Trust Lists & Actionable Recommendations.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# -------------------------------
# Helpers
# -------------------------------

async def save_upload(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename or "img.jpg")[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        ext = ".jpg"
    path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        f.write(await file.read())
    return path

def require_faiss():
    if state.faiss_index is None:
        raise HTTPException(503, "FAISS index not ready. Run extract_embeddings.py first.")

def require_model():
    if state.face_model is None:
        raise HTTPException(503, "Embedding model not loaded.")

def image_to_stream(img: np.ndarray, filename: str = "result.jpg") -> StreamingResponse:
    _, buf = cv2.imencode(".jpg", img)
    return StreamingResponse(
        io.BytesIO(buf.tobytes()), media_type="image/jpeg",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ========================
# ENDPOINTS
# ========================

# ── 1. Health ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "faiss_ready":  state.faiss_index is not None,
        "model_ready":  state.face_model  is not None,
        "threshold":    state.evaluation_threshold,
        "trust_entries": len(state.trust_db.list_all()),
    }


# ── 2. Verify — JSON قرار + توصية نصية ─────────────────────────────────────

class VerifyResponse(BaseModel):
    request_id:        str
    decision:          str
    similarity_percent: float
    distance:          float
    person_id:         Optional[str]
    trust_status:      Optional[bool]
    recommendation:    Dict[str, Any]
    top_k:             List[Dict[str, Any]]


@app.post("/verify", response_model=VerifyResponse, tags=["Face Recognition"],
          summary="Verify face → JSON decision + recommendation")
async def verify_face(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=20),
    min_similarity: float = Query(None),
):
    """
    يرفع صورة ويرجع JSON فيه:
    - قرار ACCEPT/REJECT
    - primary_action + قائمة actions مرتبة بالأولوية
    - trust_status للشخص
    """
    require_faiss(); require_model()
    threshold = min_similarity or state.evaluation_threshold
    tmp = await save_upload(file)
    rid = os.path.basename(tmp).split(".")[0]

    try:
        embs, _ = state.face_model.extract([tmp])
        if not embs:
            raise HTTPException(422, "No face embedding extracted.")

        from embeddings.faiss_index import run_vector_search_pipeline
        result = run_vector_search_pipeline(
            query_embedding=np.asarray(embs[0]),
            index=state.faiss_index,
            metadata=state.faiss_metadata,
            top_k=top_k, min_similarity=threshold,
        )
        di          = result["decision"]
        person_id   = di.get("person_id")
        api_decision = di.get("decision", "REJECT")
        trust_status = state.trust_db.is_trusted(person_id) if person_id else None

        if trust_status is False:
            api_decision = "REJECT"

        rec = state.recommender.decide(
            decision=api_decision, similarity=di.get("similarity", 0.0),
            trust_status=trust_status, person_id=person_id,
        )
        # MLflow
        with mlflow.start_run(run_name=f"verify_{rid}"):
            mlflow.set_tracking_uri("http://mlflow:5000")
            mlflow.set_experiment("FaceProtect")
            mlflow.set_tag("endpoint", "/verify")
            mlflow.set_tag("track", "MLOps-Track6")
    
            mlflow.log_param("threshold", threshold)
            mlflow.log_param("decision", api_decision)
    
            mlflow.log_metric("similarity_score", float(di.get("similarity", 0.0) * 100))
            mlflow.log_metric("decision_binary", 1 if api_decision == "ACCEPT" else 0)

        return VerifyResponse(
            request_id=rid,
            decision=api_decision,
            similarity_percent=round(di.get("similarity", 0.0) * 100, 2),
            distance=round(di.get("distance", 1.0), 4),
            person_id=person_id,
            trust_status=trust_status,
            recommendation=rec,
            top_k=result["top_k_results"],
        )
    finally:
        if os.path.exists(tmp): os.remove(tmp)


# ── 3. Apply Action — تنفيذ action محدد على صورة ────────────────────────────

@app.post(
    "/apply_action",
    tags=["Privacy Actions"],
    summary="Apply a specific privacy action to an image",
    response_class=StreamingResponse,
)
async def apply_action(
    file: UploadFile = File(...),
    action: str = Query(
        "gaussian_blur",
        description="gaussian_blur | pixelation | solid_redact",
    ),
    strength: int = Query(31, ge=5, le=81, description="Blur/block intensity (odd number)"),
):
    """
    ينفّذ action معيّن على صورة ويرجعها معدّلة.
    - gaussian_blur: طمس ناعم
    - pixelation: فسيفساء / mosaic
    - solid_redact: مستطيل أسود صريح
    """
    if action not in ACTION_EXECUTORS:
        raise HTTPException(400, f"Unknown action '{action}'. Choose: {list(ACTION_EXECUTORS)}")

    tmp = await save_upload(file)
    try:
        img = cv2.imread(tmp)
        if img is None:
            raise HTTPException(422, "Cannot read image.")
        result_img = state.recommender.execute(action, img, strength=strength)
        return image_to_stream(result_img, filename=f"{action}.jpg")
    finally:
        if os.path.exists(tmp): os.remove(tmp)


# ── 4. Smart Protect — verify + تنفيذ تلقائي في طلب واحد ───────────────────

@app.post(
    "/smart_protect",
    tags=["Privacy Actions"],
    summary="Verify + auto-execute recommended action on image",
    response_class=StreamingResponse,
)
async def smart_protect(
    file: UploadFile = File(...),
    min_similarity: float = Query(None),
    strength: int = Query(31, ge=5, le=81),
    force_action: str = Query(
        None,
        description="Override الأكشن التلقائي: gaussian_blur | pixelation | solid_redact",
    ),
):
    """
    طلب واحد يعمل:
    1. Verify الوجه ضد قاعدة البيانات
    2. يحسب الـ recommended action
    3. ينفّذ الأكشن على الصورة فعلاً
    4. يرجع الصورة المعدّلة

    الـ response headers فيها كل نتائج الـ decision.
    """
    require_faiss(); require_model()
    threshold = min_similarity or state.evaluation_threshold
    tmp = await save_upload(file)
    rid = os.path.basename(tmp).split(".")[0]

    try:
        embs, _ = state.face_model.extract([tmp])
        if not embs:
            raise HTTPException(422, "No face embedding extracted.")

        from embeddings.faiss_index import run_vector_search_pipeline
        result = run_vector_search_pipeline(
            query_embedding=np.asarray(embs[0]),
            index=state.faiss_index,
            metadata=state.faiss_metadata,
            top_k=5, min_similarity=threshold,
        )
        di           = result["decision"]
        person_id    = di.get("person_id")
        api_decision = di.get("decision", "REJECT")
        similarity   = di.get("similarity", 0.0)
        trust_status = state.trust_db.is_trusted(person_id) if person_id else None

        if trust_status is False:
            api_decision = "REJECT"

        rec = state.recommender.decide(
            decision=api_decision, similarity=similarity,
            trust_status=trust_status, person_id=person_id,
        )

        # الأكشن: force_action لو محدد، وإلا primary_action من الـ engine
        chosen_action = force_action or rec["primary_action"]

        img        = cv2.imread(tmp)
        result_img = state.recommender.execute(chosen_action, img, strength=strength)

        headers = {
            "X-Request-Id":     rid,
            "X-Decision":       api_decision,
            "X-Person-Id":      person_id or "unknown",
            "X-Similarity":     str(round(similarity * 100, 2)),
            "X-Trust-Status":   str(trust_status),
            "X-Action-Applied": chosen_action,
            "X-Reason":         rec["reason"],
        }
        
        with mlflow.start_run(run_name=f"blur_{rid}"):
            mlflow.set_tracking_uri("http://mlflow:5000")
            mlflow.set_experiment("FaceProtect")
            mlflow.set_tag("endpoint", "/smart_protect")
            mlflow.set_tag("track", "MLOps-Track6")
    
            mlflow.log_param("threshold", threshold)
            mlflow.log_param("decision", api_decision)
    
            mlflow.log_metric("similarity_score", float(di.get("similarity", 0.0) * 100))
            mlflow.log_metric("decision_binary", 1 if api_decision == "ACCEPT" else 0)
            mlflow.log_metric("blur_applied", 1 if api_decision == "REJECT" else 0)



        return image_to_stream(result_img, filename=f"protected_{rid}.jpg")

    finally:
        if os.path.exists(tmp): os.remove(tmp)


# ── 5. Verify + Apply — JSON + صورة منفصلين في نفس الـ workflow ─────────────

@app.post(
    "/verify_and_apply",
    tags=["Privacy Actions"],
    summary="Verify → JSON decision + processed image URL hint",
)
async def verify_and_apply(
    file: UploadFile = File(...),
    min_similarity: float = Query(None),
    strength: int = Query(31, ge=5, le=81),
):
    """
    زي /smart_protect بالظبط بس بيرجع JSON فيه:
    - كل نتائج الـ verify
    - الصورة المعدّلة كـ base64 مدمجة في الـ response
    مفيد للـ frontend اللي محتاج الاتنين مع بعض.
    """
    require_faiss(); require_model()
    import base64
    threshold = min_similarity or state.evaluation_threshold
    tmp = await save_upload(file)
    rid = os.path.basename(tmp).split(".")[0]

    try:
        embs, _ = state.face_model.extract([tmp])
        if not embs:
            raise HTTPException(422, "No face embedding extracted.")

        from embeddings.faiss_index import run_vector_search_pipeline
        result = run_vector_search_pipeline(
            query_embedding=np.asarray(embs[0]),
            index=state.faiss_index,
            metadata=state.faiss_metadata,
            top_k=5, min_similarity=threshold,
        )
        di           = result["decision"]
        person_id    = di.get("person_id")
        api_decision = di.get("decision", "REJECT")
        similarity   = di.get("similarity", 0.0)
        trust_status = state.trust_db.is_trusted(person_id) if person_id else None

        if trust_status is False:
            api_decision = "REJECT"

        rec = state.recommender.decide(
            decision=api_decision, similarity=similarity,
            trust_status=trust_status, person_id=person_id,
        )

        chosen_action = rec["primary_action"]
        img           = cv2.imread(tmp)
        result_img    = state.recommender.execute(chosen_action, img, strength=strength)

        _, buf = cv2.imencode(".jpg", result_img)
        img_b64 = base64.b64encode(buf.tobytes()).decode()

        return {
            "request_id":        rid,
            "decision":          api_decision,
            "similarity_percent": round(similarity * 100, 2),
            "person_id":         person_id,
            "trust_status":      trust_status,
            "recommendation":    rec,
            "action_applied":    chosen_action,
            "processed_image_b64": img_b64,
        }
    finally:
        if os.path.exists(tmp): os.remove(tmp)


# =============================
# TRUST LIST ENDPOINTS
# =============================

class TrustEntry(BaseModel):
    person_id: str  = Field(..., description="ID في قاعدة الـ embeddings")
    label:     str  = Field(..., description="اسم للعرض")
    trusted:   bool = Field(..., description="True=مسموح | False=محظور")
    owner:     str  = Field("",  description="اسم العميل")
    notes:     str  = Field("",  description="ملاحظات")


@app.get( "/trust",             tags=["Trust Lists"])
def list_trust(owner: Optional[str] = None):
    return state.trust_db.list_all(owner=owner)

@app.post("/trust",             tags=["Trust Lists"], status_code=201)
def add_trust(e: TrustEntry):
    rec = state.trust_db.add(e.person_id, e.label, e.trusted, e.owner, e.notes)
    return {"message": f"'{e.person_id}' is now {'trusted' if e.trusted else 'blocked'}.", "record": rec}

@app.get( "/trust/{person_id}", tags=["Trust Lists"])
def get_trust(person_id: str):
    r = state.trust_db.get(person_id)
    if not r: raise HTTPException(404, f"'{person_id}' not found.")
    return r

@app.delete("/trust/{person_id}", tags=["Trust Lists"])
def delete_trust(person_id: str):
    if not state.trust_db.remove(person_id):
        raise HTTPException(404, f"'{person_id}' not found.")
    return {"message": f"'{person_id}' removed."}

@app.patch("/trust/{person_id}/block",   tags=["Trust Lists"])
def block_person(person_id: str):
    r = state.trust_db.get(person_id)
    if not r: raise HTTPException(404, "Add the person first via POST /trust.")
    r["trusted"] = False; state.trust_db.add(**r)
    return {"message": f"'{person_id}' BLOCKED."}

@app.patch("/trust/{person_id}/unblock", tags=["Trust Lists"])
def unblock_person(person_id: str):
    r = state.trust_db.get(person_id)
    if not r: raise HTTPException(404, "Not found.")
    r["trusted"] = True; state.trust_db.add(**r)
    return {"message": f"'{person_id}' TRUSTED."}


# ===========================
# SYSTEM
# ===========================

@app.get("/stats", tags=["System"])
def get_stats():
    trust_all = state.trust_db.list_all()
    return {
        "total_embeddings": state.faiss_index.ntotal if state.faiss_index else 0,
        "trust_entries":    len(trust_all),
        "trusted_count":    sum(1 for r in trust_all if     r["trusted"]),
        "blocked_count":    sum(1 for r in trust_all if not r["trusted"]),
        "threshold":        state.evaluation_threshold,
        "available_actions": list(ACTION_EXECUTORS.keys()),
    }

@app.get("/threshold", tags=["System"])
def get_threshold():
    return {"threshold": state.evaluation_threshold, "source": "evaluation_report.json"}
