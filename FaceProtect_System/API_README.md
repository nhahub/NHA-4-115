# FaceProtect — Backend API (Phase 4)

## تشغيل الـ API

```bash
pip install -r api_requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

بعد التشغيل افتح: **http://localhost:8000/docs** — هتلاقي Swagger UI كامل.

---

## الـ Endpoints

### 🔍 Face Recognition
| Method | Endpoint | الوظيفة |
|--------|----------|---------|
| POST | `/verify` | التحقق من وجه ضد قاعدة البيانات |
| POST | `/blur` | طمس الوجوه في صورة |
| POST | `/smart_protect` | تحقق + طمس تلقائي لو اترفض |

### 🛡️ Trust Lists (دوائر الثقة)
| Method | Endpoint | الوظيفة |
|--------|----------|---------|
| GET | `/trust` | عرض كل القوائم |
| POST | `/trust` | إضافة / تعديل شخص |
| GET | `/trust/{person_id}` | حالة شخص محدد |
| DELETE | `/trust/{person_id}` | إزالة من القائمة |
| PATCH | `/trust/{person_id}/block` | حظر سريع |
| PATCH | `/trust/{person_id}/unblock` | رفع الحظر |

### 💡 Recommendations
| Method | Endpoint | الوظيفة |
|--------|----------|---------|
| POST | `/recommend` | توصيات بناءً على قرار التعرف |

### ⚙️ System
| Method | Endpoint | الوظيفة |
|--------|----------|---------|
| GET | `/health` | فحص صحة الـ API |
| GET | `/stats` | إحصائيات قاعدة البيانات |
| GET | `/threshold` | العتبة الحالية للقرار |

---

## سيناريوهات الاستخدام

### 1️⃣ التحقق البسيط
```bash
curl -X POST http://localhost:8000/verify \
  -F "file=@face.jpg" \
  -F "top_k=5"
```

**الرد:**
```json
{
  "request_id": "abc123",
  "decision": "REJECT",
  "similarity_percent": 45.2,
  "distance": 0.548,
  "person_id": null,
  "trust_status": null,
  "recommendation": {
    "primary_action": "auto_blur",
    "reason": "Unknown identity — applying full privacy protection.",
    "actions": [
      {"action_id": "auto_blur", "description": "Blur face automatically", "priority": 1},
      {"action_id": "anonymize_face", "description": "Replace face with avatar/mosaic", "priority": 2}
    ]
  },
  "top_k": [...]
}
```

### 2️⃣ إضافة شخص لقائمة الثقة
```bash
curl -X POST http://localhost:8000/trust \
  -H "Content-Type: application/json" \
  -d '{
    "person_id": "Ahmed_Ali",
    "label": "Ahmed Ali — Employee",
    "trusted": true,
    "owner": "HR_System",
    "notes": "Senior Engineer"
  }'
```

### 3️⃣ حظر شخص
```bash
curl -X POST http://localhost:8000/trust \
  -H "Content-Type: application/json" \
  -d '{"person_id": "Unknown_123", "label": "Blacklisted", "trusted": false, "owner": "Security"}'

# أو بسرعة:
curl -X PATCH http://localhost:8000/trust/Unknown_123/block
```

### 4️⃣ Smart Protect — تحقق + blur تلقائي
```bash
curl -X POST http://localhost:8000/smart_protect \
  -F "file=@photo.jpg" \
  --output protected.jpg
# الـ headers بتحتوي على X-Decision, X-Person-Id, X-Similarity
```

---

## منطق نظام التوصيات

```
الوجه اترفض (REJECT)
├── الشخص محظور (trust=False)     → auto_blur + flag_for_review
├── تشابه ضعيف (sim ≥ 50%)        → auto_blur + retry_with_better_image
└── وجه غير معروف                 → auto_blur + anonymize_face

الوجه اتقبل (ACCEPT)
├── الشخص موثوق (trust=True)      → allow
├── الشخص غير موجود في القائمة   → allow_with_warning + request_consent
└── الشخص محظور (trust=False)    → REJECT → auto_blur [override]
```

---

## ملاحظات
- الـ `trust_lists.json` بيتحفظ في `outputs/trust_lists.json` تلقائياً
- الـ threshold بيتحمل من `outputs/reports/evaluation_report.json` (حالياً: 0.7042)
- كل uploaded images بتتحذف من الـ temp folder بعد المعالجة
