# TennisAI Trainer — Backend

Análisis de técnica de tenis mediante visión computacional y IA. Plataforma SaaS que genera reportes biomecánicos y planes de entrenamiento personalizados.

---

## 📋 Estructura del Proyecto

```
tennis-ai-backend/
├── README.md                           # Este archivo
├── requirements.txt                    # Dependencias Python
├── .env.example                        # Variables de entorno de ejemplo
│
├── vision_pipeline/                    # Procesamiento de video
│   ├── vision_pipeline_v3.py           # Pipeline principal de análisis visual
│   ├── helpers.py                      # Utilidades (Supabase, MediaPipe)
│   └── README.md                       # Documentación específica
│
├── agents_pipeline/                    # Pipeline de análisis con LLM
│   ├── agents_pipeline_v8.py           # Orquestador principal
│   ├── agent_coordinator.py            # Coordinador de flujo
│   ├── agent_specialists.py            # Agentes de análisis (forehand, backhand, saque)
│   ├── agent_synthesizer.py            # Síntesis de reportes
│   ├── agent_coach.py                  # Recomendaciones de coaching
│   ├── coordinator_precompute.py       # Pre-cálculos sin LLM
│   ├── bone_mapping_builder.py         # Constructor de mapeo óseo
│   └── README.md                       # Documentación específica
│
├── deployment/                         # Configuración de despliegue
│   ├── modal_secrets_setup.sh          # Setup de secretos en Modal
│   └── deployment_guide.md             # Guía de despliegue
│
└── docs/                               # Documentación adicional
    ├── architecture.md                 # Arquitectura del sistema
    ├── data_model.md                   # Modelo de datos (Supabase)
    └── troubleshooting.md              # Guía de solución de problemas
```

---

## 🚀 Quick Start

### 1. Clonar y configurar

```bash
git clone https://github.com/aachondo1/TennisAI_Backend.git
cd TennisAI_Backend
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
# Completar en .env:
#   - SUPABASE_URL
#   - SUPABASE_SERVICE_KEY
#   - ANTHROPIC_API_KEY
```

### 3. Desplegar en Modal

```bash
# Vision pipeline
modal deploy vision_pipeline/vision_pipeline_v3.py

# Agents pipeline
modal deploy agents_pipeline/agents_pipeline_v8.py
```

---

## 📦 Componentes Principales

### Vision Pipeline (`vision_pipeline/`)

**Función:** Procesa videos de tenis y extrae datos biomecánicos (landmarks, trayectoria de pelota, ángulos).

**Stack:**
- **MediaPipe 0.10.14** — Detección de pose (25 landmarks)
- **YOLOv8n** — Detección de jugador/cancha
- **Ball Tracker** (HuggingFace `RJTPP/tennis-ball-detection`) — Rastreo de pelota
- **FFmpeg** — Procesamiento de video

**Archivos:**
- `vision_pipeline_v3.py` — Pipeline principal (CPU triage → GPU clip processing → merge)
- `helpers.py` — Utilidades de MediaPipe y Supabase

**Entrada:** Video MP4 (estándar: cámara trasera, 2m+ altura en baseline)
**Salida:** 
- `vision_results` table en Supabase (landmarks, ball trajectory)
- `sessions` table (campos de calidad: `mediapipe_coverage`, `ball_sync_rate`, `processing_gaps_percent`)

---

### Agents Pipeline (`agents_pipeline/`)

**Función:** Analiza datos visuales con LLM multi-agente generando reportes biomecánicos y recomendaciones de coaching.

**Arquitectura de 6 agentes:**

```
Coordinator
    ↓
├─ Forehand Agent (paralelo)
├─ Backhand Agent (paralelo)
└─ Saque Agent (paralelo)
    ↓
Synthesizer Agent
    ↓
Coach Agent
```

**Archivos:**

| Archivo | Responsabilidad |
|---------|-----------------|
| `agents_pipeline_v8.py` | Orquestador: endpoints HTTP, flujos, Supabase post |
| `agent_coordinator.py` | Lógica de coordinación y enrutamiento |
| `agent_specialists.py` | Forehand, Backhand, Saque (evaluación por golpe) |
| `agent_synthesizer.py` | Síntesis de reporte, análisis cruzado, delta vs sesión anterior |
| `agent_coach.py` | Generación de plan de entrenamiento |
| `coordinator_precompute.py` | Pre-cálculos sin LLM (estadísticas, contexto táctico) |
| `bone_mapping_builder.py` | Construcción dinámica del mapeo óseo |
| `helpers.py` | Utilidades (Supabase, Anthropic) |

**Entrada:** `vision_results` + `profiles` + sesión anterior (opcional)
**Salida:** 
- `synthesizer_metadata` → JSON con insights, root cause, deltas
- Narrative report (texto)

---

## 🛠️ Stack Técnico

### Dependencias Críticas

```
Backend:
  - Python 3.11
  - modal-client >= 1.0 (serverless GPU/CPU)
  - anthropic >= 0.40.0 (Claude API)
  - httpx >= 0.27.0 (HTTP cliente)

Vision:
  - torch >= 2.4.0
  - ultralytics >= 8.2.0 (YOLOv8)
  - mediapipe >= 0.10.14
  - opencv-python-headless
  - huggingface_hub >= 0.34.0 (ball tracker download)
  - ffmpeg (sistema)

Datos:
  - Supabase PostgreSQL (RLS + check constraints)
```

### Modelos LLM

- **Claude Sonnet** — Análisis especializado, síntesis, coaching (vía Anthropic API)

### Modelos Visión

- **MediaPipe Pose** — 25 landmarks (cabeza, torso, brazos, piernas)
- **YOLOv8n** — Detección de jugador/cancha
- **RJTPP/tennis-ball-detection** — Ball tracker vía HuggingFace Hub

---

## 📊 Modelo de Datos

**Tablas Supabase:**

1. **sessions** — Metadatos de sesión
   - `id`, `user_id`, `session_type` (clase/paleteo/partido)
   - `created_at`, `actual_session_date` (capturada del flujo de carga)
   - `status` (processing/complete)
   - Campos de calidad: `mediapipe_coverage`, `ball_sync_rate`, `processing_gaps_percent`

2. **vision_results** — Datos brutos del análisis visual
   - `id`, `session_id`, `stroke_type`, `frame_index`
   - `landmarks` (JSON 25 puntos), `ball_xy`, `angles` (codo, hombro, cadera)
   - `quality_metrics` (coverage, ball_sync)

3. **synthesizer_metadata** — Salida del synthesizer agent
   - `id`, `session_id`
   - `top_3_insights`, `root_cause`, `delta_headline`, `fatigue_analysis` (JSON)
   - `quality_score` (0–100)

4. **profiles** — Datos del jugador
   - `id`, `dominant_hand` (right/left), `nivel` (principiante/intermedio/avanzado)
   - Grip style, heights, anthropometric data

5. **profesor_alumnos** — Relación profesor–alumno para dashboard B2B

**RLS:** Service role key (`sb_secret_...` o legacy `eyJ...`) requiere permisos de inserción/actualización.

---

## 🔐 Secretos & Variables de Entorno

### Modal Secrets

```bash
# Setup inicial
modal secret create supabase-key \
  SUPABASE_URL=https://dqsjprndqltuwfilevmb.supabase.co \
  SUPABASE_SERVICE_KEY=eyJ... # Legacy service key

modal secret create anthropic-key \
  ANTHROPIC_API_KEY=sk-...
```

### `.env` (Local Development)

```env
SUPABASE_URL=https://dqsjprndqltuwfilevmb.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
ANTHROPIC_API_KEY=sk-...
MODAL_WORKSPACE_NAME=tu_workspace
```

---

## 🔄 Flujos de Datos

### 1. Upload Video → Vision Analysis

```
Frontend (Bolt)
  ↓ FormData (video, session_type)
  ↓
Vision Pipeline (Modal)
  ├─ CPU: triage, frame extraction, FPS normalization
  ├─ GPU: MediaPipe pose (batches of 8 frames)
  ├─ Merge: consolidate landmarks + ball track
  └─ Supabase: INSERT vision_results, UPDATE sessions quality_metrics
  ↓
Frontend: Poll sessions table → vision_results ready
```

### 2. Agents Analysis

```
Coordinator
  ├─ Load vision_results + profiles
  ├─ Precompute: stats, tactical context, fatigue
  ├─ Dispatch to specialists (forehand, backhand, saque) in parallel
  ├─ Synthesizer: merge + delta analysis + root cause
  └─ Coach: generate training plan
  ↓
Supabase: UPSERT synthesizer_metadata, INSERT coach_comments
```

### 3. Frontend Rendering

```
Report.tsx
  ├─ JOIN sessions + synthesizer_metadata + vision_results
  ├─ Display: quality score, insights, deltas, coaching plan
  └─ Download PDF (jspdf/react-pdf)
```

---

## ⚠️ Patrones Críticos

### 1. Service Role vs. Anon Key

- **Modal (agents/vision):** Use `SUPABASE_SERVICE_KEY` (legacy `eyJ...` format)
- **RLS:** Bypassed by service key; newer `sb_secret_...` format does NOT bypass RLS

### 2. Session Type Enum Validation

Frontend y backend deben coincidir:
```python
session_type in ['clase', 'paleteo', 'partido']  # ✅
```
Supabase check constraint rechazará silenciosamente otros valores.

### 3. Float vs. Int Frame Indices

- MediaPipe devuelve frames como **float** (0.0, 1.0, ...)
- Specialist agents **filtran por int** (frame_index % 1 == 0)
- Comparativas entre golpes requieren int indexing explícito

### 4. Grip Calibration Dinámico

- Elbow angle ranges varían por grip style (eastern/semi-western/western)
- Western grip: 140–160° (no 90–120° estándar)
- Agent coordinador detecta grip automáticamente y comunica a especialistas

### 5. Synthesizer JSON Truncation

- Claude respuesta única → token limit hit → JSON truncado
- **Solución:** Dos prompts separados (structured JSON + narrative)
- `max_tokens` debe dimensionarse per agent

### 6. Modal Cross-App Calls

- Use `.spawn()` no `.remote()` para cross-deployment
- `modal.Function.from_name()` + `.spawn()` → asincrónico
- `.remote()` bloquea sincrónico → timeouts en serverless

---

## 🧪 Testing & Deployment

### Local Testing (Sin Modal)

```python
# En agents_pipeline/
python -c "
import coordinator_precompute as cp
data = cp.compute_stroke_stats({'landmarks': [...]})
print(data)
"
```

### Modal Deployment

```bash
# Vision pipeline (GPU, ~10 min)
modal deploy vision_pipeline/vision_pipeline_v3.py

# Agents pipeline (lightweight, ~2 min)
modal deploy agents_pipeline/agents_pipeline_v8.py

# Verificar status
modal app list
modal logs tennis-vision-pipeline-v3
```

### Monitoreo en Producción

```bash
# Logs en tiempo real
modal logs tennis-agents-pipeline --follow

# Estadísticas de uso
modal workspace logs
```

---

## 📝 Convenciones de Código

### Python

```python
# Nombres de variable: snake_case
frame_landmarks = []
ball_xy_trajectory = {}

# Constantes: SCREAMING_SNAKE_CASE
MEDIAPIPE_LANDMARK_COUNT = 25
EXPECTED_FPS = 30

# Tipos: use type hints
def process_vision(session_id: str, landmarks: list[dict]) -> dict:
    pass
```

### JSON Schema (Supabase)

```json
{
  "stroke_type": "forehand",
  "frame_index": 42,
  "angles": {
    "elbow_flexion": 115,
    "shoulder_abduction": 85
  },
  "quality": {
    "mediapipe_coverage": 0.92,
    "ball_sync_rate": 0.88
  }
}
```

---

## 🚨 Troubleshooting

### Problem: Vision pipeline timeout

```
→ Check GPU queue in Modal
→ Reduce batch size (actualmente 8 frames/batch)
→ Verificar FFmpeg install en imagen
```

### Problem: Supabase INSERT falla silenciosamente

```
✅ Verificar SUPABASE_SERVICE_KEY es legacy eyJ... format
✅ Validar session_type in ['clase', 'paleteo', 'partido']
✅ Checar RLS policies permiten service role
```

### Problem: Agent responde truncado

```
✅ Aumentar max_tokens en prompt
✅ Dividir en dos llamadas (JSON + narrative)
✅ Verificar que input no excede 100k tokens
```

---

## 🔗 Links Útiles

- **Modal Docs:** https://modal.com/docs
- **Anthropic API:** https://docs.anthropic.com
- **Supabase:** https://supabase.com/docs
- **MediaPipe:** https://google.github.io/mediapipe
- **Repository:** https://github.com/aachondo1/TennisAI_Backend

---

## 📄 Licencia

Propietario — TennisAI Trainer (2025)

---

## ✉️ Contacto

Juan Alberto — Backend Lead
- Email: juan@tennisai.dev
- GitHub: @aachondo1

**Last Updated:** Abril 2025
