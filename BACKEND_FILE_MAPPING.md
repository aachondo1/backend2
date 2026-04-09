# 📁 TennisAI Backend — Mapeo de Archivos

## ✅ Archivos a Incluir en el Repositorio

### 1. **Vision Pipeline** (`vision_pipeline/`)

| Archivo Actual | → | Nuevo Nombre en Git | Ubicación | Descripción |
|---|---|---|---|---|
| `vision_pipeline_v3__13_.py` | → | `vision_pipeline_v3.py` | `vision_pipeline/` | Pipeline principal: CPU triage → GPU MediaPipe → merge |
| `helpers.py` | → | `helpers.py` | `vision_pipeline/` | Utilidades: MediaPipe, Supabase, procesamiento |
| — | → | `README.md` | `vision_pipeline/` | Documentación específica del pipeline de visión |

**Descripción:** Recibe videos MP4, extrae landmarks (MediaPipe), rastreo de pelota (YOLOv8 + HF ball tracker), y persiste en `vision_results` table de Supabase.

---

### 2. **Agents Pipeline** (`agents_pipeline/`)

| Archivo Actual | → | Nuevo Nombre en Git | Ubicación | Descripción |
|---|---|---|---|---|
| `agents_pipeline_v8__13_.py` | → | `agents_pipeline_v8.py` | `agents_pipeline/` | Orquestador principal: endpoints HTTP, flujos de agentes, Supabase |
| `agent_coordinator.py` | → | `agent_coordinator.py` | `agents_pipeline/` | Lógica de coordinación: enrutamiento, flujo entre agentes |
| `agent_specialists__7___1_.py` | → | `agent_specialists.py` | `agents_pipeline/` | Agentes especializados: forehand, backhand, saque (análisis por golpe) |
| `agent_synthesizer__2_.py` | → | `agent_synthesizer.py` | `agents_pipeline/` | Síntesis de reporte: insights, root cause, delta analysis, fatigue |
| `coordinator_precompute.py` | → | `coordinator_precompute.py` | `agents_pipeline/` | Pre-cálculos sin LLM: estadísticas, contexto táctico, quality report |
| `bone_mapping_builder__2_.py` | → | `bone_mapping_builder.py` | `agents_pipeline/` | Constructor dinámico del mapeo óseo (landmarks → articulaciones) |
| — | → | `agent_coach.py` | `agents_pipeline/` | Generación de plan de entrenamiento personalizado |
| — | → | `README.md` | `agents_pipeline/` | Documentación específica del pipeline de agentes |

**Descripción:** Análisis multi-agente con Claude LLM. 6-agent system: coordinator → forehand/backhand/saque (paralelo) → synthesizer → coach.

---

### 3. **Root** (Raíz del repositorio)

| Archivo | Ubicación | Descripción |
|---------|-----------|---|
| (README.md actual) | `/` | Documentación principal del proyecto |
| `requirements.txt` | `/` | Dependencias Python consolidadas |
| `.env.example` | `/` | Template de variables de entorno |
| `.gitignore` | `/` | Git ignore (venv, .env, __pycache__, etc.) |
| `LICENSE` | `/` | Licencia del proyecto |

---

### 4. **Documentation** (`docs/`)

| Archivo | Ubicación | Descripción |
|---------|-----------|---|
| `architecture.md` | `docs/` | Diagrama de arquitectura, flujos de datos, componentes |
| `data_model.md` | `docs/` | Schema de Supabase, relaciones, RLS policies |
| `deployment_guide.md` | `docs/` | Step-by-step para desplegar en Modal |
| `troubleshooting.md` | `docs/` | FAQ, issues comunes, soluciones |

---

### 5. **Deployment** (`deployment/`)

| Archivo | Ubicación | Descripción |
|---------|-----------|---|
| `modal_setup.sh` | `deployment/` | Script para setup de secretos en Modal |
| `requirements_modal.txt` | `deployment/` | Deps específicas para entornos Modal (si difieren) |

---

## ❌ Archivos a NO Incluir

| Archivo | Por Qué |
|---------|---------|
| `TennisAI_Proyecto.docx` | Documento Word (no es código) |
| `TennisAI_Proyecto_v2.docx` | Documento Word (no es código) |
| `vision_pipeline_v2__12_.py` | Versión anterior deprecada |
| `report_updated.py` | Frontend related (integrado en Report.tsx) |
| `vision_pipeline_improvements_roadmap.md` | Interno, incluir en `docs/roadmap.md` si es necesario |

---

## 📦 Estructura Final del Repositorio

```
TennisAI_Backend/
│
├── README.md                          # Documentación principal
├── LICENSE                            # Licencia (propietario o MIT)
├── .gitignore                         # Git ignore
├── requirements.txt                   # Dependencias consolidadas
├── .env.example                       # Template de .env
│
├── vision_pipeline/
│   ├── vision_pipeline_v3.py          # Pipeline principal
│   ├── helpers.py                     # Utilidades
│   └── README.md                      # Docs específicas
│
├── agents_pipeline/
│   ├── agents_pipeline_v8.py          # Orquestador
│   ├── agent_coordinator.py           # Coordinador
│   ├── agent_specialists.py           # Forehand, backhand, saque
│   ├── agent_synthesizer.py           # Síntesis
│   ├── agent_coach.py                 # Coaching
│   ├── coordinator_precompute.py      # Pre-cálculos
│   ├── bone_mapping_builder.py        # Mapeo óseo
│   └── README.md                      # Docs específicas
│
├── docs/
│   ├── architecture.md                # Arquitectura general
│   ├── data_model.md                  # Schema Supabase
│   ├── deployment_guide.md            # Deploy step-by-step
│   ├── troubleshooting.md             # FAQ
│   └── roadmap.md                     # Roadmap de features
│
├── deployment/
│   ├── modal_setup.sh                 # Setup de secretos Modal
│   └── requirements_modal.txt          # Deps para Modal (si aplica)
│
└── .github/
    └── workflows/
        └── deploy.yml                 # CI/CD workflow (opcional)
```

---

## 🔧 Pasos para Crear el Repositorio

### 1. Crear repo en GitHub

```bash
# En GitHub.com
→ New Repository
→ Nombre: TennisAI_Backend
→ Descripción: Backend para TennisAI Trainer - análisis de técnica de tenis con IA
→ Privado (asumiendo que es propietario)
→ NO inicializar con README (vamos a subirlo nosotros)
```

### 2. Clonar y estructura local

```bash
git clone https://github.com/aachondo1/TennisAI_Backend.git
cd TennisAI_Backend

# Crear estructura de carpetas
mkdir -p vision_pipeline agents_pipeline docs deployment

# Copiar archivos
cp /ruta/vision_pipeline_v3__13_.py vision_pipeline/vision_pipeline_v3.py
cp /ruta/helpers.py vision_pipeline/

cp /ruta/agents_pipeline_v8__13_.py agents_pipeline/agents_pipeline_v8.py
cp /ruta/agent_coordinator.py agents_pipeline/
cp /ruta/agent_specialists__7___1_.py agents_pipeline/agent_specialists.py
cp /ruta/agent_synthesizer__2_.py agents_pipeline/agent_synthesizer.py
cp /ruta/coordinator_precompute.py agents_pipeline/
cp /ruta/bone_mapping_builder__2_.py agents_pipeline/bone_mapping_builder.py
```

### 3. Crear archivos de configuración

#### `requirements.txt`

```
# Core
modal-client>=1.0.0
anthropic>=0.40.0
httpx>=0.27.0

# Vision Pipeline
torch==2.4.0
torchvision==0.19.0
ultralytics==8.2.0
mediapipe==0.10.14
opencv-python-headless
numpy==1.26.4
huggingface_hub==0.34.0

# FastAPI (para local testing, opcional)
fastapi[standard]>=0.104.0
```

#### `.env.example`

```env
# Supabase
SUPABASE_URL=https://dqsjprndqltuwfilevmb.supabase.co
SUPABASE_SERVICE_KEY=eyJ... # Legacy format (sb_secret_... NO funciona en Modal)

# Anthropic API
ANTHROPIC_API_KEY=sk-...

# Modal (opcional)
MODAL_WORKSPACE_NAME=
```

#### `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.venv

# Environment variables
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db

# Logs
*.log
logs/

# Modal
.modal/
modal_volume/
```

### 4. Crear documentación

#### `docs/architecture.md`

```markdown
# Arquitectura de TennisAI Backend

## Visión General

Sistema modular de dos pipelines independientes:

1. **Vision Pipeline** — Procesamiento de video (MediaPipe, YOLO)
2. **Agents Pipeline** — Análisis con LLM (Claude multi-agente)

## Componentes

### Vision Pipeline
- Input: MP4 video (cámara trasera, 2m+ altura)
- Output: landmarks (MediaPipe), ball trajectory (YOLOv8 + HF), angles
- Persiste en: `vision_results` table (Supabase)

### Agents Pipeline
- Input: vision_results + profiles + sesión anterior (opt)
- 6-agent system: coordinator → specialistas (paralelo) → synthesizer → coach
- Output: synthesizer_metadata, coach_comments

## Data Flow

[Diagrama mermaid aquí]
```

### 5. Commit inicial

```bash
git add .
git commit -m "Initial backend structure: vision & agents pipelines"
git push origin main
```

---

## 📋 Checklist Antes de Subir

- [ ] Verificar que `vision_pipeline_v3.py` contiene lógica completa de CPU→GPU
- [ ] Verificar que `agents_pipeline_v8.py` no tiene rutas de frontend
- [ ] Verificar que helpers.py está limpio de dependencias de frontend
- [ ] README.md explica flujos de datos correctamente
- [ ] requirements.txt lista todas las deps (torch, mediapipe, anthropic, etc.)
- [ ] `.env.example` tiene placeholders claros
- [ ] `.gitignore` excluye venv/, .env, __pycache__
- [ ] No hay archivos .docx, archivos viejos, ni código comentado grande
- [ ] URL de GitHub está actualizada en docs

---

## 🚀 Después de Subir

1. **GitHub Settings:**
   - Branch protection: require PR reviews para `main`
   - Add collaborators (equipo)

2. **Documentar en README:**
   ```bash
   ## Deploy
   
   ```bash
   modal deploy agents_pipeline/agents_pipeline_v8.py
   modal deploy vision_pipeline/vision_pipeline_v3.py
   ```
   ```

3. **Generar GitHub Pages:**
   - Enable "GitHub Pages" desde `docs/` folder
   - Docs aparecerán en `https://github.com/aachondo1/TennisAI_Backend/wiki`

---

## 🔄 Versionado

Usar **Semantic Versioning**:
- `v1.0.0` — Release inicial con vision_pipeline_v3 + agents_pipeline_v8
- `v1.1.0` — Nuevas features (e.g., agent_coach completamente integrado)
- `v1.0.1` — Bugfixes menores

Tag en Git:
```bash
git tag -a v1.0.0 -m "Initial backend release"
git push origin v1.0.0
```

---

## 📞 Notas Importantes

1. **Service Key Format:** El `SUPABASE_SERVICE_KEY` DEBE ser formato legacy (`eyJ...`), no `sb_secret_...`
2. **ViTPose Descartado:** Mantener `huggingface_hub` solo para ball tracker
3. **Float vs Int Frames:** Specialist agents filtran por `frame_index % 1 == 0`
4. **Session Type Enum:** Validar que frontend/backend usan los mismos valores
5. **Grip Calibration Dinámico:** Agent coordinator detecta automáticamente
