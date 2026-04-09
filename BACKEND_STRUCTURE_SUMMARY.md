# 📐 Estructura Final del Repositorio Backend — Sumario Visual

## 🎯 TennisAI_Backend (GitHub Repository)

```
TennisAI_Backend/
│
├── 📄 README.md                    ← Documentación principal (INCLUIDA)
├── 📄 requirements.txt             ← Dependencias Python (INCLUIDA)
├── 📄 .env.example                 ← Template de variables de entorno (INCLUIDA)
├── 📄 .gitignore                   ← Git ignore (INCLUIDA)
├── 📄 LICENSE                      ← Licencia (crear si es necesario)
│
│
├── 📁 vision_pipeline/             ← PIPELINE DE VISIÓN
│   │
│   ├── 📜 vision_pipeline_v3.py                    ← COPIAR: vision_pipeline_v3__13_.py
│   │                                                  (CPU triage → GPU MediaPipe → merge)
│   │
│   ├── 📜 helpers.py                              ← COPIAR: helpers.py
│   │                                                  (MediaPipe utils, Supabase ops)
│   │
│   └── 📄 README.md                               ← CREAR NUEVO
│                                                     (Docs específicas del vision pipeline)
│
│
├── 📁 agents_pipeline/             ← PIPELINE DE AGENTES (MULTI-LLM)
│   │
│   ├── 📜 agents_pipeline_v8.py                   ← COPIAR: agents_pipeline_v8__13_.py
│   │                                                  (Orquestador: endpoints HTTP, flujos, Supabase)
│   │
│   ├── 📜 agent_coordinator.py                    ← COPIAR: agent_coordinator.py
│   │                                                  (Coordinación de agentes)
│   │
│   ├── 📜 agent_specialists.py                    ← COPIAR: agent_specialists__7___1_.py
│   │                                                  (Forehand, backhand, saque agents)
│   │
│   ├── 📜 agent_synthesizer.py                    ← COPIAR: agent_synthesizer__2_.py
│   │                                                  (Síntesis de reporte, root cause, deltas)
│   │
│   ├── 📜 agent_coach.py                          ← CREAR/COPIAR si existe
│   │                                                  (Plan de entrenamiento)
│   │
│   ├── 📜 coordinator_precompute.py               ← COPIAR: coordinator_precompute.py
│   │                                                  (Pre-cálculos sin LLM: stats, tactics)
│   │
│   ├── 📜 bone_mapping_builder.py                 ← COPIAR: bone_mapping_builder__2_.py
│   │                                                  (Mapeo dinámico de articulaciones)
│   │
│   └── 📄 README.md                               ← CREAR NUEVO
│                                                     (Docs específicas del agents pipeline)
│
│
├── 📁 docs/                        ← DOCUMENTACIÓN
│   │
│   ├── 📄 architecture.md                         ← CREAR NUEVO
│   │                                                  (Diagrama, flujos, componentes)
│   │
│   ├── 📄 data_model.md                           ← CREAR NUEVO
│   │                                                  (Schema de Supabase, relaciones)
│   │
│   ├── 📄 deployment_guide.md                     ← CREAR NUEVO
│   │                                                  (Step-by-step: Modal setup)
│   │
│   ├── 📄 troubleshooting.md                      ← CREAR NUEVO
│   │                                                  (FAQ, issues comunes, soluciones)
│   │
│   └── 📄 roadmap.md                              ← CREAR NUEVO
│                                                     (Features en desarrollo, timeline)
│
│
├── 📁 deployment/                  ← CONFIGURACIÓN DE DESPLIEGUE
│   │
│   ├── 📜 modal_setup.sh                          ← CREAR NUEVO
│   │                                                  (Script para setup de secretos)
│   │
│   └── 📄 deployment_guide.md                     ← CREAR NUEVO
│                                                     (Instrucciones detalladas)
│
│
└── 📁 .github/                     ← GITHUB WORKFLOWS (OPCIONAL)
    │
    └── 📁 workflows/
        └── 📄 deploy.yml                          ← CREAR si necesitas CI/CD
                                                       (GitHub Actions)
```

---

## 📊 Resumen de Archivos

### ✅ Archivos a COPIAR (Existentes)

| De | Para | Descripción |
|---|---|---|
| `vision_pipeline_v3__13_.py` | `vision_pipeline/vision_pipeline_v3.py` | Vision pipeline principal |
| `helpers.py` | `vision_pipeline/helpers.py` | Utilidades comunes |
| `agents_pipeline_v8__13_.py` | `agents_pipeline/agents_pipeline_v8.py` | Orquestador de agentes |
| `agent_coordinator.py` | `agents_pipeline/agent_coordinator.py` | Coordinador de flujo |
| `agent_specialists__7___1_.py` | `agents_pipeline/agent_specialists.py` | Agentes especializados |
| `agent_synthesizer__2_.py` | `agents_pipeline/agent_synthesizer.py` | Síntesis de reporte |
| `coordinator_precompute.py` | `agents_pipeline/coordinator_precompute.py` | Pre-cálculos |
| `bone_mapping_builder__2_.py` | `agents_pipeline/bone_mapping_builder.py` | Mapeo óseo |

### ✅ Archivos a CREAR o COPIAR (Configuración)

| Archivo | Ubicación | Estado |
|---------|-----------|--------|
| `README.md` | `/` | **INCLUIDA** en `/home/claude/TennisAI_Backend_README.md` |
| `requirements.txt` | `/` | **INCLUIDA** en `/home/claude/requirements.txt` |
| `.env.example` | `/` | **INCLUIDA** en `/home/claude/.env.example` |
| `.gitignore` | `/` | **INCLUIDA** en `/home/claude/gitignore` |
| `vision_pipeline/README.md` | `vision_pipeline/` | **A CREAR** (template en QUICK_SETUP_GUIDE.md) |
| `agents_pipeline/README.md` | `agents_pipeline/` | **A CREAR** (template en QUICK_SETUP_GUIDE.md) |
| `docs/architecture.md` | `docs/` | **A CREAR** (template en QUICK_SETUP_GUIDE.md) |
| `docs/data_model.md` | `docs/` | **A CREAR** (usar schema de Supabase) |
| `docs/deployment_guide.md` | `docs/` | **A CREAR** |
| `docs/troubleshooting.md` | `docs/` | **A CREAR** |

### ❌ Archivos a NO INCLUIR

| Archivo | Por Qué |
|---------|---------|
| `TennisAI_Proyecto.docx` | Documento Word (no es código) |
| `TennisAI_Proyecto_v2.docx` | Documento Word (no es código) |
| `vision_pipeline_v2__12_.py` | Versión deprecated |
| `report_updated.py` | Frontend related (en Report.tsx) |
| `vision_pipeline_improvements_roadmap.md` | Incluir contenido en `docs/roadmap.md` |

---

## 🔄 Proceso Paso a Paso

### 1️⃣ Preparación Local

```bash
# En tu máquina
mkdir -p ~/project_setup/TennisAI_Backend
cd ~/project_setup/TennisAI_Backend

# Crear estructura
mkdir -p vision_pipeline agents_pipeline docs deployment
```

### 2️⃣ Copiar Archivos Python

```bash
# Vision pipeline
cp /ruta/vision_pipeline_v3__13_.py vision_pipeline/vision_pipeline_v3.py
cp /ruta/helpers.py vision_pipeline/

# Agents pipeline
cp /ruta/agents_pipeline_v8__13_.py agents_pipeline/agents_pipeline_v8.py
cp /ruta/agent_coordinator.py agents_pipeline/
cp /ruta/agent_specialists__7___1_.py agents_pipeline/agent_specialists.py
cp /ruta/agent_synthesizer__2_.py agents_pipeline/agent_synthesizer.py
cp /ruta/coordinator_precompute.py agents_pipeline/
cp /ruta/bone_mapping_builder__2_.py agents_pipeline/bone_mapping_builder.py
```

### 3️⃣ Copiar Archivos de Config

```bash
# Raíz
cp /home/claude/TennisAI_Backend_README.md README.md
cp /home/claude/requirements.txt .
cp /home/claude/.env.example .
cp /home/claude/gitignore .gitignore

# Docs (como referencia)
cp /home/claude/BACKEND_FILE_MAPPING.md docs/file_structure.md
```

### 4️⃣ Crear README.md para cada pipeline

Ver templates en `QUICK_SETUP_GUIDE.md` → Paso 5

### 5️⃣ Crear archivo de arquitectura

Ver template en `QUICK_SETUP_GUIDE.md` → Paso 5

### 6️⃣ Git

```bash
git init
git add .
git commit -m "Initial backend structure: vision & agents pipelines"
git branch -M main
git remote add origin https://github.com/aachondo1/TennisAI_Backend.git
git push -u origin main
```

---

## 📦 Tamaño Estimado del Repo

```
vision_pipeline/          ~100 KB (vision_pipeline_v3.py ~50KB + helpers.py ~10KB)
agents_pipeline/          ~200 KB (agents_pipeline_v8.py ~40KB + specialistas/synthesizer ~160KB)
docs/                     ~50 KB
Total (sin .git):         ~350 KB
```

---

## 🔐 Secretos Requeridos (NO IR EN REPO)

```
Modal:
  - supabase-key → SUPABASE_URL, SUPABASE_SERVICE_KEY
  - anthropic-key → ANTHROPIC_API_KEY

.env (local, .gitignored):
  - SUPABASE_URL
  - SUPABASE_SERVICE_KEY
  - ANTHROPIC_API_KEY
```

---

## 📚 Documentación Generada

Los siguientes archivos están listos en `/home/claude/`:

1. ✅ **TennisAI_Backend_README.md** — README principal (copiar a repo)
2. ✅ **BACKEND_FILE_MAPPING.md** — Mapeo de archivos (para referencia)
3. ✅ **QUICK_SETUP_GUIDE.md** — Guía paso-a-paso (para setup)
4. ✅ **requirements.txt** — Dependencias (copiar a repo)
5. ✅ **gitignore** — Git ignore rules (copiar como .gitignore)
6. ✅ **.env.example** — Template de variables (copiar a repo)

---

## ✨ Siguientes Pasos Recomendados

1. **Copiar todos los archivos** siguiendo QUICK_SETUP_GUIDE.md
2. **Crear repo en GitHub** (privado)
3. **Push inicial** con estructura base
4. **Documentación específica** por pipeline (vision/agents)
5. **Setup de secretos** en Modal
6. **Deploy inicial** para validar
7. **Enable branch protection** en GitHub

---

## 🚀 Deploy Checklist

```bash
# Una vez en producción
✅ git push origin main
✅ modal secret create supabase-key ...
✅ modal secret create anthropic-key ...
✅ modal deploy vision_pipeline/vision_pipeline_v3.py
✅ modal deploy agents_pipeline/agents_pipeline_v8.py
✅ Verificar logs en Modal dashboard
✅ Test endpoint HTTP desde Frontend
```

---

**¿Listo para subir? Usa QUICK_SETUP_GUIDE.md paso a paso.**
