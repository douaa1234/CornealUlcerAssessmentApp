# Streamlit to React + FastAPI migration

## Architecture

The production web app is split into:

- `src/api.py`: FastAPI backend. It owns session state, image upload, model inference, mask mutation, calibration, analysis, report generation, database saving, timeline loading and report downloads.
- `src/app_core.py`: framework-neutral helpers extracted from the Streamlit app. These preserve the original resizing, mask overlay, grid drawing, brush merge, Keras preprocessing/postprocessing and PDF generation behavior.
- `frontend/`: React + Vite frontend. It recreates the Streamlit workflow and uses Fabric.js HTML5 canvas for brush editing, line calibration and grey-reference rectangle selection.

The clinical logic remains in the original modules:

- `src/analysis.py`
- `src/ulcer_unet_infer.py`
- `src/db.py`
- `src/llm_report.py`

## Streamlit component mapping

| Streamlit component | New equivalent |
| --- | --- |
| `st.sidebar` | React `<aside className="sidebar">` |
| `st.tabs(["Workflow", "Guide"])` | React tab buttons with conditional panels |
| `st.file_uploader` | HTML `<input type="file">` posting `multipart/form-data` to `/api/predict` |
| `st.slider` | HTML range inputs |
| `st.radio` | HTML radio inputs |
| `st.selectbox` | HTML `<select>` |
| `st.text_input` | HTML text inputs synced to `/api/session` |
| `st.text_area` | HTML `<textarea>` |
| `st.button` | HTML buttons calling matching API endpoints |
| `st_canvas(..., drawing_mode="freedraw")` | Fabric.js free drawing canvas, exported as RGBA and merged by the original `apply_strokes` logic |
| `st_canvas(..., drawing_mode="line")` | Fabric.js line canvas used for identical mm/px calibration |
| `st_canvas(..., drawing_mode="rect")` | Fabric.js rectangle canvas used for grey reference ROI |
| `st.image` | HTML `<img>` using PNG data URLs returned by the API |
| `st.metric` | React metric blocks |
| `st.dataframe` | HTML table |
| `st.line_chart` | Local SVG line charts |
| `st.download_button` | `/api/report` download response |
| `st.session_state` | Server-side `app_sessions` table keyed by `session_id` |

## Local run

Backend:

```bash
python -m pip install -r requirements.txt
cd src
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5174`.

Optional environment variables:

```bash
export Fluorescein_MODEL_PATH=/absolute/path/to/RealDataModelv2.keras
export WHITE_CKPT_PATH=/absolute/path/to/best.pt
export GROQ_API_KEY=...
export CORS_ORIGINS=http://localhost:5174
export MAX_UPLOAD_MB=200
```

## Deployment

Deploy the backend as a Python web service. Prefer Postgres by setting `DATABASE_URL`; if `DATABASE_URL` is not set, the app uses SQLite under `DATA_DIR`.

Recommended backend environment:

```bash
export DATABASE_URL=postgresql+psycopg://user:password@host:5432/dbname
export DATA_DIR=/persistent/data
export MAX_UPLOAD_MB=200
export CORS_ORIGINS=https://your-frontend.example.com
export Fluorescein_MODEL_PATH=/persistent/models/RealDataModelv2.keras
export WHITE_CKPT_PATH=/persistent/models/best.pt
```

For SQLite fallback, mount a persistent disk and point `DATA_DIR` at it. The app stores `app.db`, SQLite WAL files, audit logs and server-side session rows there.

Example backend start command:

```bash
cd src
uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
```

Build the frontend as static assets:

```bash
cd frontend
VITE_API_URL=https://your-api-host.example.com npm run build
```

Serve `frontend/dist` from any static host. Set backend `CORS_ORIGINS` to the frontend URL.

Model weights must be available on the backend host, either at the existing paths in `src/` or via `Fluorescein_MODEL_PATH` and `WHITE_CKPT_PATH`.

Health check:

```bash
curl https://your-api-host.example.com/api/health
```
