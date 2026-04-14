import React from 'react';
import { createRoot } from 'react-dom/client';
import { Canvas, FabricImage, Line as FabricLine, PencilBrush, Rect } from 'fabric';
import './styles.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const DISPLAY_SIZE = 512;

type Mode = 'Fluorescein' | 'white';
type BrushMode = 'Add' | 'Erase';
type CalMethod = 'Line' | 'Grid (research only)';

type PredictPayload = {
  session_id: string;
  mode: Mode | null;
  step: number;
  mm_per_px?: number;
  rgb: string;
  pred_mask: string;
  edited_mask: string;
  overlay: string;
};

type ResultPayload = {
  result: Record<string, any>;
  summary: Record<string, any>;
  flags: string[];
  report_text: string;
  overlay: string;
  timeline: TimelineRow[];
};

type TimelineRow = {
  visit_date: string;
  eye: string;
  mode: string;
  area_mm2?: number;
  eq_diameter_mm?: number;
  opacity_zscore?: number;
  mm_per_px?: number;
  qc_flags?: string;
  created_at?: string;
};

type ErrorBoundaryState = {
  error: Error | null;
};

class ErrorBoundary extends React.Component<{ children: React.ReactNode }, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error) {
    console.error(error);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="bootError">
          <h2>Something stopped the app from loading.</h2>
          <p>{this.state.error.message}</p>
          <p>Restart the frontend dev server and refresh the page.</p>
        </div>
      );
    }
    return this.props.children;
  }
}

function labelForMode(mode: Mode | null) {
  return mode === 'white' ? 'White-light' : 'Fluorescein';
}

async function jsonFetch(path: string, init?: RequestInit) {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    try {
      throw new Error(JSON.parse(text).detail || text);
    } catch {
      throw new Error(text || res.statusText);
    }
  }
  return res.json();
}

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="metric">
      <div className="metricLabel">{label}</div>
      <div className="metricValue">{value}</div>
    </div>
  );
}

function Stepper({ step }: { step: number }) {
  return (
    <div className="stepper">
      {['Upload', 'Adjust', 'Confirm', 'Calibrate', 'Results'].map((label, index) => (
        <div className="chip" key={label}>
          <span className={`dot ${step >= index + 1 ? 'on' : ''}`} />
          <b>{index + 1}</b> {label}
        </div>
      ))}
    </div>
  );
}

function SimpleLineChart({
  rows,
  series,
  height = 240,
}: {
  rows: TimelineRow[];
  series: { key: keyof TimelineRow; label: string; color: string }[];
  height?: number;
}) {
  const width = 680;
  const pad = { left: 52, right: 18, top: 18, bottom: 42 };
  const values = rows.flatMap((row) =>
    series
      .map((item) => Number(row[item.key]))
      .filter((value) => Number.isFinite(value)),
  );
  if (rows.length < 2 || values.length === 0) {
    return <div className="info">Not enough data to plot yet.</div>;
  }
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const xFor = (index: number) => pad.left + (index / Math.max(rows.length - 1, 1)) * (width - pad.left - pad.right);
  const yFor = (value: number) => pad.top + ((max - value) / (max - min)) * (height - pad.top - pad.bottom);
  const ticks = [max, (max + min) / 2, min];

  return (
    <div className="simpleChartWrap">
      <svg viewBox={`0 0 ${width} ${height}`} className="simpleChart" role="img">
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} className="axis" />
        <line x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} className="axis" />
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={pad.left} y1={yFor(tick)} x2={width - pad.right} y2={yFor(tick)} className="gridLine" />
            <text x={8} y={yFor(tick) + 4} className="chartText">{tick.toFixed(2)}</text>
          </g>
        ))}
        {series.map((item) => {
          const points = rows
            .map((row, index) => {
              const value = Number(row[item.key]);
              return Number.isFinite(value) ? `${xFor(index)},${yFor(value)}` : '';
            })
            .filter(Boolean)
            .join(' ');
          return <polyline key={String(item.key)} points={points} fill="none" stroke={item.color} strokeWidth="2.5" />;
        })}
        {rows.map((row, index) => (
          <text key={`${row.visit_date}-${index}`} x={xFor(index)} y={height - 14} textAnchor="middle" className="chartText">
            {row.visit_date}
          </text>
        ))}
      </svg>
      <div className="legendRow">
        {series.map((item) => (
          <span key={String(item.key)}><i style={{ background: item.color }} />{item.label}</span>
        ))}
      </div>
    </div>
  );
}

function FabricCanvas({
  background,
  mode,
  brush,
  tool,
  onReady,
  onObjectsChange,
}: {
  background: string;
  mode: BrushMode;
  brush: number;
  tool: 'freedraw' | 'line' | 'rect';
  onReady: (canvas: Canvas) => void;
  onObjectsChange?: (canvas: Canvas) => void;
}) {
  const el = React.useRef<HTMLCanvasElement | null>(null);
  const canvasRef = React.useRef<Canvas | null>(null);
  const onReadyRef = React.useRef(onReady);
  const onObjectsChangeRef = React.useRef(onObjectsChange);

  React.useEffect(() => {
    onReadyRef.current = onReady;
  }, [onReady]);

  React.useEffect(() => {
    onObjectsChangeRef.current = onObjectsChange;
  }, [onObjectsChange]);

  const emitObjectsChange = React.useCallback(() => {
    const canvas = canvasRef.current;
    if (canvas && onObjectsChangeRef.current) {
      onObjectsChangeRef.current(canvas);
    }
  }, []);

  React.useEffect(() => {
    if (!el.current) return;
    const canvas = new Canvas(el.current, {
      width: DISPLAY_SIZE,
      height: DISPLAY_SIZE,
      selection: false,
      preserveObjectStacking: true,
    });
    canvasRef.current = canvas;
    onReadyRef.current(canvas);
    return () => {
      canvasRef.current = null;
      try {
        canvas.off();
        void canvas.dispose();
      } catch (error) {
        console.warn('Canvas cleanup skipped', error);
      }
    };
  }, []);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let cancelled = false;
    FabricImage.fromURL(background, { crossOrigin: 'anonymous' }).then((img) => {
      if (cancelled || canvasRef.current !== canvas) return;
      img.scaleToWidth(DISPLAY_SIZE);
      img.scaleToHeight(DISPLAY_SIZE);
      canvas.backgroundImage = img;
      canvas.requestRenderAll();
    }).catch((error) => {
      console.error('Could not load canvas background', error);
    });
    return () => {
      cancelled = true;
    };
  }, [background]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.isDrawingMode = tool === 'freedraw';
    canvas.off('mouse:down');
    canvas.off('mouse:move');
    canvas.off('mouse:up');
    if (tool === 'freedraw') {
      const pencil = new PencilBrush(canvas);
      pencil.color = mode === 'Add' ? '#00E676' : '#FF4DFF';
      pencil.width = brush;
      canvas.freeDrawingBrush = pencil;
    }
    if (tool === 'line') {
      let line: any = null;
      canvas.on('mouse:down', (o) => {
        const p = canvas.getPointer(o.e);
        line = new FabricLine([p.x, p.y, p.x, p.y], {
          stroke: '#00E676',
          strokeWidth: 3,
          selectable: false,
          evented: false,
        });
        clearObjects(canvas);
        canvas.add(line);
        emitObjectsChange();
      });
      canvas.on('mouse:move', (o) => {
        if (!line) return;
        const p = canvas.getPointer(o.e);
        line.set({ x2: p.x, y2: p.y });
        canvas.requestRenderAll();
        emitObjectsChange();
      });
      canvas.on('mouse:up', () => {
        line = null;
        emitObjectsChange();
      });
    }
    if (tool === 'rect') {
      let rect: Rect | null = null;
      let start: { x: number; y: number } | null = null;
      canvas.on('mouse:down', (o) => {
        start = canvas.getPointer(o.e);
        rect = new Rect({
          left: start.x,
          top: start.y,
          width: 0,
          height: 0,
          fill: 'rgba(255,255,255,0)',
          stroke: '#00E676',
          strokeWidth: 2,
          selectable: false,
          evented: false,
        });
        clearObjects(canvas);
        canvas.add(rect);
        emitObjectsChange();
      });
      canvas.on('mouse:move', (o) => {
        if (!rect || !start) return;
        const p = canvas.getPointer(o.e);
        rect.set({
          left: Math.min(start.x, p.x),
          top: Math.min(start.y, p.y),
          width: Math.abs(p.x - start.x),
          height: Math.abs(p.y - start.y),
        });
        canvas.requestRenderAll();
        emitObjectsChange();
      });
      canvas.on('mouse:up', () => {
        rect = null;
        start = null;
        emitObjectsChange();
      });
    }
  }, [mode, brush, tool, emitObjectsChange]);

  return <div className="canvasShell"><canvas ref={el} width={DISPLAY_SIZE} height={DISPLAY_SIZE} /></div>;
}

function getOnlyObjectsDataUrl(canvas: Canvas) {
  const bg = canvas.backgroundImage;
  canvas.backgroundImage = undefined;
  canvas.requestRenderAll();
  const url = canvas.toDataURL({ format: 'png', multiplier: 1 });
  canvas.backgroundImage = bg;
  canvas.requestRenderAll();
  return url;
}

function clearObjects(canvas: Canvas) {
  const objects = canvas.getObjects();
  objects.forEach((object) => {
    try {
      if (canvas.getObjects().includes(object)) canvas.remove(object);
    } catch (error) {
      console.warn('Canvas object cleanup skipped', error);
    }
  });
  canvas.requestRenderAll();
}

function latestLine(canvas: Canvas) {
  const objects = canvas.getObjects();
  const obj: any = objects[objects.length - 1];
  if (!obj || obj.type !== 'line') return null;
  return { x1: obj.x1, y1: obj.y1, x2: obj.x2, y2: obj.y2 };
}

function latestRect(canvas: Canvas) {
  const objects = canvas.getObjects();
  const obj: any = objects[objects.length - 1];
  if (!obj || obj.type !== 'rect') return null;
  return [
    Math.round(obj.left || 0),
    Math.round(obj.top || 0),
    Math.round((obj.width || 0) * (obj.scaleX || 1)),
    Math.round((obj.height || 0) * (obj.scaleY || 1)),
  ];
}

function mmPerPxFromLine(line: { x1: number; y1: number; x2: number; y2: number } | null, knownMm: number) {
  if (!line || !Number.isFinite(knownMm) || knownMm <= 0) return null;
  const pxDist = Math.hypot(line.x2 - line.x1, line.y2 - line.y1);
  return pxDist > 0 ? knownMm / pxDist : null;
}

function App() {
  const [sessionId, setSessionId] = React.useState('');
  const [caseId, setCaseId] = React.useState('');
  const [visitDate, setVisitDate] = React.useState('');
  const [eye, setEye] = React.useState<'Right' | 'Left'>('Right');
  const [modeChoice, setModeChoice] = React.useState<'Fluorescein' | 'White-light'>('Fluorescein');
  const [mode, setMode] = React.useState<Mode | null>(null);
  const [step, setStep] = React.useState(1);
  const [thresholds, setThresholds] = React.useState<Record<Mode, number>>({ Fluorescein: 0.5, white: 0.5 });
  const [prediction, setPrediction] = React.useState<PredictPayload | null>(null);
  const [uploadedFileName, setUploadedFileName] = React.useState('');
  const [editOverlay, setEditOverlay] = React.useState('');
  const [error, setError] = React.useState('');
  const [message, setMessage] = React.useState('');
  const [activeTab, setActiveTab] = React.useState<'Workflow' | 'Guide'>('Workflow');
  const [brushMode, setBrushMode] = React.useState<BrushMode>('Add');
  const [brush, setBrush] = React.useState(18);
  const [alpha, setAlpha] = React.useState(0.35);
  const [calMethod, setCalMethod] = React.useState<CalMethod>('Line');
  const [knownMm, setKnownMm] = React.useState(1.0);
  const [gridMm, setGridMm] = React.useState(0.1);
  const [gridPx, setGridPx] = React.useState(25);
  const [mmPerPx, setMmPerPx] = React.useState<number | null>(null);
  const [liveMmPerPx, setLiveMmPerPx] = React.useState<number | null>(null);
  const [lineForCalibration, setLineForCalibration] = React.useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
  const [results, setResults] = React.useState<ResultPayload | null>(null);
  const [notes, setNotes] = React.useState('');
  const [useRef, setUseRef] = React.useState(false);
  const [targetGrey, setTargetGrey] = React.useState(120);
  const [roi, setRoi] = React.useState<number[] | null>(null);
  const [calBase, setCalBase] = React.useState<{ base: string; grid: string } | null>(null);
  const editCanvas = React.useRef<Canvas | null>(null);
  const lineCanvas = React.useRef<Canvas | null>(null);
  const rectCanvas = React.useRef<Canvas | null>(null);

  React.useEffect(() => {
    jsonFetch('/api/session').then((data) => setSessionId(data.session_id)).catch((e) => setError(e.message));
  }, []);

  React.useEffect(() => {
    if (!sessionId) return;
    jsonFetch('/api/session', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, case_id: caseId, visit_date: visitDate, session_eye: eye }),
    }).catch((e) => setError(e.message));
  }, [sessionId, caseId, visitDate, eye]);

  React.useEffect(() => {
    if (!sessionId || !mode || step < 4) return;
    fetch(`${API}/api/calibration/base?session_id=${sessionId}&ns=${mode}&spacing_px=${gridPx}&grid_opacity=0.16`)
      .then((r) => r.json())
      .then(setCalBase)
      .catch(() => undefined);
  }, [sessionId, mode, step, gridPx, prediction?.overlay]);

  React.useEffect(() => {
    if (calMethod === 'Grid (research only)') {
      setLiveMmPerPx(gridPx > 0 ? Number(gridMm) / Number(gridPx) : null);
      return;
    }
    setLiveMmPerPx(mmPerPxFromLine(lineForCalibration, knownMm));
  }, [calMethod, gridMm, gridPx, lineForCalibration, knownMm]);

  React.useEffect(() => {
    if (!sessionId || !mode || !prediction) return;
    fetch(`${API}/api/overlay?session_id=${sessionId}&ns=${mode}&alpha=${alpha}`)
      .then((r) => r.json())
      .then((data) => setEditOverlay(data.overlay || prediction.overlay))
      .catch(() => setEditOverlay(prediction.overlay));
  }, [sessionId, mode, prediction?.edited_mask, prediction?.overlay, alpha]);

  const ns = mode || 'Fluorescein';
  const label = labelForMode(mode);

  async function openMode() {
    try {
      setError('');
      const selected: Mode = modeChoice === 'White-light' ? 'white' : 'Fluorescein';
      const data = await jsonFetch('/api/open', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, mode: selected }),
      });
      setSessionId(data.session_id || sessionId);
      setMode(selected);
      setStep(data.step || 1);
      setResults(null);
      setPrediction(null);
      setUploadedFileName('');
      setLiveMmPerPx(null);
      setLineForCalibration(null);
    } catch (e: any) {
      setError(`Could not open workflow. Check that the backend is running on ${API}. ${e.message || e}`);
    }
  }

  async function predict(file?: File, thresholdOverride?: number) {
    if (!mode) return;
    setError('');
    setMessage('Running prediction...');
    try {
      const form = new FormData();
      form.set('session_id', sessionId);
      form.set('ns', mode);
      form.set('threshold', String(thresholdOverride ?? thresholds[mode]));
      if (file) {
        form.set('image', file);
        setUploadedFileName(file.name);
      }
      const res = await fetch(`${API}/api/predict`, { method: 'POST', body: form });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const data = await res.json();
      setPrediction(data);
      setEditOverlay(data.overlay);
      setStep(data.step);
      setMmPerPx(data.mm_per_px || null);
      setLiveMmPerPx(null);
      setLineForCalibration(null);
      setResults(null);
    } finally {
      setMessage('');
    }
  }

  async function applyBrushChanges() {
    if (!mode || !editCanvas.current) return;
    const data = await jsonFetch('/api/mask/apply', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        ns: mode,
        mode: brushMode,
        canvas_rgba: getOnlyObjectsDataUrl(editCanvas.current),
      }),
    });
    clearObjects(editCanvas.current);
    setPrediction((prev) => (prev ? { ...prev, ...data } : prev));
  }

  async function resetMask() {
    if (!mode) return;
    const data = await jsonFetch('/api/mask/reset', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, ns: mode }),
    });
    if (editCanvas.current) clearObjects(editCanvas.current);
    setPrediction((prev) => (prev ? { ...prev, ...data } : prev));
  }

  async function confirmMask() {
    if (!mode) return;
    const data = await jsonFetch('/api/mask/confirm', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, ns: mode }),
    });
    setStep(data.step);
  }

  async function confirmCalibration() {
    if (!mode) return;
    let body: any = { session_id: sessionId, ns: mode, method: calMethod };
    if (calMethod === 'Line') {
      const line = lineCanvas.current ? latestLine(lineCanvas.current) : lineForCalibration;
      body = { ...body, known_mm: knownMm, line };
    } else {
      body = { ...body, grid_mm: gridMm, spacing_px: gridPx };
    }
    const data = await jsonFetch('/api/calibration', { method: 'POST', body: JSON.stringify(body) });
    setStep(data.step);
    setMmPerPx(data.mm_per_px);
    setLiveMmPerPx(data.mm_per_px);
    setResults(null);
    if (data.step >= 5) {
      await loadResults();
    }
  }

  async function loadResults() {
    if (!mode) return;
    const rect = rectCanvas.current ? latestRect(rectCanvas.current) : roi;
    if (rect && rect[2] > 0 && rect[3] > 0) setRoi(rect);
    const data = await jsonFetch('/api/results', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        ns: mode,
        acquisition_notes: notes,
        grey_reference: { use_ref: useRef, roi: rect, target_grey: targetGrey },
      }),
    });
    setResults(data);
  }

  React.useEffect(() => {
    if (step >= 5 && mode) loadResults().catch((e) => setError(e.message));
  }, [step, mode, notes, useRef, targetGrey, mmPerPx]);

  async function saveVisit() {
    if (!mode || !results) return;
    await jsonFetch('/api/save', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        ns: mode,
        acquisition_notes: notes,
        grey_reference: { use_ref: useRef, roi, target_grey: targetGrey },
        report_text: results.report_text,
      }),
    });
    setMessage('Saved to database.');
    await loadResults();
  }

  async function deleteCurrentCase() {
    try {
      await jsonFetch('/api/delete-case', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, case_id: caseId, visit_date: visitDate, session_eye: eye }),
      });
      setMessage(`Case ${caseId} deleted.`);
      setResults(null);
    } catch (e: any) {
      setError(e.message);
    }
  }

  function downloadReport() {
    if (!mode) return;
    fetch(`${API}/api/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        ns: mode,
        acquisition_notes: notes,
        grey_reference: { use_ref: useRef, roi, target_grey: targetGrey },
      }),
    })
      .then(async (res) => {
        const blob = await res.blob();
        const cd = res.headers.get('content-disposition') || '';
        const filename = cd.match(/filename=([^;]+)/)?.[1] || 'report.pdf';
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
      })
      .catch((e) => setError(e.message));
  }

  const chartRows = React.useMemo(() => {
    return (results?.timeline || []).map((r) => ({ ...r, visit_date: r.visit_date?.slice(0, 10) }));
  }, [results]);

  return (
    <div className="shell">
      <aside className="sidebar">
        <h3>Session</h3>
        <label>Case ID<input value={caseId} onChange={(e) => setCaseId(e.target.value)} /></label>
        <label>Visit date (YYYY-MM-DD)<input value={visitDate} onChange={(e) => setVisitDate(e.target.value)} /></label>
        <label>Eye<select value={eye} onChange={(e) => setEye(e.target.value as 'Right' | 'Left')}><option>Right</option><option>Left</option></select></label>
        <hr />
        <button className="secondary" onClick={() => deleteCurrentCase().catch((e) => setError(e.message))}>Delete this case</button>
        <hr />
        <fieldset>
          <legend>Image type</legend>
          <label className="radio"><input type="radio" checked={modeChoice === 'Fluorescein'} onChange={() => setModeChoice('Fluorescein')} /> Fluorescein</label>
          <label className="radio"><input type="radio" checked={modeChoice === 'White-light'} onChange={() => setModeChoice('White-light')} /> White-light</label>
        </fieldset>
        <button onClick={openMode}>Open</button>
      </aside>

      <main className="main">
        <div className="warning">This is a research prototype only. Not for clinical use. Case IDs must be anonymised. No patient identifiable information should be entered.</div>
        <div className="appbar">
          <div>
            <div className="brand">Corneal Ulcer Assessment</div>
            <div className="subtitle">Segmentation • Calibration • Measurements</div>
          </div>
        </div>
        <div className="tabs">
          <button className={activeTab === 'Workflow' ? 'tab on' : 'tab'} onClick={() => setActiveTab('Workflow')}>Workflow</button>
          <button className={activeTab === 'Guide' ? 'tab on' : 'tab'} onClick={() => setActiveTab('Guide')}>Guide</button>
        </div>

        {error && <div className="error">{error}</div>}
        {message && <div className="success">{message}</div>}

        {activeTab === 'Guide' && (
          <section className="guide">
            <h3>Practical notes</h3>
            <p>-1- Try to keep focus sharp and minimise glare.</p>
            <p>-2- If you want mm² and mm to mean anything you need a real calibration reference (line over known distance is best).</p>
            <h3>Grey reference (white light)</h3>
            <p>Use the grey patch when you're comparing <b>opacity proxy</b> across visits.</p>
            <p>It helps reduce day to day lighting differences but it won't fix heavy glare or poor focus.</p>
            <h3>Data and privacy</h3>
            <p>- Case IDs must be anonymised, do not enter patient names or NHS numbers</p>
            <p>- This tool is a research prototype and is not approved for clinical use</p>
          </section>
        )}

        {activeTab === 'Workflow' && !mode && <div className="info">Choose an image type in the sidebar to begin.</div>}

        {activeTab === 'Workflow' && mode && (
          <div className="workflow">
            <div className="left">
              <h2>{label}</h2>
              <h3>/1/ Upload</h3>
              <label className="uploadLabel">{label} image
                <span className="uploadBox">
                  <span className="uploadText">
                    <span className="uploadIcon">+</span>
                    <span>
                      <strong>Drag and drop file here</strong>
                      <small>{uploadedFileName || 'Limit 200MB per file • PNG, JPG, JPEG'}</small>
                    </span>
                  </span>
                  <span className="uploadButton">Choose file</span>
                  <input className="fileInput" type="file" accept={mode === 'white' ? '.png,.jpg,.jpeg,.tif,.tiff' : '.png,.jpg,.jpeg'} onChange={(e) => e.target.files?.[0] && predict(e.target.files[0]).catch((er) => setError(er.message))} />
                </span>
              </label>
              <label className="rangeLabel">Threshold<input type="range" min="0.05" max="0.95" step="0.05" value={thresholds[mode]} onChange={(e) => setThresholds({ ...thresholds, [mode]: Number(e.target.value) })} onPointerUp={(e) => predict(undefined, Number(e.currentTarget.value)).catch((er) => setError(er.message))} /></label>
              <div className="rangeValue">{thresholds[mode].toFixed(2)}</div>
              {!prediction && <div className="softError">No {label} image uploaded yet.</div>}

              {prediction && (
                <>
                  <hr />
                  <h3>/2/ Adjust mask</h3>
                  <p className="caption">If anything is missed or over-segmented, adjust the mask.</p>
                  <div className="controls three">
                    <fieldset><legend>Brush</legend><label className="radio"><input type="radio" checked={brushMode === 'Add'} onChange={() => setBrushMode('Add')} /> Add</label><label className="radio"><input type="radio" checked={brushMode === 'Erase'} onChange={() => setBrushMode('Erase')} /> Erase</label></fieldset>
                    <label>Size<input type="range" min="3" max="70" step="1" value={brush} onChange={(e) => setBrush(Number(e.target.value))} /></label>
                    <label>Overlay<input type="range" min="0.05" max="0.85" step="0.05" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} /></label>
                  </div>
                  <FabricCanvas background={editOverlay || prediction.overlay} mode={brushMode} brush={brush} tool="freedraw" onReady={(c) => (editCanvas.current = c)} />
                  <div className="buttons two">
                    <button onClick={applyBrushChanges}>Apply changes</button>
                    <button onClick={resetMask}>Reset to prediction</button>
                  </div>

                  <hr />
                  <h3>/3/ Confirm</h3>
                  <img className="displayImage" src={prediction.overlay} />
                  <button onClick={() => confirmMask().catch((e) => setError(e.message))}>Use this mask</button>

                  <hr />
                  <h3>/4/ Calibrate</h3>
                  {step < 4 ? <div className="info">Confirm the mask to continue.</div> : (
                    <>
                      <p className="caption">Calibration sets the physical scale (mm/px). Use a real in frame reference if you can.</p>
                      <fieldset><legend>Method</legend><label className="radio"><input type="radio" checked={calMethod === 'Line'} onChange={() => setCalMethod('Line')} /> Line</label><label className="radio"><input type="radio" checked={calMethod === 'Grid (research only)'} onChange={() => setCalMethod('Grid (research only)')} /> Grid (research only)</label></fieldset>
                      {calMethod === 'Line' ? (
                        <>
                          <p className="caption">Draw a line across a known distance (eg: a ruler/marker in the image).</p>
                          <label>Known distance (mm)<input type="number" min="0.01" step="0.1" value={knownMm} onChange={(e) => setKnownMm(Number(e.target.value))} /></label>
                          {calBase && (
                            <FabricCanvas
                              background={calBase.grid}
                              mode="Add"
                              brush={3}
                              tool="line"
                              onReady={(c) => (lineCanvas.current = c)}
                              onObjectsChange={(c) => setLineForCalibration(latestLine(c))}
                            />
                          )}
                        </>
                      ) : (
                        <>
                          <div className="warning small">Use grid only if the mm value is genuinely known (not guessed).</div>
                          <label>Grid square size (mm)<select value={gridMm} onChange={(e) => setGridMm(Number(e.target.value))}>{[0.05, 0.1, 0.2, 0.5, 1.0].map((v) => <option key={v}>{v}</option>)}</select></label>
                          <label>Grid spacing (px)<input type="range" min="5" max="200" step="1" value={gridPx} onChange={(e) => setGridPx(Number(e.target.value))} /></label>
                          {calBase && <img className="displayImage" src={calBase.grid} />}
                        </>
                      )}
                      <Metric label="Calibration (mm/px)" value={(liveMmPerPx ?? (step >= 5 ? mmPerPx : null)) == null ? '—' : (liveMmPerPx ?? mmPerPx)!.toFixed(6)} />
                      <button onClick={() => confirmCalibration().catch((e) => setError(e.message))}>Confirm calibration</button>
                    </>
                  )}

                  <hr />
                  <h3>/5/ Results</h3>
                  {step < 5 ? <div className="info">Confirm calibration to continue.</div> : (
                    <Results
                      mode={mode}
                      results={results}
                      notes={notes}
                      setNotes={setNotes}
                      useRef={useRef}
                      setUseRef={setUseRef}
                      targetGrey={targetGrey}
                      setTargetGrey={setTargetGrey}
                      prediction={prediction}
                      rectCanvas={rectCanvas}
                      roi={roi}
                      setRoi={setRoi}
                      loadResults={loadResults}
                      saveVisit={saveVisit}
                      downloadReport={downloadReport}
                      saveEnabled={Boolean(caseId.trim())}
                      chartRows={chartRows}
                    />
                  )}
                </>
              )}
            </div>
            <div className="right">
              <div className="sticky">
                <div className="panel">
                  <div className="panelTitle">{label}</div>
                  <Stepper step={step} />
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function Results(props: {
  mode: Mode;
  results: ResultPayload | null;
  notes: string;
  setNotes: (v: string) => void;
  useRef: boolean;
  setUseRef: (v: boolean) => void;
  targetGrey: number;
  setTargetGrey: (v: number) => void;
  prediction: PredictPayload;
  rectCanvas: React.MutableRefObject<Canvas | null>;
  roi: number[] | null;
  setRoi: (v: number[] | null) => void;
  loadResults: () => Promise<void>;
  saveVisit: () => Promise<void>;
  downloadReport: () => void;
  saveEnabled: boolean;
  chartRows: TimelineRow[];
}) {
  const r = props.results?.result;
  return (
    <>
      {props.mode === 'white' && (
        <details>
          <summary>Grey reference (white light only)</summary>
          <label className="check"><input type="checkbox" checked={props.useRef} onChange={(e) => props.setUseRef(e.target.checked)} /> Use a grey reference (if visible)</label>
          <label>Target grey level<input type="range" min="80" max="170" step="1" value={props.targetGrey} onChange={(e) => props.setTargetGrey(Number(e.target.value))} /></label>
          {props.useRef && (
            <>
              <p><b>Why this helps:</b> white light slit lamp images can vary in brightness/exposure.<br />If you mark a neutral grey patch, the app can normalise brightness so the <b>opacity proxy</b> is more comparable across visits.</p>
              <p className="caption">Draw a rectangle around the grey patch.</p>
              <FabricCanvas background={props.prediction.rgb} mode="Add" brush={2} tool="rect" onReady={(c) => (props.rectCanvas.current = c)} />
              {props.roi ? <p className="caption">Grey patch ROI: x={props.roi[0]}, y={props.roi[1]}, w={props.roi[2]}, h={props.roi[3]}</p> : <div className="info">No grey patch selected yet.</div>}
              <button onClick={() => props.setRoi(null)}>Clear grey patch selection</button>
              <button onClick={props.loadResults}>Apply grey reference</button>
            </>
          )}
        </details>
      )}
      {r && (
        <>
          <div className="metrics four">
            <Metric label="Area (mm²)" value={r.area_mm2 == null ? 'N/A' : r.area_mm2.toFixed(4)} />
            <Metric label="Eq diameter (mm)" value={r.eq_diameter_mm == null ? 'N/A' : r.eq_diameter_mm.toFixed(2)} />
            <Metric label="Zone" value={String(r.zone || '—')} />
            <Metric label="Blur" value={(r.blur_laplacian_var || 0).toFixed(2)} />
          </div>
          <div className="metrics three">
            <Metric label="Vertical" value={String(r.vertical_sector || '—')} />
            <Metric label="Horizontal" value={String(r.horizontal_sector || '—')} />
            <Metric label="Eye" value={String(r.eye || 'Right')} />
          </div>
          {props.mode === 'white' && (
            <div className="metrics three">
              <Metric label="Opacity mean" value={r.opacity_mean == null ? '—' : r.opacity_mean.toFixed(3)} />
              <Metric label="Opacity contrast" value={r.opacity_contrast == null ? '—' : r.opacity_contrast.toFixed(3)} />
              <Metric label="Opacity z-score" value={r.opacity_zscore == null ? '—' : r.opacity_zscore.toFixed(2)} />
            </div>
          )}
          {props.results?.flags?.length ? <div className="info">Quality notes: {props.results.flags.join(', ')}</div> : null}
        </>
      )}

      <hr />
      <h3>Report</h3>
      <label>Notes (optional)<textarea value={props.notes} onChange={(e) => props.setNotes(e.target.value)} rows={4} /></label>
      <pre className="report">{props.results?.report_text || ''}</pre>
      <button onClick={props.downloadReport}>Download report</button>

      <hr />
      <h3>Timeline</h3>
      {!props.saveEnabled && <div className="info">Add a Case ID to save this visit to the timeline.</div>}
      <button disabled={!props.saveEnabled} onClick={() => props.saveVisit().catch((e) => console.error(e))}>Save this visit</button>
      {props.results?.timeline?.length ? (
        <>
          <table>
            <thead><tr>{['visit_date', 'eye', 'mode', 'area_mm2', 'eq_diameter_mm', 'opacity_zscore', 'mm_per_px', 'qc_flags', 'created_at'].map((h) => <th key={h}>{h}</th>)}</tr></thead>
            <tbody>{props.results.timeline.map((row, i) => <tr key={i}>{['visit_date', 'eye', 'mode', 'area_mm2', 'eq_diameter_mm', 'opacity_zscore', 'mm_per_px', 'qc_flags', 'created_at'].map((h) => <td key={h}>{String((row as any)[h] ?? '')}</td>)}</tr>)}</tbody>
          </table>
          {props.chartRows.length >= 2 && (
            <div className="trend">
              <div>
                <b>How to read the charts</b>
                <p>- <b>X-axis:</b> visit date</p>
                <p>- <b>Y-axis:</b> metric value</p>
                <p>- Lines connect visits in chronological order.</p>
                <p>- <b>Area/diameter</b> reflect size change.</p>
                <p>- <b>Opacity z-score</b> is a <b>proxy</b> (white light only)/sensitive to lighting and focus.</p>
              </div>
              <div className="charts">
                <b>Size metrics</b>
                <SimpleLineChart
                  rows={props.chartRows}
                  series={[
                    { key: 'area_mm2', label: 'area_mm2', color: '#ff4d4d' },
                    { key: 'eq_diameter_mm', label: 'eq_diameter_mm', color: '#00a36c' },
                  ]}
                  height={250}
                />
                {props.chartRows.filter((r) => r.opacity_zscore != null).length >= 2 && (
                  <>
                    <b>Opacity proxy (white-light)</b>
                    <SimpleLineChart
                      rows={props.chartRows}
                      series={[{ key: 'opacity_zscore', label: 'opacity_zscore', color: '#7c3aed' }]}
                      height={220}
                    />
                  </>
                )}
              </div>
            </div>
          )}
        </>
      ) : null}
    </>
  );
}

window.addEventListener('error', (event) => {
  const root = document.getElementById('root');
  if (root && !root.innerHTML.trim()) {
    root.innerHTML = `<div class="bootError"><h2>Something stopped the app from loading.</h2><p>${event.message}</p></div>`;
  }
});

window.addEventListener('unhandledrejection', (event) => {
  const root = document.getElementById('root');
  if (root && !root.innerHTML.trim()) {
    root.innerHTML = `<div class="bootError"><h2>Something stopped the app from loading.</h2><p>${String(event.reason)}</p></div>`;
  }
});

const root = document.getElementById('root');
if (!root) {
  throw new Error('Missing root element');
}

createRoot(root).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>,
);
