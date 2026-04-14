#llm_report.py
from __future__ import annotations

import os
import json
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import streamlit as st
except Exception:
    class _Secrets(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    class _StreamlitShim:
        secrets = _Secrets()

        def error(self, *args, **kwargs):
            return None

        def code(self, *args, **kwargs):
            return None

        def write(self, *args, **kwargs):
            return None

        class _NoopExpander:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def expander(self, *args, **kwargs):
            return self._NoopExpander()

    st = _StreamlitShim()

try:
    from openai import OpenAI
    OPENAI_SDK_OK = True
except Exception as import_e:
    OPENAI_SDK_OK = False
    _OPENAI_IMPORT_ERROR = import_e

#Uses Groq’s OpenAI compatible API to turn a summary dict into a UK clinical style documentation note. If anything goes wrong (no SDK, no key, API error, JSON parse error) it falls back to a safe template report.
@dataclass
class LLMReportResult:
    report_text:str
    used_model:str
    used_llm:bool

def clinician_report_fallback(summary:Dict[str,Any]) -> str:
    flags= (summary.get("qc_flags") or "").strip()
    qc_line= (
        "No major quality warnings flagged."
        if not flags
        else f"Quality notes: {flags}. Please interpret measurements with caution."
    )

    lines= [
        "Corneal ulcer assessment (assistive)",
        f"Case: {summary.get('case_id','—')} • Date: {summary.get('visit_date','—')} • Eye: {summary.get('eye','—')} • Modality: {summary.get('mode','—')}",
        "",
        "Findings",
        f"- Area: {summary.get('area_mm2','—')} mm²",
        f"- Equivalent diameter: {summary.get('eq_diameter_mm','—')} mm",
        f"- Location: {summary.get('zone','—')} zone ({summary.get('vertical_sector','—')}, {summary.get('horizontal_sector','—')})",
    ]
    if summary.get("opacity_zscore") is not None:
        lines.append(f"- Opacity proxy (z-score, white-light): {summary.get('opacity_zscore','—')}")

    lines+=[
        "",
        "Quality/limitations",
        f"- {qc_line}",
        f"- Blur score: {summary.get('blur','—')}",
        f"- Calibration: {summary.get('mm_per_px','—')} mm/px",
        "",
        "Disclaimer",
        "- Assistive output for documentation support; not a diagnosis and does not recommend treatment.",
    ]
    return "\n".join(lines)


def mask_key(k: Optional[str]) -> str:
    if not k:
        return "None"
    k = str(k)
    if len(k) <= 10:
        return "***"
    return f"{k[:6]}…{k[-4:]}"


def get_groq_key() -> Optional[str]:
    try:
        key= st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        key= None
    if not key:
        key= os.environ.get("GROQ_API_KEY")
    return key

#LLMs sometimes return extra text or code fences even when you ask them not to
def strip_json_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip()
        # Remove leading ``` / ```json
        t = t.split("```", 1)[-1].strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
        # Remove trailing ```
        if "```" in t:
            t = t.split("```", 1)[0].strip()
    return t

def _extract_json_object(text: str) -> str:
    #If the model includes extra text, extract the first {...} JSON object.
    t = strip_json_fences(text).strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]
    return t

#Build a fallback report first (always available)
def generate_report_with_llm(
    summary: Dict[str,Any],
    *,
    acquisition_notes: str = "",
    debug:bool= False,  #default off for a clean UI
) -> LLMReportResult:
    fallback = clinician_report_fallback(summary)
    MODEL = "llama-3.3-70b-versatile"

    # if not OPENAI_SDK_OK:
    #     if debug:
    #         st.error(f"OpenAI SDK import failed: {type(_OPENAI_IMPORT_ERROR).__name__}: {_OPENAI_IMPORT_ERROR}")
    #     return LLMReportResult(report_text=fallback, used_model="(template)", used_llm=False)
    api_key = get_groq_key()
    if not api_key:
        if debug:
            st.error("No GROQ_API_KEY found in st.secrets or environment.")
        return LLMReportResult(report_text=fallback, used_model="(template)", used_llm=False)

    if debug:
        with st.expander("LLM Debug", expanded=False):
            st.write("Provider:", "Groq")
            st.write("Model:", MODEL)
            st.write("Has GROQ_API_KEY in st.secrets:", ("GROQ_API_KEY" in getattr(st, "secrets", {})))
            st.write("Has GROQ_API_KEY in env:", bool(os.environ.get("GROQ_API_KEY")))
            st.write("Key masked:", mask_key(api_key))

    schema= {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "context": {"type": "string"},
            "findings": {"type": "array", "items": {"type": "string"}},
            "interpretation": {"type": "array", "items": {"type": "string"}},
            "quality_limitations": {"type": "array", "items": {"type": "string"}},
            "audit": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "context", "findings", "interpretation", "quality_limitations", "audit"],
    }

#This prompt I used chtgpt to help to generate it 
    system= (
        "You are drafting a UK clinical-style documentation note for an assistive imaging tool.\n"
        "Hard rules:\n"
        "- Use ONLY the values present in the provided JSON.\n"
        "- Do NOT add any new measurements, thresholds, dates, or clinical facts.\n"
        "- Do NOT infer a diagnosis and do NOT recommend treatment.\n"
        "- If a value is missing/null, explicitly state it is unavailable.\n"
        "- Mention quality flags and whether calibration is present.\n"
        "- If mm_per_px is present, state calibration is present.\n"
        "- Return ONLY a single JSON object matching the provided schema.\n"
        "- No markdown, no prose, no code fences.\n"
    )

    payload= {
        "summary": summary,
        "acquisition_notes": acquisition_notes.strip()[:400],
    }

    user_prompt= (
        "Return ONLY valid JSON.\n"
        "Do not include any extra keys or text.\n"
        f"Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"INPUT (JSON):\n{json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        #Make Groq call using OpenAI-compatible client
        #It;s not calling OpenAI's servers here base_url points to Groq. It just uses the OpenAI SDK as a client library.
        client= OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            # Force JSON output
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
        text = _extract_json_object(text)
        if debug:
            with st.expander("Raw JSON (debug)", expanded=False):
                st.code(text[:2000])

        data= json.loads(text)

        # Render final report
        lines:list[str] = []
        lines.append(data["title"])
        lines.append(data["context"])
        lines.append(
            f"Case: {summary.get('case_id','—')} • Date: {summary.get('visit_date','—')} • "
            f"Eye: {summary.get('eye','—')} • Modality: {summary.get('mode','—')}"
        )
        lines.append("")

        if data.get("findings"):
            lines.append("Findings")
            lines += [f"- {x}" for x in data["findings"]]
            lines.append("")

        if data.get("interpretation"):
            lines.append("Interpretation (assistive)")
            lines += [f"- {x}" for x in data["interpretation"]]
            lines.append("")

        if data.get("quality_limitations"):
            lines.append("Quality / limitations")
            lines += [f"- {x}" for x in data["quality_limitations"]]
            lines.append("")

        if data.get("audit"):
            lines.append("Audit")
            lines += [f"- {x}" for x in data["audit"]]

        return LLMReportResult(report_text="\n".join(lines).strip(), used_model=MODEL, used_llm=True)

    except Exception as e:
        if debug:
            st.error(f"Groq call failed: {type(e).__name__}: {e}")
            st.code(traceback.format_exc())
        return LLMReportResult(report_text=fallback, used_model="(template)", used_llm=False)
