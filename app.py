"""Streamlit frontend for automated solar layout with interactive client-side polygon editing."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
from hashlib import sha1
from typing import Any

import httpx
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components


st.set_page_config(page_title="Solar Layout Engine", page_icon="☀️", layout="wide")


def init_state() -> None:
    defaults: dict[str, Any] = {
        "backend_url": "http://localhost:8000",
        "upload_id": None,
        "image_data_uri": None,
        "image_width": 0,
        "image_height": 0,
        "roof_polygon": [],
        "polygon_sync_payload": "",
        "last_polygon_sync_payload": "",
        "analysis_result": None,
        "analysis_error": None,
        "estimate_result": None,
        "estimate_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


async def analyze_roof_async(base_url: str, image_bytes: bytes, meters_per_pixel: float) -> dict[str, Any]:
    """Call FastAPI /analyze-roof asynchronously using the Qwen-VL backend."""
    files = {"image": ("roof.png", image_bytes, "image/png")}
    data = {"backend": "qwen_vl", "meters_per_pixel": str(meters_per_pixel)}
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(f"{base_url.rstrip('/')}/analyze-roof", files=files, data=data)
        response.raise_for_status()
        return response.json()


def run_async(coro: Any) -> Any:
    """Safely run an async coroutine from Streamlit's sync execution context."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def normalize_polygon(points: Any, width: int, height: int) -> list[list[float]]:
    """Validate and clamp polygon points to image bounds."""
    normalized: list[list[float]] = []
    if not isinstance(points, list):
        return normalized

    for point in points:
        if not (isinstance(point, list) and len(point) == 2):
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            continue
        x = min(max(x, 0.0), float(width))
        y = min(max(y, 0.0), float(height))
        normalized.append([round(x, 2), round(y, 2)])
    return normalized


def parse_backend_polygon(response_json: dict[str, Any]) -> list[list[float]]:
    raw = response_json.get("data", {}).get("roof_polygon", [])
    return normalize_polygon([[p.get("x"), p.get("y")] for p in raw if isinstance(p, dict)], st.session_state["image_width"], st.session_state["image_height"])


def make_canvas_html(image_data_uri: str, polygon: list[list[float]], image_width: int, image_height: int) -> str:
    """Create a pure client-side polygon editor that syncs only on mouse release/lock."""
    max_display_width = 1000
    display_width = min(max_display_width, max(400, image_width))
    scale = display_width / image_width if image_width else 1.0
    display_height = max(250, int(image_height * scale))

    payload = json.dumps({
        "image": image_data_uri,
        "polygon": polygon,
        "imageWidth": image_width,
        "imageHeight": image_height,
        "displayWidth": display_width,
        "displayHeight": display_height,
    })

    return f"""
<div style=\"font-family: sans-serif;\">
  <canvas id=\"roofCanvas\" width=\"{display_width}\" height=\"{display_height}\" style=\"border:1px solid #334155; border-radius:8px; cursor: crosshair; max-width:100%;\"></canvas>
  <div style=\"display:flex; gap:8px; margin-top:8px; align-items:center;\">
    <button id=\"lockBtn\" style=\"background:#2563eb;color:white;border:0;padding:6px 10px;border-radius:6px;cursor:pointer;\">Lock Layout</button>
    <small style=\"color:#475569\">Drag vertices • Double-click edge to add point • Right-click vertex to remove</small>
  </div>
</div>
<script>
(() => {{
  const config = {payload};
  const canvas = document.getElementById('roofCanvas');
  const ctx = canvas.getContext('2d');
  const lockBtn = document.getElementById('lockBtn');
  const image = new Image();

  const imageWidth = config.imageWidth;
  const imageHeight = config.imageHeight;
  const scaleX = canvas.width / imageWidth;
  const scaleY = canvas.height / imageHeight;

  let points = (config.polygon || []).map(p => ({{ x: p[0], y: p[1] }}));
  let dragIndex = -1;
  let dragging = false;

  function toCanvasPoint(p) {{
    return {{ x: p.x * scaleX, y: p.y * scaleY }};
  }}

  function fromCanvasPoint(p) {{
    return {{ x: p.x / scaleX, y: p.y / scaleY }};
  }}

  function findVertexIndex(x, y, radius = 10) {{
    for (let i = 0; i < points.length; i++) {{
      const p = toCanvasPoint(points[i]);
      const d = Math.hypot(p.x - x, p.y - y);
      if (d <= radius) return i;
    }}
    return -1;
  }}

  function distanceToSegment(px, py, x1, y1, x2, y2) {{
    const dx = x2 - x1;
    const dy = y2 - y1;
    if (dx === 0 && dy === 0) return Math.hypot(px - x1, py - y1);
    const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
    const nx = x1 + t * dx;
    const ny = y1 + t * dy;
    return Math.hypot(px - nx, py - ny);
  }}

  function findInsertionEdge(x, y) {{
    if (points.length < 2) return -1;
    let bestIdx = -1;
    let bestDist = Number.POSITIVE_INFINITY;
    for (let i = 0; i < points.length; i++) {{
      const a = toCanvasPoint(points[i]);
      const b = toCanvasPoint(points[(i + 1) % points.length]);
      const d = distanceToSegment(x, y, a.x, a.y, b.x, b.y);
      if (d < bestDist) {{
        bestDist = d;
        bestIdx = i;
      }}
    }}
    return bestDist <= 16 ? bestIdx : -1;
  }}

  function sendPolygon() {{
    const value = JSON.stringify(points.map(p => [Number(p.x.toFixed(2)), Number(p.y.toFixed(2))]));
    const parentDoc = window.parent.document;
    const bridge = parentDoc.querySelector('textarea[aria-label="Polygon Sync Payload"]');
    if (!bridge || bridge.value === value) return;

    const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    valueSetter.call(bridge, value);
    bridge.dispatchEvent(new Event('input', {{ bubbles: true }}));
  }}

  function draw() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (image.complete) ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    if (points.length > 0) {{
      ctx.beginPath();
      points.forEach((p, idx) => {{
        const cp = toCanvasPoint(p);
        if (idx === 0) ctx.moveTo(cp.x, cp.y);
        else ctx.lineTo(cp.x, cp.y);
      }});
      if (points.length >= 3) ctx.closePath();
      ctx.fillStyle = 'rgba(37, 99, 235, 0.2)';
      if (points.length >= 3) ctx.fill();
      ctx.strokeStyle = '#1d4ed8';
      ctx.lineWidth = 2;
      ctx.stroke();
    }}

    points.forEach((p, idx) => {{
      const cp = toCanvasPoint(p);
      ctx.beginPath();
      ctx.arc(cp.x, cp.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#0ea5e9';
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = 'white';
      ctx.font = '11px sans-serif';
      ctx.fillText(String(idx + 1), cp.x + 8, cp.y - 8);
    }});
  }}

  canvas.addEventListener('mousedown', (event) => {{
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    dragIndex = findVertexIndex(x, y);
    dragging = dragIndex !== -1;
  }});

  canvas.addEventListener('mousemove', (event) => {{
    if (!dragging || dragIndex === -1) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const img = fromCanvasPoint({{ x, y }});
    points[dragIndex] = {{
      x: Math.max(0, Math.min(imageWidth, img.x)),
      y: Math.max(0, Math.min(imageHeight, img.y)),
    }};
    draw();
  }});

  canvas.addEventListener('mouseup', () => {{
    if (dragging) sendPolygon();
    dragging = false;
    dragIndex = -1;
  }});

  canvas.addEventListener('mouseleave', () => {{
    if (dragging) sendPolygon();
    dragging = false;
    dragIndex = -1;
  }});

  canvas.addEventListener('dblclick', (event) => {{
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (points.length < 2) return;
    const edgeIndex = findInsertionEdge(x, y);
    if (edgeIndex === -1) return;

    const img = fromCanvasPoint({{ x, y }});
    points.splice(edgeIndex + 1, 0, {{
      x: Math.max(0, Math.min(imageWidth, img.x)),
      y: Math.max(0, Math.min(imageHeight, img.y)),
    }});
    draw();
    sendPolygon();
  }});

  canvas.addEventListener('contextmenu', (event) => {{
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const idx = findVertexIndex(x, y);
    if (idx !== -1 && points.length > 3) {{
      points.splice(idx, 1);
      draw();
      sendPolygon();
    }}
  }});

  lockBtn.addEventListener('click', () => sendPolygon());

  image.onload = draw;
  image.src = config.image;
}})();
</script>
"""


def estimate_panels(base_url: str, meters_per_pixel: float, annual_yield_factor: float, placement_backend: str, panel_width_m: float, panel_height_m: float, row_spacing_m: float, col_spacing_m: float, panel_power_kw: float) -> dict[str, Any]:
    payload = {
        "image_width": st.session_state["image_width"],
        "image_height": st.session_state["image_height"],
        "roof_polygon": [
            {"x": p[0], "y": p[1]} for p in st.session_state["roof_polygon"]
        ],
        "obstacles": [],
        "meters_per_pixel": meters_per_pixel,
        "panel_config": {
            "panel_width_m": panel_width_m,
            "panel_height_m": panel_height_m,
            "row_spacing_m": row_spacing_m,
            "col_spacing_m": col_spacing_m,
            "panel_power_kw": panel_power_kw,
        },
        "annual_yield_factor_kwh_per_kw": annual_yield_factor,
        "placement_backend": placement_backend,
    }
    response = httpx.post(f"{base_url.rstrip('/')}/estimate-panels", json=payload, timeout=180.0)
    response.raise_for_status()
    return response.json()


st.title("☀️ Automated Solar Layout Engine")
st.caption("Qwen-VL roof polygon detection + client-side polygon correction + inclination-corrected scale")

with st.sidebar:
    st.header("Configuration")
    backend_url = st.text_input("Backend URL", value=st.session_state["backend_url"])
    st.session_state["backend_url"] = backend_url

    planar_mpp = st.number_input(
        "Planar Meters per Pixel",
        min_value=0.001,
        max_value=10.0,
        value=0.1,
        step=0.001,
        format="%.3f",
    )

    inclination_deg = st.number_input(
        "Roof Inclination Angle (Degrees)",
        min_value=0,
        max_value=60,
        value=30,
        step=1,
    )

    cosine = max(math.cos(math.radians(float(inclination_deg))), 1e-9)
    corrected_mpp = planar_mpp / math.sqrt(cosine)
    st.info(f"Inclination-corrected scale: **{corrected_mpp:.4f} m/px**")

    st.subheader("Panel Inputs")
    panel_width_m = st.number_input("Panel Width (m)", min_value=0.1, max_value=5.0, value=1.134, step=0.001)
    panel_height_m = st.number_input("Panel Height (m)", min_value=0.1, max_value=5.0, value=2.278, step=0.001)
    panel_power_kw = st.number_input("Panel Power (kW)", min_value=0.05, max_value=2.0, value=0.62, step=0.01)
    row_spacing_m = st.number_input("Row Spacing (m)", min_value=0.0, max_value=2.0, value=0.1, step=0.01)
    col_spacing_m = st.number_input("Column Spacing (m)", min_value=0.0, max_value=2.0, value=0.1, step=0.01)
    annual_yield_factor = st.number_input(
        "Annual Yield Factor (kWh/kW)", min_value=100.0, max_value=3000.0, value=1450.0, step=10.0
    )
    placement_backend = st.selectbox("Placement Backend", ["raster", "opencv"], index=0)

uploaded = st.file_uploader("Upload rooftop drone/satellite image", type=["png", "jpg", "jpeg", "webp"])

if uploaded is not None:
    image_bytes = uploaded.getvalue()
    upload_id = sha1(image_bytes).hexdigest()

    if upload_id != st.session_state["upload_id"]:
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB")
            st.session_state["image_width"], st.session_state["image_height"] = rgb.size

        mime = uploaded.type or "image/png"
        st.session_state["image_data_uri"] = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
        st.session_state["upload_id"] = upload_id
        st.session_state["roof_polygon"] = []
        st.session_state["analysis_result"] = None
        st.session_state["analysis_error"] = None
        st.session_state["estimate_result"] = None
        st.session_state["estimate_error"] = None

        with st.spinner("Analyzing roof polygon with Qwen-VL backend..."):
            try:
                response_json = run_async(analyze_roof_async(st.session_state["backend_url"], image_bytes, planar_mpp))
                st.session_state["analysis_result"] = response_json
                st.session_state["roof_polygon"] = parse_backend_polygon(response_json)
                st.session_state["polygon_sync_payload"] = json.dumps(st.session_state["roof_polygon"])
                st.session_state["last_polygon_sync_payload"] = st.session_state["polygon_sync_payload"]
            except Exception as exc:  # noqa: BLE001
                st.session_state["analysis_error"] = str(exc)

        st.rerun()

if st.session_state["image_data_uri"] is None:
    st.info("Upload an image to begin. The app will auto-call `/analyze-roof` with `backend=qwen_vl`.")
else:
    if st.session_state["analysis_error"]:
        st.error(f"Roof analysis failed: {st.session_state['analysis_error']}")

    st.markdown(
        """
        <style>
            div[data-testid="stTextArea"]:has(textarea[aria-label="Polygon Sync Payload"]) {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    bridge_payload = st.text_area("Polygon Sync Payload", key="polygon_sync_payload", height=1)
    if bridge_payload and bridge_payload != st.session_state["last_polygon_sync_payload"]:
        try:
            parsed = json.loads(bridge_payload)
            polygon = normalize_polygon(parsed, st.session_state["image_width"], st.session_state["image_height"])
            if len(polygon) >= 3:
                st.session_state["roof_polygon"] = polygon
                st.session_state["last_polygon_sync_payload"] = bridge_payload
                st.session_state["estimate_result"] = None
                st.session_state["estimate_error"] = None
        except json.JSONDecodeError:
            pass

    st.write(f"Detected/edited polygon vertices: **{len(st.session_state['roof_polygon'])}**")
    components.html(
        make_canvas_html(
            st.session_state["image_data_uri"],
            st.session_state["roof_polygon"],
            st.session_state["image_width"],
            st.session_state["image_height"],
        ),
        height=min(760, int(max(300, st.session_state["image_height"] * min(1000, max(400, st.session_state["image_width"])) / st.session_state["image_width"])) + 70),
        scrolling=False,
    )

    if st.button("Lock Layout & Estimate Panels", type="primary", disabled=len(st.session_state["roof_polygon"]) < 3):
        with st.spinner("Estimating solar panel layout..."):
            try:
                st.session_state["estimate_result"] = estimate_panels(
                    st.session_state["backend_url"],
                    corrected_mpp,
                    annual_yield_factor,
                    placement_backend,
                    panel_width_m,
                    panel_height_m,
                    row_spacing_m,
                    col_spacing_m,
                    panel_power_kw,
                )
                st.session_state["estimate_error"] = None
            except Exception as exc:  # noqa: BLE001
                st.session_state["estimate_error"] = str(exc)

    if st.session_state["estimate_error"]:
        st.error(f"Panel estimation failed: {st.session_state['estimate_error']}")

    if st.session_state["estimate_result"]:
        data = st.session_state["estimate_result"].get("data", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estimated Panels", data.get("estimated_panel_count", 0))
        c2.metric("Estimated Power (kW)", data.get("estimated_power_kw", 0.0))
        c3.metric("Annual Energy (kWh)", data.get("estimated_annual_energy_kwh", 0.0))
        c4.metric("Used Roof Area (m²)", data.get("used_area_m2", 0.0))

        st.download_button(
            "Download Estimate JSON",
            data=json.dumps(st.session_state["estimate_result"], indent=2),
            file_name="estimate_panels_result.json",
            mime="application/json",
        )
