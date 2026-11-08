"""Interactive, privacy-focused Streamlit experience for Shapehunter."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from shapehunter.core import LightCurve
from shapehunter.data.synthetic import generate_lightcurve, simulate_planet_transit
from shapehunter.pipeline.run_star import search_star

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _password_gate() -> bool:
    import streamlit as st

    configured = os.getenv("SHAPEHUNTER_PASSWORD", "")
    if not configured:
        return True
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        password = st.text_input("Private access password", type="password")
        if st.button("Enter Shapehunter", use_container_width=True):
            given = hashlib.sha256(password.encode()).digest()
            expected = hashlib.sha256(configured.encode()).digest()
            st.session_state.authenticated = hmac.compare_digest(given, expected)
            if not st.session_state.authenticated:
                st.error("Password not recognized.")
        if not st.session_state.authenticated:
            st.stop()
    return True


def _load_npz(upload: Any) -> LightCurve:
    raw = upload.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload must be smaller than 20 MB")
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        if "arr_time" in data and "arr_flux" in data:
            time = data["arr_time"]
            flux = data["arr_flux"]
            flux_err = data["arr_flux_err"] if "arr_flux_err" in data else None
            centroid_x = data["arr_centroid_x"] if "arr_centroid_x" in data else None
            centroid_y = data["arr_centroid_y"] if "arr_centroid_y" in data else None
            meta = {}
            if "meta_json" in data:
                meta = json.loads(str(data["meta_json"]))
        else:
            time = data["time"] if "time" in data else None
            flux = data["flux"] if "flux" in data else None
            flux_err = data["flux_err"] if "flux_err" in data else None
            centroid_x = data["centroid_x"] if "centroid_x" in data else None
            centroid_y = data["centroid_y"] if "centroid_y" in data else None
            meta = {"source": "user-upload"}
    if time is None or flux is None:
        raise ValueError("NPZ must contain time and flux arrays")
    return LightCurve(np.asarray(time, float), np.asarray(flux, float), None if flux_err is None else np.asarray(flux_err, float), centroid_x=None if centroid_x is None else np.asarray(centroid_x, float), centroid_y=None if centroid_y is None else np.asarray(centroid_y, float), meta=meta)


def _search(seed: int, injected: bool, period: float, radius: float, family: str, max_candidates: int, threshold: float):
    curve = generate_lightcurve(seed=seed)
    if injected:
        curve = simulate_planet_transit(curve, period=period, rp=radius, family=family)
    return curve, search_star(curve, min_power=threshold, max_candidates=max_candidates)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Shapehunter", page_icon="S", layout="wide")
    st.markdown("<style>.block-container{max-width:1180px;padding-top:1.5rem}.stMetric{border:1px solid #e5e7eb;border-radius:8px;padding:.75rem;background:#fff}.stAlert{border-radius:8px}</style>", unsafe_allow_html=True)
    _password_gate()

    st.title("Shapehunter")
    st.caption("Explore a light curve. Every result is a screening finding, not a confirmation.")

    with st.sidebar:
        st.header("Analysis")
        source = st.radio("Data source", ["Synthetic demo", "Upload NPZ", "Open findings JSON"])
        st.divider()
        seed = st.number_input("Seed", 0, 100000, 7, 1)
        injected = st.toggle("Inject a transit", value=True)
        period = st.slider("Period (days)", 1.0, 10.0, 3.5, 0.1, disabled=not injected)
        radius = st.slider("Planet radius / star radius", 0.03, 0.25, 0.10, 0.01, disabled=not injected)
        family = st.selectbox("Injected shape", ["sphere", "oblate", "ringed", "binary_planet", "comet_tail", "dust_cloud"], disabled=not injected)
        threshold = st.slider("Screening threshold", 5.0, 20.0, 8.0, 0.5)
        max_candidates = st.slider("Candidates to search", 1, 5, 1)
        st.caption("Your uploads stay in memory for this session and are not written to disk by this app.")

    curve: LightCurve | None = None
    candidates = []
    candidate_payload: list[dict[str, Any]] = []

    if source == "Synthetic demo":
        curve, candidates = _search(int(seed), bool(injected), float(period), float(radius), family, int(max_candidates), float(threshold))
    elif source == "Upload NPZ":
        upload = st.file_uploader("Choose a Shapehunter .npz light curve", type=["npz"])
        if upload:
            try:
                curve = _load_npz(upload)
                candidates = search_star(curve, min_power=float(threshold), max_candidates=int(max_candidates))
            except Exception as exc:
                st.error(f"Could not analyze this file: {exc}")
    else:
        upload = st.file_uploader("Choose candidates.json", type=["json"])
        if upload:
            try:
                candidate_payload = json.loads(upload.getvalue().decode("utf-8"))
                if not isinstance(candidate_payload, list) or not all(isinstance(item, dict) for item in candidate_payload):
                    raise ValueError("Expected a JSON list of candidate objects")
                candidates = []
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                st.error(f"Invalid candidate file: {exc}")

    if source == "Open findings JSON" and candidate_payload:
        candidates = []
        payload = candidate_payload
    elif curve is not None:
        payload = [candidate.to_dict() for candidate in candidates]
    else:
        payload = []

    tab_explore, tab_findings, tab_about = st.tabs(["Light curve", "Findings", "How to interpret"])
    with tab_explore:
        if curve is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Samples", f"{len(curve):,}")
            c2.metric("Baseline", f"{curve.baseline:.1f} d")
            c3.metric("Cadence", f"{curve.cadence:.4f} d")
            c4.metric("Findings", str(len(candidates)))
            st.line_chart({"relative flux": curve.flux}, height=300)
        elif candidate_payload:
            st.info(f"Loaded {len(candidate_payload)} saved finding(s).")
        else:
            st.info("Choose a source to begin. No data leaves this page.")

    with tab_findings:
        if not payload:
            st.success("No screening findings met the current threshold.")
        for index, finding in enumerate(payload, 1):
            label = finding.get("family", "unknown")
            status = finding.get("status", "candidate")
            period_value = finding.get("period")
            title = f"Finding {index}  |  {label.replace('_', ' ').title()}  |  {status}"
            with st.expander(title, expanded=index == 1):
                cols = st.columns(4)
                cols[0].metric("Period", "n/a" if period_value is None else f"{float(period_value):.4f} d")
                cols[1].metric("Depth", f"{float(finding.get('depth', 0) or 0):.4%}")
                cols[2].metric("SNR", f"{float(finding.get('snr', 0) or 0):.2f}")
                cols[3].metric("Transits", f"{float(finding.get('n_transits', 0) or 0):.0f}")
                st.json(finding, expanded=False)
        if payload:
            st.download_button("Download findings JSON", json.dumps(payload, indent=2), file_name="shapehunter-findings.json", mime="application/json", use_container_width=True)

    with tab_about:
        st.subheader("What this evidence means")
        st.markdown("- A found signal is a candidate for follow-up, not a confirmed planet.\n- Screening thresholds are empirical and should not be read as calibrated false-alarm probabilities.\n- Shape matches can be ambiguous, especially for near-degenerate projections.\n- Vetting may show **unknown** when catalogue, centroid, or systematics data is unavailable.\n- Synthetic recovery does not establish real-survey performance.")
        st.warning("Do not use Shapehunter for mission-critical decisions, medical conclusions, or claims of discovery without independent analysis and peer review.")


if __name__ == "__main__":
    main()
