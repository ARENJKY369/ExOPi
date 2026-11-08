"""Streamlit viewer for a shapehunter candidates.json file."""

from __future__ import annotations

import json


def main() -> None:
    """Run the optional Streamlit interface; imports Streamlit only here."""
    import streamlit as st

    st.title("Shapehunter candidate viewer")
    st.caption("Candidates are not confirmed planets. Missing vetting data remains unknown.")
    upload = st.file_uploader("Upload candidates.json", type="json")
    if upload is None:
        st.info("Choose a candidates.json output file to inspect.")
        return
    try:
        candidates = json.load(upload)
        if not isinstance(candidates, list) or not all(isinstance(c, dict) for c in candidates):
            raise ValueError("Expected a JSON list of candidate objects")
    except (ValueError, UnicodeDecodeError) as exc:
        st.error(f"Invalid candidate file: {exc}")
        return
    st.write(f"Candidates: {len(candidates)}")
    for index, candidate in enumerate(candidates, 1):
        with st.expander(f"Candidate {index}: {candidate.get('family', 'unknown')} — {candidate.get('status', 'unknown')}"):
            st.json(candidate)


if __name__ == "__main__":
    main()
