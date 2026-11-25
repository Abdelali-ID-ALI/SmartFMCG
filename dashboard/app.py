"""SmartFMCG Streamlit entry point. Run from the repository root."""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "2")

import streamlit as st

from dashboard import operations, science

PAGES = {
    "Overview": operations.overview,
    "Scenario configuration": operations.scenario_page,
    "Route planning": operations.route_page,
    "Dispatch replay": operations.simulation_page,
    "Travel-time prediction": science.traffic_page,
    "Learning policies": science.learning_page,
    "Scientific benchmark": science.benchmark_page,
}


def main():
    st.set_page_config(page_title="SmartFMCG", page_icon="📍", layout="wide", initial_sidebar_state="auto")
    st.html("""
    <style>
    [data-testid="stMainBlockContainer"] { max-width: 1420px; padding-top: 2.6rem; }
    [data-testid="stSidebar"] { border-right: 1px solid #d9e4ec; }
    [data-testid="stSidebar"] h1 { font-size: 1.7rem; padding-bottom: .05rem; letter-spacing: -.045em; }
    [data-testid="stMetric"] { background: #ffffff; border-color: #d9e4ec; border-radius: .75rem; }
    [data-testid="stMetricValue"] { font-size: clamp(1.15rem, 2vw, 1.85rem); }
    h1 { letter-spacing: -.04em; }
    </style>
    """)
    with st.sidebar:
        st.title("SmartFMCG")
        st.caption("Predict · Optimize · Adapt")
        st.divider()
        page = st.radio("Workspace", PAGES, label_visibility="collapsed")
        st.divider()
        case = operations.get_inputs()
        st.caption(f"Casablanca · {case.customers} customers · Seed {case.seed}")
        st.caption("Synthetic orders and traffic")
    try:
        PAGES[page]()
    except FileNotFoundError as exc:
        name = os.path.basename(exc.filename or str(exc))
        st.error(f"Required artifact is missing ({name}). Restore it from the project checkpoint and reload.")


if __name__ == "__main__":
    main()
