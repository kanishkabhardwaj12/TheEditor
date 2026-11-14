import warnings

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL.*",
)

import streamlit as st

from src.ui.research_tab import render_research_tab
from src.ui.writing_tab import render_writing_tab


def main() -> None:
    st.set_page_config(page_title="The Editorial", layout="wide")
    st.title("The Editorial")
    st.caption(
        "Research assistant + writing assistant for newsletter-style editorial workflows."
    )

    research_tab, writing_tab = st.tabs(["Research Assistant", "Writing Assistant"])

    with research_tab:
        render_research_tab()

    with writing_tab:
        render_writing_tab()


if __name__ == "__main__":
    main()
