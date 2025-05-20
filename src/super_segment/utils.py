from pathlib import Path
import streamlit as st
from functools import cache


@cache
def display_markdown_file(
    filepath: str,
    remove_title: str = None,
    encoding: str = "utf-8",
    warn_if_missing: bool = True,
):
    path = Path(filepath)
    if path.exists():
        contents = path.read_text(encoding=encoding)
        if remove_title:
            contents = contents.replace(remove_title, "")
        st.markdown(contents)
    elif warn_if_missing:
        st.warning(f"{path.name} file not found in the project directory: {path}")
