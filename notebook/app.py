import streamlit as st
from classify_page import render_classify
from regression_page import render_regression

st.set_page_config(page_title="World Happiness ML Suite ", page_icon="🌍", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "home"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def go(page):
    
    st.session_state.page = page


# Small "Home" button visible once inside a sub-app
if st.session_state.page != "home":
    st.sidebar.button("🏠 Back to Home", on_click=go, args=("home",), use_container_width=True)
    st.sidebar.markdown("---")

if st.session_state.page == "home":
    st.markdown(
        "<h1 style='text-align:center;'>🌍 World Happiness ML Suite </h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;font-size:18px;'>Choose what you want to do with the "
        "World Happiness Report data.</p>",
        unsafe_allow_html=True,
    )
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 😊 HappyClassify")
        st.write("Classify a country into a happiness **level** "
                 "(Unhappy / Moderately Happy / Happy / Very Happy).")
        st.button("Go to Classification →", on_click=go, args=("classify",),
                   use_container_width=True, type="primary")
    with col2:
        st.markdown("### 📈 HappyPredict")
        st.write("Predict the continuous happiness **score** (Life Ladder) for a country.")
        st.button("Go to Regression →", on_click=go, args=("regress",),
                   use_container_width=True, type="primary")

elif st.session_state.page == "classify":
    render_classify()

elif st.session_state.page == "regress":
    render_regression()
