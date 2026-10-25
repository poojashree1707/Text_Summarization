
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Text Summarization AI",
    page_icon="AI",
    layout="centered"
)

# Keep the dark purple and blue theme
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101025, #191936, #20204a);
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

textarea {
    background-color: #242442 !important;
    color: white !important;
    border: 1px solid #7777ff !important;
    border-radius: 10px !important;
}

.stButton > button {
    background: linear-gradient(90deg, #7657ff, #4389ff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 25px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #6040e8, #286be0);
    color: white;
}

.summary-box {
    background-color: #242442;
    border-left: 4px solid #7777ff;
    border-radius: 10px;
    padding: 18px;
    color: white;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# Load the summarization model
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

st.title("Text Summarization AI")
st.write("Paste your text below and get a short, clear summary.")

text = st.text_area(
    "Enter your text",
    placeholder="Paste your paragraph or article here...",
    height=220
)

if st.button("Summarize"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            with st.spinner("Generating your summary..."):
                summarizer = load_model()

                result = summarizer(
                    text,
                    max_length=100,
                    min_length=10,
                    do_sample=False,
                    truncation=True
                )

                summary = result[0]["summary_text"]

            st.subheader("Your Summary")
            st.markdown(
                f'<div class="summary-box">{summary}</div>',
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")
