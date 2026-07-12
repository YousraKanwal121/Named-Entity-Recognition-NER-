import streamlit as st
import spacy
import pandas as pd
import plotly.express as px
from collections import Counter

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Named Entity Recognition",
    page_icon="🧠",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return spacy.load("ner_model")

nlp = load_model()

# =========================
# CUSTOM CSS (MODERN UI)
# =========================
st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
    background: radial-gradient(circle at top, #0b1437 0%, #050816 50%, #020617 100%);
    color:white;
}

section[data-testid="stSidebar"]{
    background: linear-gradient(
        180deg,
        #050816 0%,
        #060b1f 50%,
        #030712 100%
    ) !important;

    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Main Title */
.title{
    text-align:center;
    font-size:52px;
    font-weight:800;
    margin-top:10px;
    background: linear-gradient(90deg,#60a5fa,#a78bfa,#22d3ee);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle{
    text-align:center;
    color:#b8c1ec;
    font-size:18px;
    margin-bottom:25px;
}

/* Input Box */
/* Input Box (DARK IMPROVED) */
.stTextArea textarea{
    background: rgba(10, 15, 35, 0.85)!important;
    color:white!important;
    border-radius:16px!important;
    border:1px solid rgba(255,255,255,0.18)!important;
    font-size:16px!important;
    padding:12px;
    box-shadow: 0px 0px 20px rgba(99,102,241,0.15);
}

.stTextArea textarea:focus{
    border:1px solid #7c3aed!important;
    outline:none!important;
}

/* Button */
.stButton button{
    background: linear-gradient(90deg,#2563eb,#7c3aed)!important;
    color:white!important;
    font-weight:600!important;
    height:50px;
    width:220px;
    border-radius:12px!important;
    border:none!important;
}

/* Glass Card */
.card{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius:18px;
    padding:18px;
    backdrop-filter: blur(10px);
}

/* Entity Pills */
.pill{
    display:inline-block;
    padding:10px 15px;
    margin:5px;
    border-radius:999px;
    background: linear-gradient(90deg,#0ea5a4,#6366f1);
    color:white;
    font-weight:600;
    font-size:14px;
}

/* Stat Cards */
.stat{
    background: rgba(255,255,255,0.05);
    border-radius:16px;
    padding:15px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.08);
}

/* Badge */
.badge{
    background:#22c55e;
    padding:5px 10px;
    border-radius:999px;
    font-size:12px;
    float:right;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR (LIKE IMAGE)
# =========================
with st.sidebar:
    st.markdown(
    "<h1 style='font-size:34px; font-weight:900; color:white;'>NER Dashboard</h1>",
    unsafe_allow_html=True
)
    st.markdown("### Supported Entities")
    st.success("👤 PERSON")
    st.info("🏢 ORG")
    st.warning("🌍 GPE")

    st.markdown("---")
    st.markdown("### Model Status")
    st.success("Loaded Successfully")

# =========================
# HEADER
# =========================
st.markdown('<div class="title">Named Entity Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect PERSON, ORG, GPE, DATE and more using AI</div>', unsafe_allow_html=True)

# =========================
# INPUT
# =========================
text = st.text_area("", placeholder="Type or paste your sentence here...")

# =========================
# BUTTON (CENTER FIX)
st.markdown("""
<style>
div.stButton {
    display: flex;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

run = st.button("🔍 Predict Entities")
# =========================
# PREDICTION
# =========================
if run:

    if text.strip():

        doc = nlp(text)
        ents = doc.ents

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("Detected Entities")
        st.markdown(f"<span class='badge'>{len(ents)} Found</span>", unsafe_allow_html=True)

        if ents:

            # ENTITY PILL UI
            for ent in ents:
                st.markdown(f"<span class='pill'>{ent.text} → {ent.label_}</span>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # COUNTS
            labels = [e.label_ for e in ents]
            counts = Counter(labels)

            col1, col2, col3 = st.columns(3)

            col1.markdown(f"<div class='stat'><h2>{len(ents)}</h2><p>Total Entities</p></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='stat'><h2>{len(counts)}</h2><p>Entity Types</p></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='stat'><h2>{max(counts.values())}</h2><p>Max Type Count</p></div>", unsafe_allow_html=True)

            # CHART
            df = pd.DataFrame({
                "Entity": list(counts.keys()),
                "Count": list(counts.values())
            })

            fig = px.pie(df, names="Entity", values="Count", hole=0.65)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=450
            )

            st.plotly_chart(fig, use_container_width=True)

            # TABLE
            st.subheader("Entity Breakdown")
            st.dataframe(df, use_container_width=True)

        else:
            st.warning("No entities found.")

    else:
        st.error("Please enter text first.")