import streamlit as st
import pandas as pd
import plotly.express as px
import os
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(page_title="Steam Games Dashboard", layout="wide")
st.title("🎮 Steam Games Analytics & Chatbot")

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    return pd.read_csv("data/a_steam_data_2021_2025.csv")

df = load_data()

# =====================
# LOAD GEMINI AI
# =====================
load_dotenv(find_dotenv())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

USE_AI = False
model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Correct model name
        model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
        USE_AI = True
        st.sidebar.success("✅ Gemini AI Connected")
    except Exception as e:
        st.sidebar.warning("⚠ Gemini API error. AI disabled.")
        st.sidebar.text(str(e))
        USE_AI = False
else:
    st.sidebar.warning("⚠ GEMINI_API_KEY not found in .env. AI disabled.")

# =====================
# SIDEBAR NAVIGATION
# =====================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard", "Dataset", "Chatbot"])

# =====================
# DASHBOARD PAGE
# =====================
if page == "Dashboard":
    st.header("📊 Steam Games Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("🎮 Total Games", len(df))
    col2.metric("💲 Average Price", f"${df['price'].mean():.2f}")
    col3.metric("🏆 Top Genre", df['genres'].value_counts().idxmax())

    games_per_year = df.groupby("release_year").size().reset_index(name="count")
    fig = px.bar(
        games_per_year,
        x="release_year",
        y="count",
        title="Games Released per Year"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================
# DATASET PAGE
# =====================
elif page == "Dataset":
    st.header("📋 Steam Dataset")
    st.dataframe(df.head(50), use_container_width=True)

# =====================
# CHATBOT PAGE
# =====================
elif page == "Chatbot":
    st.header("🤖 Steam AI Chatbot")

    # Sidebar options
    with st.sidebar:
        show_data = st.checkbox("📊 Show Dataset", False)
        use_ai = st.checkbox("🤖 Enable AI Answer", USE_AI)

    # =====================
    # RULE-BASED CHATBOT
    # =====================
    def simple_chatbot(q):
        q = q.lower()
        if "total games" in q:
            return f"🎮 Total Games: {len(df)}"
        if "average price" in q:
            return f"💲 Average Price: ${df['price'].mean():.2f}"
        if "top genre" in q:
            return f"🏆 Top Genre: {df['genres'].value_counts().idxmax()}"
        if "top developer" in q:
            return f"🏢 Top Developer: {df['developer'].value_counts().idxmax()}"
        if "games per year" in q:
            return df.groupby("release_year").size().to_frame("games")
        if "summary" in q or "insight" in q:
            return (
                f"📊 Steam Dataset Summary:\n"
                f"- Total Games: {len(df)}\n"
                f"- Average Price: ${df['price'].mean():.2f}\n"
                f"- Top Genre: {df['genres'].value_counts().idxmax()}\n"
                f"- Top Developer: {df['developer'].value_counts().idxmax()}"
            )
        return "❌ Try: total games, average price, top genre, top developer"

    # =====================
    # GEMINI AI CHATBOT
    # =====================
    def ai_chatbot(question):
        if not USE_AI:
            return "⚠ AI is not enabled or API key is invalid."
        prompt = f"""
You are a data analyst.
Answer ONLY using the Steam dataset below.

DATA SAMPLE:
{df.head(40).to_string()}

QUESTION:
{question}

Give a clear and concise answer.
"""
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Gemini API error: {str(e)}"

    # =====================
    # UI INPUT
    # =====================
    query = st.text_input(
        "Ask something about the Steam games data",
        placeholder="e.g. What is the average price of games?"
    )

    if query:
        st.markdown("### 🔍 Result")
        result = simple_chatbot(query)
        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
        else:
            st.success(result)

        if use_ai and USE_AI:
            st.markdown("### 🤖 AI Explanation (Gemini)")
            st.info(ai_chatbot(query))

    if show_data:
        st.markdown("### 📊 Dataset Preview")
        st.dataframe(df.head(50), use_container_width=True)
