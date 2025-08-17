import os
import base64
import requests
import tempfile
import streamlit as st
from groq import Groq
#from elevenlabs.client import ElevenLabs
#from elevenlabs import VoiceSettings
import assemblyai as aai

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq

# --- Initialize Session State for API Keys ---
if "GROQ_API_KEY" not in st.session_state:
    st.session_state.GROQ_API_KEY = None
if "ELEVENLABS_API_KEY" not in st.session_state:
    st.session_state.ELEVENLABS_API_KEY = None
if "ASSEMBLYAI_API_KEY" not in st.session_state:
    st.session_state.ASSEMBLYAI_API_KEY = None
if "RAPIDAPI_KEY" not in st.session_state:
    st.session_state.RAPIDAPI_KEY = None

# Create temp directory if not exists
os.makedirs("temp", exist_ok=True)

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Sidebar: API Key Inputs ---
st.sidebar.header("🔐 Enter Your API Keys")

# Groq API Key
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=st.session_state.GROQ_API_KEY or "",
    #help="Get your Groq API key from https://console.groq.com/keys"
)
if groq_api_key:
    st.session_state.GROQ_API_KEY = groq_api_key

# ElevenLabs API Key
#elevenlabs_api_key = st.sidebar.text_input(
#    "ElevenLabs API Key",
#    type="password",
#    value=st.session_state.ELEVENLABS_API_KEY or "",
#    #help="Get your ElevenLabs API key from https://elevenlabs.io/"
#)
#if elevenlabs_api_key:
#    st.session_state.ELEVENLABS_API_KEY = elevenlabs_api_key

# AssemblyAI API Key
assemblyai_api_key = st.sidebar.text_input(
    "AssemblyAI API Key",
    type="password",
    value=st.session_state.ASSEMBLYAI_API_KEY or "",
    #help="Get your AssemblyAI API key from https://www.assemblyai.com/"
)
if assemblyai_api_key:
    st.session_state.ASSEMBLYAI_API_KEY = assemblyai_api_key

# RapidAPI Key
rapidapi_key = st.sidebar.text_input(
    "RapidAPI Key",
    type="password",
    value=st.session_state.RAPIDAPI_KEY or "",
    #help="Get your RapidAPI key from https://rapidapi.com/"
)
if rapidapi_key:
    st.session_state.RAPIDAPI_KEY = rapidapi_key

# --- Validate Keys & Initialize Clients ---
def get_clients():
    if not st.session_state.GROQ_API_KEY:
        st.warning("⚠️ Please enter your Groq API key.")
        return None, None, None
    #if not st.session_state.ELEVENLABS_API_KEY:
    #    st.warning("⚠️ Please enter your ElevenLabs API key.")
    #    return None, None, None
    if not st.session_state.ASSEMBLYAI_API_KEY:
        st.warning("⚠️ Please enter your AssemblyAI API key.")
        return None, None, None
    if not st.session_state.RAPIDAPI_KEY:
        st.warning("⚠️ Please enter your RapidAPI key.")
        return None, None, None

    try:
        # Initialize clients
        client = Groq(api_key=st.session_state.GROQ_API_KEY)
        #eleven_client = ElevenLabs(api_key=st.session_state.ELEVENLABS_API_KEY)
        aai.settings.api_key = st.session_state.ASSEMBLYAI_API_KEY
        RAPIDAPI_KEY = st.session_state.RAPIDAPI_KEY

        return client, RAPIDAPI_KEY
    except Exception as e:
        st.error(f"❌ Failed to initialize API clients: {e}")
        return None, None

# Get API clients
client, RAPIDAPI_KEY = get_clients()

# Exit early if any key is missing
if not all([client, RAPIDAPI_KEY]):
    st.markdown("### 🚀 Stock Analysis Assistant")
    st.markdown("Welcome! Please enter your API keys in the sidebar to get started.")
    st.stop()

# --- Clear Keys Button ---
st.sidebar.divider()
if st.sidebar.button("Clear All API Keys"):
    st.session_state.GROQ_API_KEY = None
    #st.session_state.ELEVENLABS_API_KEY = None
    st.session_state.ASSEMBLYAI_API_KEY = None
    st.session_state.RAPIDAPI_KEY = None
    st.rerun()

# ==== INTENTS ====
FILTERABLE_INTENTS = [
    "tickerId", "industry", "companyProfile", "currentPrice",
    "stockTechnicalData", "percentChange", "yearHigh", "yearLow", "financials",
    "keyMetrics", "futureExpiryDates", "futureOverviewData", "initialStockFinancialData",
    "analystView", "recosBar", "riskMeter", "shareholding", "stockCorporateActionData",
    "stockDetailsReusableData", "stockFinancialData", "recentNews",
    "board_meetings", "dividends", "splits", "bonus", "rights"
]

NON_FILTERABLE_INTENTS = [
    "quarter_results", "yoy_results", "balancesheet", "cashflow", "ratios",
    "shareholding_pattern_quarterly", "shareholding_pattern_yearly"
]

# ==== FUNCTION TO FILTER JSON DATA ==========
def filter_json_data(raw_data, intent):
    if intent in FILTERABLE_INTENTS:
        filtered_data = raw_data.get(intent, None)
        if filtered_data is None:
            return {"error": f"No data found for intent: {intent}"}
        return filtered_data
    elif intent in NON_FILTERABLE_INTENTS:
        return raw_data
    else:
        return {"error": "Unknown intent type"}

# ==== FUNCTION TO FETCH COMPANY NAME ==========
def get_company_name(user_query, last_company):
    prompt = f"""
    You are a finance industry expert assistant specialized in extracting company names from queries.
    From the following query, extract and return ONLY the company symbol or name.
    Do not include any extra text or explanation.

    Query: "{user_query}"

    If no new company is mentioned, return the last known company name: "{last_company}"

    Output:
    """
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
        temperature=0
    )
    stock_name = response.choices[0].message.content.strip()
    return stock_name

# ==== FUNCTION TO CLASSIFY INTENT ==========
def get_classify_intent(user_query, last_intent):
    prompt = f"""
    You are a finance industry expert assistant classifying user intent.
    Classify the intent as one of:
    - tickerId, industry, companyProfile, currentPrice, stockTechnicalData
    - percentChange, yearHigh, yearLow, financials, keyMetrics
    - futureExpiryDates, futureOverviewData, initialStockFinancialData
    - analystView, recosBar, riskMeter, shareholding
    - stockCorporateActionData, stockDetailsReusableData, stockFinancialData
    - recentNews, quarter_results, yoy_results, balancesheet, cashflow
    - ratios, shareholding_pattern_quarterly, shareholding_pattern_yearly
    - board_meetings, dividends, splits, bonus, rights

    If unclear, assume it's similar to the previous intent: "{last_intent}"

    From the query, return ONLY the intent name. No explanation.
    Query: "{user_query}"
    Output:
    """
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
        temperature=0
    )
    intent = response.choices[0].message.content.strip()
    return intent

# === FUNCTION TO FETCH COMPANY DETAILS ===
def get_company_details(stock_name):
    url = "https://indian-stock-exchange-api2.p.rapidapi.com/stock"
    querystring = {"name": stock_name}
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "indian-stock-exchange-api2.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}, {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# === FUNCTION TO FETCH HISTORICAL STATS ===
def get_hist_company_stats_details(stock_name, intent):
    url = "https://indian-stock-exchange-api2.p.rapidapi.com/historical_stats"
    querystring = {"stock_name": stock_name, "stats": intent}
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "indian-stock-exchange-api2.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}, {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# === FUNCTION TO FETCH CORPORATE ACTIONS ===
def get_corporate_actions(stock_name):
    url = "https://indian-stock-exchange-api2.p.rapidapi.com/corporate_actions"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "indian-stock-exchange-api2.p.rapidapi.com"
    }
    params = {"stock_name": stock_name}
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}, {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# === DETERMINE WHICH API TO CALL ===
def get_api_to_call(intent):
    if intent in ["tickerId", "industry", "companyProfile", "currentPrice",
                  "stockTechnicalData", "percentChange", "yearHigh", "yearLow",
                  "financials", "keyMetrics", "futureExpiryDates", "futureOverviewData",
                  "initialStockFinancialData", "analystView", "recosBar", "riskMeter",
                  "shareholding", "stockCorporateActionData", "stockDetailsReusableData",
                  "stockFinancialData", "recentNews"]:
        return "get_company_details"
    elif intent in ["board_meetings", "dividends", "splits", "bonus", "rights"]:
        return "get_corporate_actions"
    elif intent in ["quarter_results", "yoy_results", "balancesheet", "cashflow", "ratios",
                    "shareholding_pattern_quarterly", "shareholding_pattern_yearly"]:
        return "get_hist_company_stats_details"
    else:
        return None

# === CALL API ===
def call_api(api_name, stock_name, intent):
    if api_name == "get_company_details":
        return get_company_details(stock_name)
    elif api_name == "get_hist_company_stats_details":
        return get_hist_company_stats_details(stock_name, intent)
    elif api_name == "get_corporate_actions":
        return get_corporate_actions(stock_name)
    else:
        return {"error": "Invalid API selected"}

# === SUMMARIZE WITH GROQ ===
def summarize_with_groq(data, query, user_query):
    prompt = f"""
    You are a finance expert assistant.
    The user asked: "{user_query}"
    Specifically about: "{query}"
    
    Here is the data:
    {data}

    Explain it clearly and simply. If no relevant data, say so. Do not mention JSON.
    """
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3
    )
    return response.choices[0].message.content

#---------------Voice Groq Play AI TTS--------------------
# === Helper function for TTS ===
def play_audio_response_groq(text: str):
    try:
        #speech_file = Path("response.wav")
        response = client.audio.speech.create(
            model="playai-tts",
            voice="Aaliyah-PlayAI",
            input=text,  # the generated response text
            response_format="wav"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio_path = tmp_audio.name
        response.write_to_file(tmp_audio_path)  # Groq writes full file

        with open(tmp_audio_path, "rb") as f:
            audio_bytes = f.read()
            b64_audio = base64.b64encode(audio_bytes).decode()

        st.subheader("🔊 Audio Response:")
        st.markdown(
            f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
            </audio>
            """,
            unsafe_allow_html=True,
        )
        st.audio(tmp_audio_path, format="audio/wav")

    except Exception as tts_error:
        st.warning(f"Error generating audio: {tts_error}")

# ------------- SESSION STATE -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_company" not in st.session_state:
    st.session_state.last_company = None
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Streamlit UI ---
st.set_page_config(page_title="AI Stock Assistant", page_icon="📈", layout="wide")
st.title("AI Assistant - Stock Market")
#st.markdown("*AI-Powered Stock Market Analysis with Document Q&A*")

# --- Sidebar: Upload PDF ---
with st.sidebar:
    st.header("📂 Upload your PDF")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file and st.button("📌 Process Document"):
        with st.spinner("Reading and indexing PDF..."):
            pdf_reader = PdfReader(uploaded_file)
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text()

            splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len
            )
            chunks = splitter.split_text(full_text)

            @st.cache_resource
            def create_vectorstore(chunks):
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                return FAISS.from_texts(chunks, embedding=embeddings)

            st.session_state.vectorstore = create_vectorstore(chunks)
            st.success("✅ Document processed!")

# Clear chat button
col1, col2 = st.columns([8, 1])
with col2:
    if st.button("🗑️", help="Clear chat"):
        st.session_state.messages = []
        st.session_state.last_company = None
        st.session_state.last_intent = None
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.subheader("Ask Your Query")
query_type = st.radio(
    "Select AI Mode:",
    ["🧠 General QA", "🔍 Research QA", "🧾 Document QA", "💼 Portfolio Management"],
    horizontal=True
)

# === GENERAL QA ===
if query_type == "🧠 General QA":
    input_method = st.radio("Talk or Type:", ["⌨️ Text", "🎤 Voice"], horizontal=True)
    transcript_text = None

    if input_method == "🎤 Voice":
        wav_audio_data = st.audio_input("Record a voice message")
        if wav_audio_data:
            st.audio(wav_audio_data, format='audio/wav')
            audio_bytes = wav_audio_data.read()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(tmp_path)
            if transcript.text:
                transcript_text = transcript.text.strip()
                st.success("Transcribed:")
                st.markdown(f"> {transcript_text}")
            else:
                st.warning("Transcription failed.")
    else:
        transcript_text = st.chat_input("Ask anything about stock market...")

    if transcript_text:
        st.session_state.messages.append({"role": "user", "content": transcript_text})
        with st.chat_message("user"):
            st.markdown(transcript_text)

        with st.spinner("🧠 Generating response..."):
            answer = summarize_with_groq({}, "stock market", transcript_text)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

# === RESEARCH QA ===
elif query_type == "🔍 Research QA":
    input_method = st.radio("Talk or Type:", ["⌨️ Text", "🎤 Voice"], horizontal=True)
    transcript_text = None
    
# === Voice Input ===
    if input_method == "🎤 Voice":
        wav_audio_data = st.audio_input("Record a voice message")
        if wav_audio_data:
            st.audio(wav_audio_data, format='audio/wav')
            # Read the uploaded file as bytes
            audio_bytes = wav_audio_data.read()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(tmp_path)
            if transcript.text:
                transcript_text = transcript.text.strip()
                st.success("Transcribed:")
                st.markdown(f"> {transcript_text}")
            else:
                st.warning("Transcription failed.")

# === Text Input ===
    else:
        transcript_text = st.chat_input("Ask anything related to company details...")

# === MAIN LOGIC "🔍 Research QA" ===
    if transcript_text:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": transcript_text})
        
        # Show user message
        with st.chat_message("user"):
            st.markdown(transcript_text)

        with st.spinner("🧠 Understanding company..."):
            stock_name = get_company_name(transcript_text,st.session_state.last_company)
            #st.write(f"🔍 Recognized company name: **{stock_name}**")

        # Step 2: Extract intent
        with st.spinner("🧠 Understanding query intent..."):
            intent = get_classify_intent(transcript_text,st.session_state.last_intent)
            #st.write(f"🔍 Recognized intent: **{intent}**")

        with st.spinner("Understanding your request..."):
            if not stock_name:
                response = "Could not identify the company. Please specify the company name."
            else:
                #st.info(f"Looking up {intent} for {stock_name}...")

                with st.spinner("Fetching company details..."):
                    api_name = get_api_to_call(intent)
                    raw_data = call_api(api_name,stock_name,intent)

                    if "error" not in raw_data:
                        #st.subheader("Raw API Response:")
                        #st.json(raw_data)
                        filtered_data = filter_json_data(raw_data,intent)
                        #st.info(filtered_data)
                        
                        # Summarize using LLM
                        with st.spinner("🧠 Generating summary..."):
                            response = summarize_with_groq(filtered_data, intent,transcript_text)

                    else:
                        response = raw_data["error"]
            # Save context for follow-up questions
            st.session_state.last_company = stock_name
            st.session_state.last_intent = intent

        # Add assistant message to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Show assistant message
        with st.chat_message("assistant"):st.markdown(response)
        if input_method == "🎤 Voice":
            #play_audio_response(response)
            play_audio_response_groq(response)

# === DOCUMENT QA ===
elif query_type == "🧾 Document QA":
    st.write("📄 Ask questions about your uploaded PDF.")
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload and process a PDF first.")
    else:
        transcript_text = st.chat_input("Ask a question related to uploaded document...")
        if transcript_text:
            st.session_state.messages.append({"role": "user", "content": transcript_text})
            with st.chat_message("user"):
                st.markdown(transcript_text)

            docs = st.session_state.vectorstore.similarity_search(transcript_text)
            llm = ChatGroq(groq_api_key=st.session_state.GROQ_API_KEY, model_name="llama3-8b-8192")
            chain = load_qa_chain(llm, chain_type="stuff")

            with st.spinner("🤖 Thinking..."):
                answer = chain.run(input_documents=docs, question=transcript_text)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

# === PORTFOLIO MANAGEMENT ===
elif query_type == "💼 Portfolio Management":
    st.write("💼 Portfolio Management is under development. Stay tuned!")