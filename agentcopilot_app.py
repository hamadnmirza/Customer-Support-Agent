import streamlit as st
from openai import OpenAI
import json
import datetime
import faiss
import numpy as np
import requests
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# -------------------------
# LOAD ENVIRONMENT VARIABLES
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN")
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
ZENDESK_API_TOKEN = os.getenv("ZENDESK_API_TOKEN")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# APP CONFIG
# -------------------------
st.set_page_config(page_title="AgentCopilot – AI Support Reviewer", layout="wide")
st.title("🧭 AgentCopilot – AI Support Reviewer")

# -------------------------
# ZENDESK HELPERS
# -------------------------
if ZENDESK_SUBDOMAIN and ZENDESK_EMAIL and ZENDESK_API_TOKEN:
    ZENDESK_BASE = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2"
    auth = (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)
else:
    ZENDESK_BASE, auth = None, None

def fetch_recent_tickets(limit=5):
    if not ZENDESK_BASE:
        return []
    url = f"{ZENDESK_BASE}/tickets.json?page[size]={limit}"
    r = requests.get(url, auth=auth)
    r.raise_for_status()
    return r.json()["tickets"]

def post_internal_note(ticket_id, note):
    if not ZENDESK_BASE:
        return
    data = {"ticket": {"comment": {"body": f"AgentCopilot QA Feedback:\n{note}", "public": False}}}
    url = f"{ZENDESK_BASE}/tickets/{ticket_id}.json"
    r = requests.put(url, json=data, auth=auth)
    r.raise_for_status()

# -------------------------
# LLM MEMORY (FAISS)
# -------------------------
dimension = 384
index = faiss.IndexFlatL2(dimension)
memory = []
model = SentenceTransformer("all-MiniLM-L6-v2")

def add_to_memory(text):
    embedding = model.encode([text]).astype("float32")
    index.add(embedding)
    memory.append(text)

def recall_from_memory(query, k=2):
    if len(memory) == 0:
        return []
    q_emb = model.encode([query]).astype("float32")
    D, I = index.search(q_emb, k)
    return [memory[i] for i in I[0] if i < len(memory)]

# -------------------------
# PERSONA CONFIGURATION
# -------------------------
persona = st.radio(
    "🎭 Choose QA Reviewer Persona:",
    ["Polite", "Technical", "Empathetic"],
    horizontal=True
)
persona_tone = {
    "Polite": "Provide balanced and courteous feedback.",
    "Technical": "Focus on factual accuracy and clarity.",
    "Empathetic": "Prioritise warmth, empathy, and reassurance."
}[persona]

# -------------------------
# FETCH ZENDESK TICKETS
# -------------------------
st.markdown("### 🎟️ Fetch Zendesk Tickets")
ticket_id = None
tickets = []
try:
    tickets = fetch_recent_tickets()
    if tickets:
        chosen = st.selectbox(
            "Select a Zendesk ticket:",
            [f"{t['id']} – {t['subject']}" for t in tickets]
        )
        ticket_id = int(chosen.split("–")[0].strip())
        ticket = next(t["description"] for t in tickets if t["id"] == ticket_id)
    else:
        st.info("No tickets fetched or Zendesk not configured. You can test manually below.")
except Exception as e:
    st.warning(f"Zendesk fetch failed: {e}")
    ticket = ""
    ticket_id = None

# -------------------------
# MANUAL INPUTS (Fallback)
# -------------------------
ticket = st.text_area("🧾 Customer message:", ticket or "My refund still hasn’t arrived after 2 weeks.")
reply = st.text_area("💬 Agent reply:", "Please wait a few more days.")

# -------------------------
# MAIN REVIEW FUNCTION
# -------------------------
if st.button("🔍 Review Reply"):
    with st.spinner("Analyzing reply..."):
        recalled_examples = recall_from_memory(ticket)
        context_text = "\n\n".join([f"- {e}" for e in recalled_examples]) if recalled_examples else "No prior feedback yet."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""
                You are AgentCopilot, a {persona} QA reviewer for customer support replies.
                {persona_tone}

                Here are prior approved examples (memory context):
                {context_text}

                Ticket: {ticket}
                Agent reply: {reply}

                Evaluate the reply from 1–10 for:
                1. Empathy
                2. Accuracy
                3. Resolution completeness
                4. Tone appropriateness

                Then propose an improved version that better resolves the issue.
                Respond in JSON:
                {{ "scores": {{ "empathy": x, "accuracy": x, "resolution": x, "tone": x }}, "suggested_reply": "..." }}
                """
            }]
        )

        result = json.loads(response.choices[0].message.content)

        st.subheader("🧠 Evaluation Scores")
        st.json(result["scores"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 💬 Original Reply")
            st.info(reply)
        with col2:
            st.markdown("### 🤖 Suggested Reply")
            st.success(result["suggested_reply"])

        # -------------------------
        # AUTO-DRAFT REPLY
        # -------------------------
        if st.button("🪫 Auto-draft New Reply"):
            with st.spinner("Drafting autonomous reply..."):
                draft = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": f"""
                        You are an AI customer support agent. Draft a complete, friendly, and accurate reply
                        to the following customer message, using a {persona} style.

                        Ticket: {ticket}

                        Make sure the tone aligns with: {persona_tone}.
                        """
                    }]
                )
                auto_draft = draft.choices[0].message.content
                st.markdown("### ✉️ Auto-Drafted Reply")
                st.write(auto_draft)

        # -------------------------
        # FEEDBACK LOOP
        # -------------------------
        st.subheader("🧩 Feedback")
        colA, colB = st.columns(2)

        def log_feedback(decision):
            log_entry = {
                "timestamp": str(datetime.datetime.now()),
                "decision": decision,
                "persona": persona,
                "ticket": ticket,
                "original_reply": reply,
                "suggested_reply": result["suggested_reply"],
                "scores": result["scores"]
            }
            with open("agentcopilot_feedback_log.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            if decision == "approved":
                add_to_memory(result["suggested_reply"])
                st.success("✅ Approved suggestion added to LLM memory.")
            else:
                st.warning("👎 Suggestion rejected — not added to memory.")

        with colA:
            if st.button("👍 Approve Suggestion"):
                log_feedback("approved")
        with colB:
            if st.button("👎 Reject Suggestion"):
                log_feedback("rejected")

        # -------------------------
        # ZENDESK FEEDBACK POSTING
        # -------------------------
        if ticket_id and st.button("📩 Send Feedback to Zendesk"):
            note_text = f"Scores: {result['scores']}\n\nSuggested Reply:\n{result['suggested_reply']}"
            try:
                post_internal_note(ticket_id, note_text)
                st.success("✅ Feedback posted to Zendesk as an internal note.")
            except Exception as e:
                st.error(f"Failed to post to Zendesk: {e}")
