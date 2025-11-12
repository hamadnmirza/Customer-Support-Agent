"""
Alex - AI Support Reviewer
A Streamlit application for reviewing and improving customer support responses
using OpenAI and optional Zendesk integration.
"""

import json
import requests
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================================
# FAISS Setup (Optional Vector Search)
# ============================================================================
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

# ============================================================================
# Constants
# ============================================================================
EMBED_DIM = 384

PERSONA_TONES = {
    "Polite": "Provide balanced and courteous feedback.",
    "Technical": "Focus on factual accuracy and clarity.",
    "Empathetic": "Prioritise warmth, empathy, and reassurance.",
}

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Alex – AI Support Reviewer",
    layout="wide"
)
st.title("🧭 Alex – AI Support Reviewer")

# ============================================================================
# Utility Functions
# ============================================================================
def parse_llm_json(response_text: str) -> dict:
    """
    Extract and parse JSON from LLM responses.
    Handles markdown code fences and extra text around JSON.
    """
    text = response_text.strip()
    
    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.startswith("```")]
        text = "\n".join(lines)
    
    # Extract JSON object
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]
    
    return json.loads(text)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors for cosine similarity calculation."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


@st.cache_resource(show_spinner=False)
def load_sentence_encoder():
    """Load the sentence transformer model (cached)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================================
# Session State Initialization
# ============================================================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "api_key": "",
        "zendesk_subdomain": "",
        "zendesk_email": "",
        "zendesk_token": "",
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# ============================================================================
# Memory Management Class
# ============================================================================
class MemoryManager:
    """Manages the semantic memory system for storing approved responses."""
    
    def __init__(self, encoder):
        self.encoder = encoder
        self.texts = []
        
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(EMBED_DIM)
            self.vectors = None
        else:
            self.index = None
            self.vectors = np.empty((0, EMBED_DIM), dtype=np.float32)
    
    def add(self, text: str):
        """Add a text entry to memory."""
        embedding = self.encoder.encode([text]).astype("float32")
        self.texts.append(text)
        
        if FAISS_AVAILABLE:
            self.index.add(embedding)
        else:
            self.vectors = np.vstack([self.vectors, embedding])
    
    def recall(self, query: str, top_k: int = 2) -> list:
        """Retrieve the most similar texts from memory."""
        if not self.texts:
            return []
        
        query_embedding = self.encoder.encode([query]).astype("float32")
        
        if FAISS_AVAILABLE:
            distances, indices = self.index.search(
                query_embedding,
                min(top_k, len(self.texts))
            )
            valid_indices = [i for i in indices[0] if 0 <= i < len(self.texts)]
            return [self.texts[i] for i in valid_indices]
        else:
            # Cosine similarity using normalized vectors
            query_norm = normalize_vectors(query_embedding)
            memory_norm = normalize_vectors(self.vectors)
            similarities = (query_norm @ memory_norm.T)[0]
            
            top_indices = similarities.argsort()[::-1][:top_k]
            return [self.texts[i] for i in top_indices]


# ============================================================================
# Zendesk Integration
# ============================================================================
class ZendeskClient:
    """Handles all Zendesk API interactions."""
    
    def __init__(self, subdomain: str, email: str, token: str):
        self.base_url = f"https://{subdomain}.zendesk.com/api/v2"
        self.auth = (f"{email}/token", token)
    
    def fetch_recent_tickets(self, limit: int = 5) -> list:
        """Fetch recent tickets from Zendesk."""
        url = f"{self.base_url}/tickets.json?page[size]={limit}"
        response = requests.get(url, auth=self.auth, timeout=20)
        response.raise_for_status()
        return response.json().get("tickets", [])
    
    def post_internal_note(self, ticket_id: int, note_text: str):
        """Post an internal note to a Zendesk ticket."""
        data = {
            "ticket": {
                "comment": {
                    "body": f"Alex QA Feedback:\n{note_text}",
                    "public": False
                }
            }
        }
        url = f"{self.base_url}/tickets/{ticket_id}.json"
        response = requests.put(url, json=data, auth=self.auth, timeout=20)
        response.raise_for_status()


# ============================================================================
# AI Review Functions
# ============================================================================
def generate_review_prompt(persona: str, persona_tone: str, context: str,
                          ticket_text: str, agent_reply: str) -> str:
    """Generate the prompt for reviewing an agent's reply."""
    return f"""
You are Alex, a {persona} QA reviewer for customer support replies.
{persona_tone}

Here are prior approved examples (memory context):
{context}

Ticket: {ticket_text}
Agent reply: {agent_reply}

Evaluate the reply from 1–10 for:
1. Empathy
2. Accuracy
3. Resolution completeness
4. Tone appropriateness

Then propose an improved version that better resolves the issue.
Respond in JSON:
{{ "scores": {{ "empathy": x, "accuracy": x, "resolution": x, "tone": x }}, "suggested_reply": "..." }}
""".strip()


def generate_draft_prompt(persona: str, persona_tone: str, ticket_text: str) -> str:
    """Generate the prompt for drafting a new reply."""
    return f"""
You are an AI customer support agent. Draft a complete, friendly, and accurate reply
to the following customer message, using a {persona} style.

Ticket: {ticket_text}

Make sure the tone aligns with: {persona_tone}.
""".strip()


# ============================================================================
# Sidebar Configuration
# ============================================================================
def setup_sidebar():
    """Configure the sidebar with API keys and settings."""
    st.sidebar.header("🔑 Configuration")
    
    # OpenAI Settings
    st.sidebar.subheader("OpenAI Settings")
    api_key = st.sidebar.text_input(
        "OpenAI API key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-...",
        help="Get yours at https://platform.openai.com/account/api-keys",
    )
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
    
    st.sidebar.markdown("---")
    
    # Zendesk Settings
    st.sidebar.subheader("Zendesk Settings")
    
    subdomain = st.sidebar.text_input(
        "Zendesk Subdomain",
        value=st.session_state.zendesk_subdomain,
        placeholder="example",
    )
    email = st.sidebar.text_input(
        "Zendesk Email",
        value=st.session_state.zendesk_email,
        placeholder="you@company.com",
    )
    token = st.sidebar.text_input(
        "Zendesk API Token",
        value=st.session_state.zendesk_token,
        type="password",
        placeholder="Paste token",
    )
    
    # Update session state
    if subdomain != st.session_state.zendesk_subdomain:
        st.session_state.zendesk_subdomain = subdomain
    if email != st.session_state.zendesk_email:
        st.session_state.zendesk_email = email
    if token != st.session_state.zendesk_token:
        st.session_state.zendesk_token = token
    
    return api_key, subdomain, email, token


# ============================================================================
# Main Application
# ============================================================================
def main():
    """Main application logic."""
    
    # Load environment variables
    load_dotenv()
    
    # Initialize session state
    init_session_state()
    
    # Setup sidebar and get credentials
    api_key, zendesk_subdomain, zendesk_email, zendesk_token = setup_sidebar()
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=api_key) if api_key else None
    if not openai_client:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to enable AI features.")
    
    # Initialize Zendesk client
    zendesk_client = None
    if zendesk_subdomain and zendesk_email and zendesk_token:
        zendesk_client = ZendeskClient(zendesk_subdomain, zendesk_email, zendesk_token)
        st.sidebar.success("✅ Zendesk connected.")
    else:
        st.sidebar.info("ℹ️ Zendesk not configured — manual mode only.")
    
    # Initialize memory system
    encoder = load_sentence_encoder()
    memory = MemoryManager(encoder)
    
    # Persona selection
    persona = st.radio(
        "🎭 Choose QA Reviewer Persona:",
        ["Polite", "Technical", "Empathetic"],
        horizontal=True,
    )
    persona_tone = PERSONA_TONES[persona]
    
    # Zendesk ticket fetching
    st.markdown("### 🎟️ Fetch Zendesk Tickets")
    ticket_id = None
    ticket_description = ""
    
    if zendesk_client:
        try:
            tickets = zendesk_client.fetch_recent_tickets()
            if tickets:
                ticket_options = [
                    f"{ticket['id']} – {ticket.get('subject', '(no subject)')}"
                    for ticket in tickets
                ]
                choice = st.selectbox("Select a Zendesk ticket:", ticket_options)
                ticket_id = int(choice.split("–")[0].strip())
                
                selected_ticket = next(t for t in tickets if t["id"] == ticket_id)
                ticket_description = selected_ticket.get("description", "")
            else:
                st.info("No tickets fetched. Use manual inputs below.")
        except Exception as error:
            st.warning(f"Zendesk fetch failed: {error}")
    
    # Manual input fields
    ticket_text = st.text_area(
        "🧾 Customer message:",
        value=ticket_description or "My refund still hasn't arrived after 2 weeks.",
    )
    agent_reply = st.text_area(
        "💬 Agent reply:",
        value="Please wait a few more days."
    )
    
    # Review button
    if st.button("🔍 Review Reply"):
        if not openai_client:
            st.error("Enter your OpenAI API key first.")
            return
        
        with st.spinner("Analyzing reply..."):
            # Retrieve context from memory
            prior_examples = memory.recall(ticket_text, top_k=2)
            context_text = (
                "\n\n".join([f"- {example}" for example in prior_examples])
                if prior_examples
                else "No prior feedback yet."
            )
            
            # Generate review
            prompt = generate_review_prompt(
                persona, persona_tone, context_text, ticket_text, agent_reply
            )
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            
            raw_content = response.choices[0].message.content
            
            try:
                result = parse_llm_json(raw_content)
            except Exception:
                st.error("The model returned unexpected output. Showing raw text:")
                st.code(raw_content)
                return
            
            # Display evaluation scores
            st.subheader("🧠 Evaluation Scores")
            st.json(result["scores"])
            
            # Display original vs suggested reply
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 💬 Original Reply")
                st.info(agent_reply)
            with col2:
                st.markdown("### 🤖 Suggested Reply")
                st.success(result["suggested_reply"])
            
            # Auto-draft new reply
            if st.button("✉️ Auto-draft New Reply"):
                with st.spinner("Drafting autonomous reply..."):
                    draft_prompt = generate_draft_prompt(persona, persona_tone, ticket_text)
                    
                    draft_response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": draft_prompt}],
                    )
                    
                    st.markdown("### ✉️ Auto-Drafted Reply")
                    st.write(draft_response.choices[0].message.content)
            
            # Feedback buttons
            st.subheader("🧩 Feedback")
            
            col_approve, col_reject = st.columns(2)
            
            with col_approve:
                if st.button("👍 Approve Suggestion"):
                    memory.add(result["suggested_reply"])
                    st.success("✅ Approved suggestion added to memory.")
            
            with col_reject:
                if st.button("👎 Reject Suggestion"):
                    st.warning("👎 Suggestion rejected — not added to memory.")
            
            # Post to Zendesk
            if ticket_id and zendesk_client and st.button("📩 Send Feedback to Zendesk"):
                note_text = (
                    f"Scores: {result['scores']}\n\n"
                    f"Suggested Reply:\n{result['suggested_reply']}"
                )
                try:
                    zendesk_client.post_internal_note(ticket_id, note_text)
                    st.success("✅ Feedback posted to Zendesk as an internal note.")
                except Exception as error:
                    st.error(f"Failed to post to Zendesk: {error}")


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    main()
