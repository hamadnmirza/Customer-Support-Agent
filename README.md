<h1 align="center">Alex</h1>

<p align="center">
The AI-powered assistant for Customer Support teams. 
</p>

---

## Overview

Alex is an intelligent QA assistant that reviews customer support responses in real-time, providing actionable feedback to maintain consistent quality across your team. Think of Alex as your tireless QA lead who never sleeps—evaluating empathy, accuracy, completeness, and tone while learning from your team's best practices. Alex was built to demonstrate: how AI can augment (not replace) human expertise in customer-facing operations while maintaining brand voice and quality standards.

By combining LLM reasoning, memory feedback loops, and human supervision, Alex demonstrates how enterprises can achieve consistent, high-quality customer experiences while reducing manual review effort.

---

## Key Capabilities

- AI-Driven QA Evaluation: Alex automatically scores support replies against defined quality metrics.  
- Reviewer Personas: tell Alex what type of review you want. Do you want his review to be *Polite*, *Technical*, or *Empathetic*?  
- Memory Feedback Loop: approved suggestions are stored in a vector database for contextual learning. This allows Alex to continuously learn from fellow teammates' approved examples, adapting his responses over time to match the organisation’s tone, quality standards, and decision patterns. 
- Autonomous Reply Drafting: Alex generates accurate and brand-aligned replies for unhandled tickets.  
- Zendesk Integration: Alex fetches live tickets and posts internal QA notes automatically.  
- Audit Logging: Alex tracks all your approvals and rejections for transparency and governance.  

---

## Business Value

Alex demonstrates how autonomous QA and coaching can transform customer support operations:
- He enforces tone of voice with every interaction.
- He uses human feedback to continously refine his evaluation and writing styles.
- He keeps a verifiable record of every decision for reporting/governance. 

---

## Product Roadmap

- Continuous Learning: persist approved suggestions to a database (Postgres/Supabase) and tune scoring logic.
- Platform Integrations: extend support to Intercom, Freshdesk, and Salesforce Service Cloud.

---
## Getting Started

- Prerequisites: OpenAI API key, Zendesk credentials, and some actual tickets on Zendesk.
- Try it live by cloning this repository and running the app via streamlit. 

--- 

## Demo: 

<img width="1926" height="994" alt="Screenshot 2025-11-12 at 21 28 26" src="https://github.com/user-attachments/assets/91844cb7-84ea-4e4f-9ce2-66f8404cbfce" />


---

## Technical Architecture: 
Stack:
- Frontend: Streamlit (rapid prototyping, easy deployment).
- LLM: OpenAI GPT-4o-mini (cost-effective, fast responses).
- Embeddings: sentence-transformers. 
- Integration: Zendesk API v2.

Key Design Decisions:
- Bring-your-own-key model for security and cost control.
- Stateless architecture for Streamlit Cloud compatibility.
- Human-in-the-loop by design – AI suggests, humans decide.



NOTE: This is a personal project built for experimenting with AI applications in customer support operations.
