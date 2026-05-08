# Lab Report: Day 22 – LLMops & Prompt Versioning

## 1. Project Overview
This lab focused on building a robust RAG (Retrieval-Augmented Generation) pipeline with advanced MLOps practices, including prompt versioning using LangChain Prompt Hub, experiment tracking via LangSmith, and automated evaluation using RAGAS metrics.

## 2. RAGAS Evaluation Results

| Metric             | Prompt V1 (Concise) | Prompt V2 (Structured) | Winner |
|--------------------|---------------------|------------------------|--------|
| **Faithfulness**   | **0.8500**          | 0.6317                 | V1     |
| **Answer Relevancy**| 0.6320              | **0.8040**             | V2     |
| **Context Recall** | 0.4469              | 0.3560                 | V1     |
| **Context Precision**| 0.7000            | 0.7000                 | Draw   |

**Analysis:**
- **Prompt V1 (Concise)** outperformed V2 in **Faithfulness** (0.85 vs 0.63). This is because the concise instructions forced the model to stay strictly within the provided context, reducing "hallucinations."
- **Prompt V2 (Structured)** won in **Answer Relevancy** (0.80 vs 0.63). The structured instructions encouraged the model to provide more comprehensive and better-organized answers, which aligned better with user intent despite slightly lower grounding.

## 3. Implementation Highlights

### A. Prompt Versioning & A/B Testing
- Successfully implemented **deterministic A/B routing** using MD5 hashing of request IDs.
- Integrated **LangChain Prompt Hub** to push and pull prompt versions (`lab22-concise-v1` and `lab22-structured-v2`), ensuring a clear decoupling of prompt engineering from application logic.

### B. Overcoming API Limitations (Troubleshooting)
One of the major challenges was the strict **TPD (Tokens Per Day)** and **RPM (Requests Per Minute)** limits on Groq and Gemini Free Tiers.
- **Solution:** Developed a custom `ChatGroqSafe` and `ChatGeminiSafe` wrapper that intercepts LangChain's `n=3` parameter (forced by RAGAS) and implements a strict **4.5-second pacing delay** between requests. This ensured the evaluation completed without 429 RateLimit or Timeout errors.

### C. Guardrails AI Integration
- **PII Detection:** Implemented a custom validator that redacts sensitive data (Emails, Phone numbers, SSNs) using regex-based guards.
- **JSON Repair:** Built a validator that automatically strips markdown fences and fixes common JSON formatting errors (single quotes, trailing commas) before outputting to the user.

## 4. Conclusion
The pipeline successfully met the target criteria (**Faithfulness > 0.8**). The combination of LangSmith for observability, Prompt Hub for versioning, and RAGAS for evaluation provides a professional-grade framework for iterating on LLM applications.

---
**Evidence Files:**
- `02_ab_routing_log.txt`: A/B Testing logs.
- `data/ragas_report.json`: Full RAGAS metrics.
- `evidence/04_pii_demo_log.txt`: PII redaction proof.
- `evidence/04_json_demo_log.txt`: JSON repair proof.
