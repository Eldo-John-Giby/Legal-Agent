# Legal Agent (AutoGen)

Run a mock trial with evidence-cited arguments (Judge/Plaintiff/Defendant) plus optional Human-in-the-loop and Weaviate-backed RAG retrieval.

1) `copy .env.example .env` and set `GROQ_API_KEY`
2) `pip install -r requirements.txt`
3) `python run_court.py --case data\sample_case.txt`

Always-on:
- Human-in-the-loop is enabled (a Human turn is included; press Enter to skip)
- Indian Constitution retrieval is enabled from `data\\constitution.txt` (expects ARTICLE headings)

Weaviate (optional):
- Set `WEAVIATE_URL` (default `http://localhost:8080`) to enable Weaviate RAG indexing/search; otherwise a local fallback search is used

Outputs are written under `outputs\<timestamp>\`.
