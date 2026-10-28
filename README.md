# Kirana Ordering Assistant (educational prototype)

Files: `app.py` (Streamlit UI) · `engine.py` (guardrails, reader, catalog matcher, pricing, nudge) ·
`eval_llm.py` (accuracy test on 24 labelled order lines) · `data/` (catalog, stores, pre-computed nudge scores).

Run locally:  pip install streamlit pandas  →  streamlit run app.py
Evaluate:     python eval_llm.py                          (rule-based baseline, free)
              set OPENAI_API_KEY=sk-...  then  python eval_llm.py   (AI reader, a few cents)

Deploy (Streamlit Community Cloud): push this folder to a public GitHub repo → share.streamlit.io →
Create app → main file `app.py` → Advanced settings → Secrets: paste the lines from
`.streamlit/secrets.toml.example` with your key → Deploy. Without a key the app still works using the rule-based reader.
