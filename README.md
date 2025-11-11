# Reason AI – Universal Data Cleaner

One-file Streamlit app with built-in API mode for NeuralSeek.

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Deploy on Streamlit Cloud
- Push this repo to GitHub (public).
- On https://share.streamlit.io select this repo and `app.py`.

## API mode (for NeuralSeek REST)
GET:
https://<your-app>.streamlit.app/?api=1&question=<exact title>

Example:
https://<your-app>.streamlit.app/?api=1&question=Monthly%20revenue%20(sum%20of%20fee)

Tip: Use the app UI to see the **supported_questions** list.

## Notes
- Cleaned files are saved to `artifacts/` (gitignored).
- Never commit secrets or API keys. Use Streamlit **Secrets**.
