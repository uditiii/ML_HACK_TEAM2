# PathSnap UI (Person A)

Streamlit front-end. Runs on **Laptop 2** and talks to the FastAPI backend on **Laptop 1**.
It only sends a `trip_id`; it never requests or shows ground-truth arcs or the reference route.

## Run (demo)
Linux/macOS:
    API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py
Windows PowerShell:
    $env:API_URL="http://<laptop1-ip>:8000"; streamlit run ui/app.py

## Test without the real API (NOT for the demo)
    uvicorn ui.api_stub:app --port 8000      # terminal 1: FAKE data, clearly marked
    streamlit run ui/app.py                  # terminal 2

The JSON format the real API must return is documented at the top of `ui/api_stub.py`.
