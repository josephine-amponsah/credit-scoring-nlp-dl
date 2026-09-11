# frontend/services/api_client.py
import os
import httpx

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

def get_quarterly_data(period=None, loan_purpose=None):
    params = {}
    if period:
        params["period"] = period
    if loan_purpose:
        if isinstance(loan_purpose, (list, tuple, set)):
            params["loan_purpose"] = ",".join(str(v).strip() for v in loan_purpose if str(v).strip())
        else:
            params["loan_purpose"] = str(loan_purpose)

    resp = httpx.get(f"{API_BASE}/overview/summary", params=params)
    resp.raise_for_status()
    return resp.json()