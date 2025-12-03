# tests/eval.py
import json
import requests
import os
from urllib.parse import urljoin

API_URL = os.getenv("API_URL", "http://localhost:8000/query")

def run_test(q_text, expect):
    payload = {"question": q_text}
    try:
        r = requests.post(API_URL, json=payload, timeout=60)
    except Exception as e:
        return False, f"request-failed: {e}"
    if r.status_code != 200:
        return False, f"status-{r.status_code}: {r.text[:200]}"
    ans = r.json().get("answer", "")
    return (expect.lower() in ans.lower()), ans

def main():
    with open("tests.json", "r", encoding="utf-8") as f:
        tests = json.load(f)

    passed = 0
    total = len(tests)
    results = []
    for t in tests:
        ok, out = run_test(t["input"], t["expect"])
        results.append((t["input"], ok, out))
        print(f"Q: {t['input']}\nExpect: {t['expect']} | Pass: {ok}\nAnswer (truncated): {out[:300]}\n---")
        if ok:
            passed += 1

    print(f"Passed {passed}/{total} = {passed/total:.2%}")

if __name__ == "__main__":
    main()