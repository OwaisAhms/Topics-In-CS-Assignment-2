import json
import requests
import sys
from datetime import datetime

API_URL = "http://localhost:8000/query"

def run_tests():
    # Load test cases
    try:
        with open("tests.json", "r") as f:
            tests = json.load(f)
    except FileNotFoundError:
        print("ERROR: tests.json not found.")
        sys.exit(1)

    total = len(tests)
    passed = 0

    print(f"Running {total} tests...\n")

    for i, test in enumerate(tests, start=1):
        query = test["input"]
        expected_substring = test["expect"].lower()

        # Send query to FastAPI
        response = requests.post(API_URL, json={"query": query})

        if response.status_code != 200:
            print(f"[{i}] ❌ API ERROR {response.status_code}")
            continue

        answer = response.json().get("answer", "").lower()

        # Check match
        if expected_substring in answer:
            result = "PASS"
            passed += 1
        else:
            result = "FAIL"

        print(f"[{i}] {result}")
        print(f"  Q: {query}")
        print(f"  Expected substring: '{expected_substring}'")
        print(f"  Got: {answer[:100]}...\n")

    print("==== RESULTS ====")
    print(f"Passed: {passed}/{total}")
    print(f"Score: {passed/total:.2%}")

    # Log results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"eval_log_{timestamp}.txt", "w") as log:
        log.write(f"Score: {passed}/{total} ({passed/total:.2%})\n")

    print("\nResults saved to log file.")

if __name__ == "__main__":
    run_tests()