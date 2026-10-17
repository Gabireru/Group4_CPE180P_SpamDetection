"""Client node (Windows 11 / Ubuntu WSL): sends HTTP POST JSON to /v1/predict.

Examples:
  export SERVER_URL=http://<sakura-vm-ip>:8000 ; export API_KEY=...
  python client.py "Free entry! Text WIN to 80085"
  python client.py --drift-demo       # sends out-of-distribution traffic to trigger an alert
"""
import argparse
import os
import time

import requests

URL = os.getenv("SERVER_URL", "http://localhost:8000")
HEADERS = {"X-API-Key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}


def predict(message: str):
    t0 = time.perf_counter()
    r = requests.post(f"{URL}/v1/predict", json={"message": message},
                      headers=HEADERS, timeout=5)
    rtt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    d = r.json()
    print(f"{d['label'].upper():4} p_spam={d['spam_probability']:.3f} "
          f"server={d['latency_ms']:.1f}ms round-trip={rtt:.1f}ms "
          f"drift_alert={d['drift_alert']} | {message[:60]}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("messages", nargs="*")
    ap.add_argument("--drift-demo", action="store_true")
    args = ap.parse_args()

    if args.drift_demo:
        weird = "CALL 0800123456789 NOW " * 30      # long, digit/upper heavy
        for _ in range(60):
            predict(weird)
        print("drift state:", requests.get(f"{URL}/v1/drift", timeout=5).json())
        return
    for m in args.messages or ["Hey, lunch tomorrow?",
                               "WINNER!! Claim your free prize, text 80085 now"]:
        predict(m)


if __name__ == "__main__":
    main()
