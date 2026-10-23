"""
Helper script for headless evaluation.
Reads a training payload JSON file, passes it to backend.lambda_function.lambda_handler,
and saves the resulting model bundle JSON to an output file.
"""

import os
import sys
import json

def main():
    if len(sys.argv) < 3:
        print("Usage: python train_local_helper.py <payload_in.json> <result_out.json>")
        sys.exit(1)

    payload_path = sys.argv[1]
    result_path = sys.argv[2]

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(root_dir, "backend"))

    os.environ["TRAIN_API_KEY"] = "eval-key-32chars"
    import lambda_function as lf

    with open(payload_path, "r", encoding="utf-8") as f:
        payload_data = json.load(f)

    event = {
        "requestContext": {"http": {"method": "POST"}},
        "rawPath": "/train",
        "headers": {"x-api-key": "eval-key-32chars"},
        "body": json.dumps(payload_data)
    }

    resp = lf.lambda_handler(event)
    status_code = resp.get("statusCode", 500)
    body = json.loads(resp.get("body", "{}"))

    if status_code != 200:
        print(f"Error from Lambda handler (HTTP {status_code}): {body.get('error')}", file=sys.stderr)
        sys.exit(1)

    with open(result_path, "w", encoding="utf-8") as f_out:
        json.dump(body, f_out)

if __name__ == "__main__":
    main()
