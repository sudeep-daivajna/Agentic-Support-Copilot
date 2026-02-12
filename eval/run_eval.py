import json
from pathlib import Path
from main import app  # imports your compiled graph

def main():
    cases = Path("eval/cases.jsonl").read_text(encoding="utf-8").splitlines()
    total = 0
    action_correct = 0
    schema_ok = 0

    for line in cases:
        if not line.strip():
            continue
        case = json.loads(line)
        state = {
            "user_query": case["user_query"],
            "context": "",
            "allowed_citation_ids": [],
            "model_output_text": None,
            "parsed": None,
            "error": None,
            "attempts": 0,
        }
        out = app.invoke(state)
        total += 1

        parsed = out.get("parsed")
        if parsed:
            schema_ok += 1
            
            expected = case.get("expected_next_action")
            pred = parsed.get("next_action")

            print("next_action:", pred)
            print("expected_next_action:", expected)

            if pred == expected:
                action_correct += 1


    print("total:", total)
    print("schema_ok:", schema_ok, f"({schema_ok/total:.0%})")
    print("action_accuracy:", action_correct, f"({action_correct/total:.0%})")

if __name__ == "__main__":
    main()
