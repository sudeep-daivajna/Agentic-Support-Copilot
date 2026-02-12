from __future__ import annotations
from typing import TypedDict, Optional, Literal, Dict, Any
import json
import boto3
from pydantic import BaseModel, Field, ValidationError

from langgraph.graph import StateGraph, START, END

import os
from dotenv import load_dotenv

load_dotenv()

SUPPORT_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "next_action": {"type": "string", "enum": ["reply", "ask_clarifying", "escalate"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "next_action", "confidence", "citations"],
}


# ----------------------------
# 1) Define the output contract (generation-quality target)
# ----------------------------
class SupportResponse(BaseModel):
    answer: str = Field(..., description="Customer-facing answer.")
    next_action: Literal["reply", "ask_clarifying", "escalate"] = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    citations: list[str] = Field(
        default_factory=list,
        description="IDs/refs for evidence (can be empty in this starter).",
    )


# ----------------------------
# 2) Define LangGraph state
# ----------------------------
class GraphState(TypedDict):
    user_query: str
    context: str
    model_output_text: Optional[str]
    parsed: Optional[Dict[str, Any]]
    error: Optional[str]
    attempts: int

    last_error: Optional[str]
    last_model_output_text: Optional[str]



# ----------------------------
# 3) Bedrock call via Converse API (boto3)
# ----------------------------
def bedrock_converse(prompt: str, model_id: str, region: str = "us-east-1") -> str:
    client = boto3.client("bedrock-runtime", region_name=region)

    resp = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 600, "temperature": 0.0},

        # ✅ Structured outputs
        outputConfig={
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "name": "support_response",
                        "description": "Return a SupportResponse JSON object",
                        "schema": json.dumps(SUPPORT_RESPONSE_SCHEMA),
                    }
                },
            }
        },
    )

    blocks = resp["output"]["message"]["content"]

    # With structured outputs, Bedrock can return a JSON content block.
    for b in blocks:
        if "json" in b:
            return json.dumps(b["json"])

    # Fallback if model returns text (should be rare with structured outputs)
    return "\n".join(b.get("text", "") for b in blocks if "text" in b).strip()



# ----------------------------
# 4) Graph nodes
# ----------------------------
# MODEL_ID = "mistral.voxtral-mini-3b-2507"  # example IDs are shown in AWS docs for Converse. :contentReference[oaicite:5]{index=5}
# AWS_REGION = "us-east-1"


def draft_node(state: GraphState) -> GraphState:
    prompt = f"""
    You are a JSON-only API. Output MUST be a SINGLE JSON object.
    - No markdown
    - No triple backticks
    - No explanations
    - First char must be {{ and last char must be }}

    Return JSON with EXACT keys:
    - answer: string
    - next_action: one of ["reply", "ask_clarifying", "escalate"]
    - confidence: number between 0 and 1
    - citations: array of strings

    If context is insufficient, set next_action="ask_clarifying" and confidence <= 0.5.

    USER_QUERY: {state["user_query"]}
    CONTEXT: {state["context"]}
    """.strip()

    try:
        out = bedrock_converse(prompt, model_id=os.getenv("MODEL_ID"), region=os.getenv("AWS_REGION"))
        return {
            **state,
            "model_output_text": out,
            "attempts": state["attempts"] + 1,
            "error": None,
        }
    except Exception as e:
        return {
            **state,
            "error": f"bedrock_call_failed: {e}",
            "attempts": state["attempts"] + 1,
        }


def validate_node(state: GraphState) -> GraphState:
    if not state.get("model_output_text"):
        return {**state, "error": state.get("error") or "no_model_output"}

    raw = state["model_output_text"].strip()

    # Try strict JSON parse
    try:
        obj = json.loads(raw)
    except Exception as e:
        # return {**state, "parsed": None, "error": f"json_parse_failed: {e}"}
        return {**state, "parsed": None, "error": f"json_parse_failed: {e}", "last_error": f"json_parse_failed: {e}", "last_model_output_text": state.get("model_output_text")}


    # Validate against schema (format correctness)
    try:
        validated = SupportResponse(**obj)
        return {**state, "parsed": validated.model_dump(), "error": None}
    except ValidationError as ve:
        return {
            **state,
            "parsed": None,
            "error": f"schema_validation_failed: {ve.errors()[:2]}",
        }


def repair_node(state: GraphState) -> GraphState:
    # If we fail, ask the model to fix *only* formatting/schema.
    prompt = f"""
    Your previous output was invalid.

    Return ONLY ONE JSON object with EXACT keys:
    answer, next_action, confidence, citations
    (No markdown, no triple backticks, no schema.)

    USER_QUERY: {state["user_query"]}
    CONTEXT: {state["context"]}

    Now return the corrected JSON object ONLY:
    """.strip()

    out = bedrock_converse(prompt, model_id=os.getenv("MODEL_ID"), region=os.getenv("AWS_REGION"))
    return {**state, "model_output_text": out, "attempts": state["attempts"] + 1}


def route_after_validate(state: GraphState) -> Literal["ok", "repair", "fail"]:
    if state.get("error") is None and state.get("parsed") is not None:
        return "ok"
    if state["attempts"] >= 3:
        return "fail"
    return "repair"


def fallback_node(state: GraphState) -> GraphState:
    # Safe fallback (never hallucinate)
    return {
        **state,
        "parsed": {
            "answer": "I’m not confident I can answer that correctly yet. Could you share one more detail?",
            "next_action": "ask_clarifying",
            "confidence": 0.3,
            "citations": [],
        },
        # "error": None,
    }


# ----------------------------
# 5) Build graph (conditional edges)
# ----------------------------
g = StateGraph(GraphState)
g.add_node("draft", draft_node)
g.add_node("validate", validate_node)
g.add_node("repair", repair_node)
g.add_node("fallback", fallback_node)

g.add_edge(START, "draft")
g.add_edge("draft", "validate")

g.add_conditional_edges(
    "validate",
    route_after_validate,
    {
        "ok": END,
        "repair": "repair",
        "fail": "fallback",
    },
)
g.add_edge("repair", "validate")
g.add_edge("fallback", END)

app = g.compile()

if __name__ == "__main__":
    init: GraphState = {
        "user_query": "I was charged twice. What should I do?",
        "context": "Refund policy: refunds are issued within 5-7 business days after confirmation.",
        "model_output_text": None,
        "parsed": None,
        "error": None,
        "attempts": 0,
    }
    result = app.invoke(init)
    print("attempts:", result["attempts"])
    print("last_error:", result.get("last_error"))
    print("last_model_output_text:\n", result.get("last_model_output_text"))

    print(json.dumps(result["parsed"], indent=2))
