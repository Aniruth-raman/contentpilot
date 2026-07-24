"""
ContentPilot (Python edition) — Telegram delivery with human-in-the-loop approval.

Run:
    pip install requests openrouter python-dotenv
    export OPENROUTER_API_KEY=your_api_key_here
    export TELEGRAM_BOT_TOKEN=your_bot_token
    export TELEGRAM_CHAT_ID=your_chat_id
    python content_pilot.py

Intended to be run on a schedule via GitHub Actions (see generate.yml).

This module is also imported by handle_callback.py, which is triggered by a
`repository_dispatch` event when the user taps Approve/Regenerate in Telegram.
This file only ever GENERATES and SENDS a draft — it never commits anything
to history and never advances the rotation counter. That only happens once a
human approves it (see handle_callback.py), so tapping Regenerate a few times
before approving never burns through extra topic/format/angle combos.
"""

import os
import json
import hashlib
import itertools
from pathlib import Path
from openrouter import OpenRouter

import requests
from dotenv import load_dotenv

load_dotenv()  # load .env file if present

# ---- Config -----------------------------------------------------------

TOPICS = ["Java",
          "Spring Boot",
          "Microservices",
          "Kafka",
          "System Design",
          "Distributed Systems",
          "Database (SQL/NoSQL)",
          "Performance Optimization",
          "Concurrency / Multithreading",
          "Design Patterns",
          "Debugging / Production Issues",
          "API Design",
          "Security (OAuth2/JWT)",
          "GenAI Integration"]
FORMATS = [ "Deep Dive",
            "Quick Tip",
            "Mistake / Pitfall",
            "Interview Question",
            "Real-world Scenario",
            "Comparison",
            "Step-by-step Guide",
            "Myth Busting",
            "Code Walkthrough",
            "Architecture Breakdown"]
ANGLES = [
    "Most developers do this wrong",
            "This will break in production",
            "I learned this the hard way",
            "If you don’t know this, you’ll fail interviews",
            "This looks simple but isn’t",
            "Here’s what no one tells you",
            "Stop doing this in your code",
            "This one mistake costs hours of debugging",
            "You’re overengineering this",
            "This is why your system doesn’t scale"
]

# Lives inside the repo so it can be committed back by the GH Actions workflow.
STATE_FILE = Path(__file__).parent / "counter.json"

# Append-only log of every APPROVED post — never truncated, never overwritten.
HISTORY_FILE = Path(__file__).parent / "post_history.jsonl"

# How many recent posts to remind the model about, so it avoids repeating ideas/phrasing.
RECENT_CONTEXT_COUNT = 8

# How similar (cosine similarity, 0-1) a new post can be to any past post before we
# consider it a semantic duplicate and retry. 1.0 = identical, 0.0 = unrelated.
SIMILARITY_THRESHOLD = 0.87

EMBEDDING_MODEL = "google/gemini-embedding-2"

# How many times to retry generation if a candidate is too similar to a past post.
MAX_DEDUP_RETRIES = 3

# Model selection now lives in an OpenRouter preset, not here — this is just the
# preset/model id OpenRouter routes to.
MODEL = "google/gemini-3.6-flash"

client = OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])

PROMPT_TEMPLATE = """
You are a senior software engineer writing a high-quality LinkedIn post for developers.

Generate content using:

Topic: {topic}
Format: {format}
Angle: {angle}

Requirements:

- Output ONLY the final LinkedIn post.
- Start with a strong curiosity-driven hook that makes developers stop scrolling.
- Write in a conversational, human, slightly opinionated tone.
- Make the content feel like it comes from real production experience.
- Add relevant emojis naturally, but do not overuse them.
- Avoid generic textbook explanations.
- Avoid vague statements such as:
  - "Many developers do this wrong"
  - "This can cause issues"
  - "Security is important"
- Always explain:
  - What exactly is wrong
  - Why developers do it
  - What problems it creates in production
  - How experienced engineers solve it
- Include at least one concrete example, scenario, incident, or real-world situation.
- Include at least one actionable takeaway that developers can immediately apply.
- Do not just identify problems. Always provide practical solutions.
- Keep the content insightful and educational rather than motivational.
- Do NOT use bold, italics, markdown, asterisks, bullets, numbered lists, or special formatting.
- Do NOT mention AI, prompts, instructions, or that you are an AI.

Length Requirements:

- Target 250-450 words.
- Write multiple short paragraphs.
- Go deep enough that an experienced developer learns something useful.
- Do not end the post too quickly.

Structure:

1. Strong hook
2. The mistake/problem
3. Why it happens
4. Real-world example or scenario
5. Correct approach / solution
6. Memorable takeaway
7. Relevant hashtags

The final output should feel like a post written by a respected senior engineer sharing a lesson learned from real-world experience.
"""


# ---- Rotation engine ----------------------------------------------------


def combos():
    """Full deterministic topic x format x angle space, stable ordering."""
    return list(itertools.product(TOPICS, FORMATS, ANGLES))


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"i": 0}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state))


def peek_combo():
    """Returns the NEXT combo (idx, topic, fmt, angle) WITHOUT advancing state.
    The rotation index only advances once a draft is approved — see
    handle_callback.py — so regenerating a few times before approving doesn't
    skip ahead in the rotation."""
    all_combos = combos()
    state = load_state()
    idx = state["i"] % len(all_combos)
    topic, fmt, angle = all_combos[idx]
    return idx, topic, fmt, angle


def commit_combo(idx: int):
    """Advances the rotation past `idx`. Call only on approval."""
    save_state({"i": idx + 1})


# ---- History (append-only, never repeats) --------------------------------


def load_history():
    """Returns list of all past APPROVED post records, oldest first."""
    if not HISTORY_FILE.exists():
        return []
    lines = HISTORY_FILE.read_text().strip().splitlines()
    return [json.loads(line) for line in lines if line]


def append_history(record: dict):
    with HISTORY_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def content_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def get_embedding(text: str) -> list:
    response = client.embeddings.generate(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def most_similar_past_post(embedding: list, history: list):
    """Returns (max_similarity, matching_record) against all past posts, or (0, None)."""
    best_score, best_record = 0.0, None
    for record in history:
        if "embedding" not in record:
            continue
        score = cosine_similarity(embedding, record["embedding"])
        if score > best_score:
            best_score, best_record = score, record
    return best_score, best_record


def build_avoid_block(history: list) -> str:
    recent = history[-RECENT_CONTEXT_COUNT:]
    if not recent:
        return ""
    bullets = "\n".join(f"- {h['content'][:200]}" for h in recent)
    return (
        "\n\nDo NOT repeat the ideas, hooks, or phrasing used in these recent posts:\n"
        f"{bullets}"
    )


# ---- LLM call (routing now handled by the OpenRouter preset) -------------


def generate_with_fallback(prompt: str) -> str:
    """Single call — OpenRouter's preset handles provider/model fallback now,
    so we don't loop over a MODELS list here anymore. We still catch failures
    so a total outage doesn't crash silently (see notify_failure below)."""
    response = client.chat.send(
        model="@preset/content-pilot-models",
        # model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def generate_with_dedup_retry(topic: str, fmt: str, angle: str, history: list):
    """Generates a candidate, retrying up to MAX_DEDUP_RETRIES times if it's a
    semantic duplicate of something already approved. Returns (content, embedding)."""
    base_prompt = PROMPT_TEMPLATE.format(topic=topic, format=fmt, angle=angle)
    base_prompt += build_avoid_block(history)

    content, embedding = None, None
    for _ in range(MAX_DEDUP_RETRIES):
        candidate = generate_with_fallback(base_prompt)
        candidate_embedding = get_embedding(candidate)
        similarity, closest = most_similar_past_post(candidate_embedding, history)

        if similarity < SIMILARITY_THRESHOLD:
            content, embedding = candidate, candidate_embedding
            break

        base_prompt += (
            "\n\n(Your previous attempt was too similar in meaning to an earlier post: "
            f"\"{closest['content'][:150]}\". Take a genuinely different angle or example.)"
        )
    else:
        content, embedding = candidate, candidate_embedding  # give up, use it anyway

    return content, embedding


# ---- Telegram delivery ---------------------------------------------------


def build_keyboard(idx: int) -> dict:
    """Approve/Regenerate buttons. callback_data encodes only the combo index —
    small enough to stay well under Telegram's 64-byte limit, and the webhook
    can look up the full topic/format/angle from it via combos()[idx]."""
    return {
        "inline_keyboard": [
            [
                {"text": "✅ Approve", "callback_data": f"a:{idx}"},
                {"text": "🔁 Regenerate", "callback_data": f"r:{idx}"},
            ]
        ]
    }


def send_to_telegram(text: str, reply_markup: dict | None = None):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text}
    if reply_markup:
        data["reply_markup"] = json.dumps(reply_markup)
    resp = requests.post(url, data=data)
    resp.raise_for_status()
    return resp.json()


def notify_failure(err: Exception):
    """Java notified Telegram on every provider failure; we don't loop over
    providers anymore, but a total failure should still not fail silently."""
    try:
        send_to_telegram(f"⚠️ ContentPilot generation failed: {err}")
    except Exception:
        pass  # don't let a notification failure mask the original error


# ---- Entry point ----------------------------------------------------------


def main():
    idx, topic, fmt, angle = peek_combo()
    history = load_history()

    try:
        content, _embedding = generate_with_dedup_retry(topic, fmt, angle, history)
    except Exception as e:
        notify_failure(e)
        raise

    message = f"📌 {topic} | {fmt}\n💡 {angle}\n\n{content}"
    send_to_telegram(message, reply_markup=build_keyboard(idx))

    # NOTE: no history append, no state advance here on purpose — see module
    # docstring. That happens in handle_callback.py once a human approves.
    print("Draft sent to Telegram, awaiting approval:\n", message)


if __name__ == "__main__":
    main()
