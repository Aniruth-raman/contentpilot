"""
Triggered by a GitHub Actions workflow `on: repository_dispatch` when the
Cloudflare Worker forwards a Telegram button tap (Approve/Regenerate).

Expected environment variables (set from client_payload in the workflow):
    EVENT        "tg-approve" or "tg-regenerate"
    COMBO_IDX    the rotation index encoded in the button's callback_data
    CHAT_ID      Telegram chat id (from the original message)
    MESSAGE_ID   Telegram message id to edit
    TEXT         full text of the original Telegram message (approve only)

Run:
    python handle_callback.py
"""

import os
import json
import requests

from content_pilot import (
    combos,
    commit_combo,
    load_history,
    append_history,
    content_hash,
    get_embedding,
    generate_with_fallback,
    build_avoid_block,
    build_keyboard,
    PROMPT_TEMPLATE,
)


def edit_telegram_message(chat_id: str, message_id: str, text: str, reply_markup=None):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    url = f"https://api.telegram.org/bot{token}/editMessageText"
    data = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if reply_markup:
        data["reply_markup"] = json.dumps(reply_markup)
    else:
        # Explicitly clear the keyboard once a post is approved.
        data["reply_markup"] = json.dumps({"inline_keyboard": []})
    resp = requests.post(url, data=data)
    resp.raise_for_status()


def extract_content(message_text: str) -> str:
    """Message body is "📌 {topic} | {fmt}\\n💡 {angle}\\n\\n{content}" — content
    is everything after the first blank line."""
    return message_text.split("\n\n", 1)[1]


def handle_approve(idx: int, chat_id: str, message_id: str, message_text: str):
    content = extract_content(message_text)
    # Recompute the embedding rather than trying to pass it through
    # callback_data (way too large) — one extra embedding call, only on approval.
    embedding = get_embedding(content)
    topic, fmt, angle = combos()[idx]

    record = {
        "topic": topic,
        "format": fmt,
        "angle": angle,
        "content": content,
        "hash": content_hash(content),
        "embedding": embedding,
    }
    append_history(record)
    commit_combo(idx)  # rotation index only advances now, on approval

    edit_telegram_message(chat_id, message_id, message_text + "\n\n✅ Approved")


def handle_regenerate(idx: int, chat_id: str, message_id: str):
    topic, fmt, angle = combos()[idx]
    history = load_history()

    # Skip the embedding-similarity retry loop here — a human tapping
    # Regenerate is already the dedup filter for this draft.
    prompt = PROMPT_TEMPLATE.format(topic=topic, format=fmt, angle=angle)
    prompt += build_avoid_block(history)
    content = generate_with_fallback(prompt)

    message = f"📌 {topic} | {fmt}\n💡 {angle}\n\n{content}"
    edit_telegram_message(chat_id, message_id, message, reply_markup=build_keyboard(idx))


def main():
    event = os.environ["EVENT"]
    idx = int(os.environ["COMBO_IDX"])
    chat_id = os.environ["CHAT_ID"]
    message_id = os.environ["MESSAGE_ID"]

    if event == "tg-approve":
        handle_approve(idx, chat_id, message_id, os.environ["TEXT"])
    elif event == "tg-regenerate":
        handle_regenerate(idx, chat_id, message_id)
    else:
        raise ValueError(f"Unknown event: {event}")


if __name__ == "__main__":
    main()