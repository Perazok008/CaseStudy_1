import json
import uuid
import gradio as gr
from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError
from config import API_MODEL, LOCAL_MODEL

MEMORY_START = "<<<MEMORY_DELTA_JSON>>>"
MEMORY_END = "<<<END_MEMORY_DELTA_JSON>>>"

_local_pipe = None


def _get_local_pipe():
    """Lazy-load the local text-generation pipeline on first use."""
    global _local_pipe
    if _local_pipe is None:
        from transformers import pipeline
        _local_pipe = pipeline("text-generation", model=LOCAL_MODEL)
    return _local_pipe


def chat_completion(messages, max_tokens, temperature, top_p, use_local, hf_token=None):
    """Send messages to either the API or local model and return the assistant's response text."""
    if use_local:
        # Normalize multimodal content (lists) to plain strings for the local pipeline
        normalized = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            normalized.append({"role": msg["role"], "content": content})

        outputs = _get_local_pipe()(
            normalized,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        return outputs[0]["generated_text"][-1]["content"] or ""

    client = InferenceClient(model=API_MODEL, token=hf_token.token)
    try:
        response = client.chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content or ""
    except BadRequestError as e:
        # Some models occasionally wrap output in a tool-call format, causing the
        # API to reject it. The actual content is in the error's failed_generation field.
        try:
            body = e.response.json()
            failed = body.get("failed_generation", "")
            marker = '"arguments": '
            idx = failed.find(marker)
            if idx != -1:
                content = failed[idx + len(marker):]
                last_brace = content.rfind("}")
                if last_brace != -1:
                    content = content[:last_brace]
                return content.strip()
        except Exception:
            pass
        raise e


def _find_memory_json(text):
    """Locate the memory JSON object in text by finding "write_memory" and
    brace-counting outward. Returns (start_index, parsed_dict) or (-1, None)."""
    wm = text.find('"write_memory"')
    if wm == -1:
        return -1, None
    start = text.rfind("{", 0, wm)
    if start == -1:
        return -1, None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return start, json.loads(text[start : i + 1])
                except (json.JSONDecodeError, TypeError):
                    return -1, None
    return -1, None


def parse_response(raw_text):
    """Split model output into (chat_text, memory_items).

    Tries sentinel delimiters first, then falls back to locating
    the JSON block by its "write_memory" key.
    """
    # --- Split chat text from memory data ---
    data = None
    if MEMORY_START in raw_text:
        chat_text, remainder = raw_text.split(MEMORY_START, 1)
        json_text = remainder.split(MEMORY_END, 1)[0].strip()
        chat_text = chat_text.strip()
        try:
            data = json.loads(json_text)
        except (json.JSONDecodeError, TypeError):
            pass
    else:
        idx, data = _find_memory_json(raw_text)
        chat_text = raw_text[:idx].strip() if idx != -1 and data else raw_text.strip()

    # --- Extract validated memory items ---
    if not isinstance(data, dict) or not data.get("write_memory"):
        return chat_text, []

    items = []
    for item in data.get("items", []):
        if not isinstance(item, dict):
            continue
        note = str(item.get("note", "")).strip()
        if note:
            try:
                importance = int(item.get("importance", 1))
            except (TypeError, ValueError):
                importance = 1
            items.append({
                "label": str(item.get("label", "")).strip(),
                "note": note,
                "importance": importance,
            })
    return chat_text, items


def get_personality_memory(memory_store, user_id, personality):
    """Read the memory list for a given user + personality from the store."""
    return list(memory_store.get(user_id, {}).get(personality.lower(), []))


def respond(
    message,
    history,
    personality,
    system_message,
    max_tokens,
    temperature,
    top_p,
    memory_store,
    session_id,
    use_local,
    min_importance,
    recent_turns,
    hf_token: gr.OAuthToken = None,
    profile: gr.OAuthProfile = None,
):
    # Resolve user ID: OAuth profile > existing session > new UUID
    if profile is not None and getattr(profile, "username", None):
        user_id = profile.username
    elif session_id:
        user_id = session_id
    else:
        user_id = str(uuid.uuid4())

    current_memory = get_personality_memory(memory_store, user_id, personality)

    if not use_local and (hf_token is None or not getattr(hf_token, "token", None)):
        return (
            "Please log in with your Hugging Face account to use the API model.",
            memory_store, user_id, current_memory,
        )

    # Build message list
    messages = [{"role": "system", "content": system_message}]

    # Inject memory context for items at or above the importance threshold
    min_importance = int(min_importance)
    relevant = [m for m in current_memory if m.get("importance", 0) >= min_importance]
    if relevant:
        lines = [f"- [{m['label']}] {m['note']} (importance: {m['importance']})" for m in relevant]
        messages.append({"role": "system", "content": "Known facts about the user:\n" + "\n".join(lines)})

    # Append recent conversation history
    max_msgs = int(recent_turns) * 2
    messages.extend(history[-max_msgs:] if len(history) > max_msgs else list(history))

    messages.append({"role": "user", "content": message})

    # Get and parse model response
    raw = chat_completion(messages, max_tokens, temperature, top_p, use_local, hf_token)
    chat_text, new_items = parse_response(raw)

    # Persist new memory items
    current_memory.extend(new_items)
    store = dict(memory_store)
    user_data = dict(store.get(user_id, {}))
    user_data[personality.lower()] = current_memory
    store[user_id] = user_data

    return chat_text, store, user_id, current_memory
