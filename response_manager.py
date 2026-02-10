# gradio system honestly seems very fragile, it's documentation uses a lot of depreciated elements, has inconsistent types adn formats and seems to be version fragile which is driving me crazy. I added checks a bit of noramlization but I will wiher if I try to cover all edge cases.
import json
import re
import time
import uuid
import gradio as gr
from huggingface_hub import InferenceClient
from config import API_MODEL, LOCAL_MODEL, MEMORY_START, MEMORY_END

_local_pipe = None

def _get_local_pipe():
    """ Load the local pipeline on first use. Will permanently occupy RAM for the duration of the program. """
    global _local_pipe
    if _local_pipe is None:
        from transformers import pipeline
        _local_pipe = pipeline("text-generation", model=LOCAL_MODEL)
    return _local_pipe


def _normalize_messages(messages):
    """Ensure every message content is a plain string (Gradio can pass list-of-dicts for multimodal)."""
    normalized = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        normalized.append({"role": msg["role"], "content": str(content)})
    return normalized


def chat_completion(messages, max_tokens, temperature, top_p, use_local, hf_token=None):
    """Send messages to either the API or local model and return the response text."""
    if use_local:
        local_messages = _normalize_messages(messages)
        start = time.time()
        outputs = _get_local_pipe()(
            local_messages,
            max_new_tokens=max_tokens,
            do_sample=True, # To allow temperature and top_p to work
            temperature=temperature,
            top_p=top_p,
        )
        elapsed = time.time() - start
        print(f"[LOCAL] Response generated in {elapsed:.2f}s")
        content = outputs[0]["generated_text"][-1]["content"]
        if not content:
            print("[LOCAL] Empty response, retrying once...")
            start = time.time()
            outputs = _get_local_pipe()(
                local_messages,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            elapsed = time.time() - start
            print(f"[LOCAL] Retry generated in {elapsed:.2f}s")
            content = outputs[0]["generated_text"][-1]["content"]
            if not content:
                print("[LOCAL] Retry also returned empty content")
        return content or ""

    client = InferenceClient(model=API_MODEL, token=hf_token.token)
    start = time.time()
    response = client.chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        tool_choice="none", # To avoid tool-call format errors
    )
    elapsed = time.time() - start
    print(f"[API] Response generated in {elapsed:.2f}s (model: {API_MODEL})")
    content = response.choices[0].message.content
    if not content:
        print("[API] Empty response, retrying once...")
        start = time.time()
        response = client.chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tool_choice="none",
        )
        elapsed = time.time() - start
        print(f"[API] Retry generated in {elapsed:.2f}s (model: {API_MODEL})")
        content = response.choices[0].message.content
        if not content:
            print("[API] Retry also returned empty content")
    return content or ""


_MEMORY_RE = re.compile(
    rf"(?s)(.*?){re.escape(MEMORY_START)}\s*(.*?)\s*{re.escape(MEMORY_END)}",
)

# Fallback: find a JSON object containing "write_memory" when delimiters are missing
_FALLBACK_RE = re.compile(r'(?s)\{[^{}]*"write_memory"[^{}]*\{.*?\}[^{}]*\}')


def split_response(raw_text):
    """ Separate response text into chat text and memory JSON data. """
    if not raw_text:
        print("[PARSE] Received empty response text")
        return "", {}

    # Primary: use delimiters
    m = _MEMORY_RE.search(raw_text)
    if m:
        chat_text = m.group(1).strip()
        json_text = m.group(2).strip()
        try:
            memory_data = json.loads(json_text)
            return chat_text, memory_data
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[PARSE] JSON decode error between delimiters: {e}")
            return chat_text, {}

    # Fallback: try to find a JSON object with "write_memory" key (for weaker models)
    # A bit of anightmare, but it's the best we can do to stabilize the local model responses.
    print("[PARSE] No memory delimiters found, trying fallback JSON extraction")
    print(f"[PARSE] Raw response (last 300 chars): ...{raw_text[-300:]}")

    wm_idx = raw_text.find('"write_memory"')
    if wm_idx != -1:
        # Walk backwards to find the opening brace
        start = raw_text.rfind("{", 0, wm_idx)
        if start != -1:
            # Brace-count forward to find matching close
            depth = 0
            for i in range(start, len(raw_text)):
                if raw_text[i] == "{":
                    depth += 1
                elif raw_text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        json_text = raw_text[start:i + 1]
                        try:
                            memory_data = json.loads(json_text)
                            chat_text = raw_text[:start].strip()
                            print(f"[PARSE] Fallback extracted JSON successfully")
                            return chat_text, memory_data
                        except (json.JSONDecodeError, TypeError) as e:
                            print(f"[PARSE] Fallback JSON decode error: {e}")
                            break
            else:
                print("[PARSE] Fallback: unbalanced braces, could not extract JSON")

    print("[PARSE] No memory JSON found in response")
    return raw_text.strip(), {}


def extract_memory_items(data):
    """ Validate and extract memory items from parsed JSON. """
    if not isinstance(data, dict):
        if data:
            print(f"[MEMORY] Expected dict but got {type(data).__name__}, skipping memory extraction")
        return []
    items = []
    for item in data.get("items", []):
        if not isinstance(item, dict):
            print(f"[MEMORY] Skipping non-dict item: {item!r}")
            continue
        note = str(item.get("note", "")).strip()
        if not note:
            print(f"[MEMORY] Skipping item with empty note (label: {item.get('label', '?')})")
            continue
        try:
            importance = int(item.get("importance", 1))
        except (TypeError, ValueError):
            print(f"[MEMORY] Invalid importance '{item.get('importance')}' for '{note}', defaulting to 1")
            importance = 1
        items.append({
            "label": str(item.get("label", "")).strip(),
            "note": note,
            "importance": importance,
        })
    if items:
        print(f"[MEMORY] Extracted {len(items)} new memory item(s)")
    return items


def get_personality_memory(memory_store, user_id, personality):
    """ Read the memory list for a given user + personality from the store. """
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
    min_recall_importance,
    min_save_importance,
    recent_turns,
    hf_token: gr.OAuthToken = None,
    profile: gr.OAuthProfile = None,
):
    """ Respond to a user message and return the chat text, memory store, user_id, and current memory. """
    # Resolve user_id for memory storage
    if profile is not None and getattr(profile, "username", None):
        user_id = profile.username  # OAuth profile username
    elif session_id:
        user_id = session_id  # Existing session ID
    else:
        user_id = str(uuid.uuid4())  # New user UUID

    current_memory = get_personality_memory(memory_store, user_id, personality)

    # If using API model and not logged in, return error message
    if not use_local and (hf_token is None or not getattr(hf_token, "token", None)):
        print("[AUTH] API request rejected: user not logged in")
        return ("Please log in to use the API model.", memory_store, user_id, current_memory,)

    # Build messages list
    messages = [{"role": "system", "content": system_message}]

    # Inject memory context for items at or above the recall importance threshold
    recall_threshold = int(min_recall_importance)
    # Including importance score in prompt messes up responses, so sort instead
    relevant = sorted(
        [m for m in current_memory if m.get("importance", 0) >= recall_threshold],
        key=lambda m: m.get("importance", 0),
        reverse=True,
    )
    if relevant:
        print(f"[MEMORY] Injecting {len(relevant)} memory item(s) (importance >= {recall_threshold})")
        lines = [f"- [{m['label']}] {m['note']}" for m in relevant]
        # More predictable when sent from user
        messages.append({"role": "user", "content": "Known facts about me:\n" + "\n".join(lines)})

    # Append recent conversation history
    max_msgs = int(recent_turns) * 2
    messages.extend(history[-max_msgs:])

    # Append new user message
    messages.append({"role": "user", "content": message})

    # Get and parse model response
    try:
        raw = chat_completion(messages, max_tokens, temperature, top_p, use_local, hf_token)
    except Exception as e:
        print(f"[ERROR] Model request failed: {e}")
        return ("Error: could not get a response from the model.", memory_store, user_id, current_memory)

    chat_text, data = split_response(raw)
    new_items = extract_memory_items(data)

    # Ensure we never send an empty message to the chat
    if not chat_text:
        print("[WARN] Model returned empty chat text, sending fallback")
        chat_text = "Error: the model returned an empty response. Please try again."

    # Filter new items by save importance threshold before persisting
    save_threshold = int(min_save_importance)
    saved_items = [item for item in new_items if item["importance"] >= save_threshold]
    if len(saved_items) < len(new_items):
        dropped = len(new_items) - len(saved_items)
        print(f"[MEMORY] Dropped {dropped} item(s) below save threshold (importance < {save_threshold})")

    # Persist new memory items
    current_memory.extend(saved_items)
    store = dict(memory_store)
    user_data = dict(store.get(user_id, {}))
    user_data[personality.lower()] = current_memory
    store[user_id] = user_data

    return chat_text, store, user_id, current_memory
