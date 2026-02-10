import json
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


def chat_completion(messages, max_tokens, temperature, top_p, use_local, hf_token=None):
    """Send messages to either the API or local model and return the response text."""
    if use_local:
        start = time.time()
        outputs = _get_local_pipe()(
            messages,
            max_new_tokens=max_tokens,
            do_sample=True, # To allow temperature and top_p to work
            temperature=temperature,
            top_p=top_p,
        )
        elapsed = time.time() - start
        print(f"[LOCAL] Response generated in {elapsed:.2f}s")
        content = outputs[0]["generated_text"][-1]["content"]
        if not content:
            print("[LOCAL] Warning: model returned empty content")
        return content or ""

    start = time.time()
    client = InferenceClient(model=API_MODEL, token=hf_token.token)
    response = client.chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        tool_choice="none", # To avoid tool-call format errors
    )
    elapsed = time.time() - start
    print(f"[API] Response generated in {elapsed:.2f}s")
    content = response.choices[0].message.content
    if not content:
        print("[API] Warning: model returned empty content")
    return content or ""


def split_response(raw_text):
    """ Separate response text into chat text and memory JSON data. """
    try:
        # Find index between text and json. Either by delimiter or attempt to find bracket.
        ds = raw_text.find(MEMORY_START)
        if ds != -1:
            json_start = ds + len(MEMORY_START)
            chat_end = ds
        else:
            print("[PARSE] Missing start delimiter, attempting bracket fallback")
            wm = raw_text.index('"write_memory"')
            json_start = raw_text.rindex("{", 0, wm)
            chat_end = json_start

        # Find index of the end of json. Either by delimiter or attempt to find bracket.
        de = raw_text.find(MEMORY_END, json_start)
        if de == -1:
            print("[PARSE] Missing end delimiter, using last '}' as fallback")
        json_end = de if de != -1 else raw_text.rindex("}") + 1

        chat_text = raw_text[:chat_end].strip()
        memory_data = json.loads(raw_text[json_start:json_end].strip())

        return chat_text, memory_data
    except (ValueError, json.JSONDecodeError, TypeError) as e:
        print(f"[PARSE] Failed to extract memory JSON: {e}")
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
    min_importance,
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

    # Inject memory context for items at or above the importance threshold
    relevant = [m for m in current_memory if m.get("importance", 0) >= int(min_importance)]
    if relevant:
        lines = [f"- [{m['label']}] {m['note']} (importance: {m['importance']})" for m in relevant]
        messages.append({"role": "system", "content": "Known facts about the user:\n" + "\n".join(lines)})

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

    # Persist new memory items
    current_memory.extend(new_items)
    store = dict(memory_store)
    user_data = dict(store.get(user_id, {}))
    user_data[personality.lower()] = current_memory
    store[user_id] = user_data

    return chat_text, store, user_id, current_memory
