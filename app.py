import gradio as gr
from config import PERSONALITIES
from response_manager import respond, get_personality_memory
from components.SettingsSidebar import SettingsSidebar

DEFAULT_PERSONALITY = "Teacher"


def personality_html(name):
    """Return banner HTML + a <style> tag that sets the accent CSS variable."""
    s = PERSONALITIES[name]["style"]
    return (
        f'<style>:root {{ --accent: {s["accent"]}; --accent-tint: {s["accent"]}18; }}</style>'
        f'<div style="padding:10px 14px; border-radius:8px; font-weight:600;'
        f" background:{s['accent']}15; color:{s['accent']};"
        f' border-left:4px solid {s["accent"]}; display:flex; align-items:center;">'
        f'<span style="font-size:1.3em; margin-right:8px;">{s["emoji"]}</span>{name}</div>'
    )


def update_profile(personality, memory_store, session_id):
    """Switch personality: clear chat, update banner and memory display."""
    p = PERSONALITIES[personality]
    memory_items = get_personality_memory(memory_store, session_id, personality)
    return p["system_prompt"], [], [], memory_items, None, personality_html(personality)


CSS = """
.chat-col { border-top: 3px solid var(--accent, #2563EB); border-radius: 8px; padding-top: 8px; }
.chat-col .bot .message-bubble { border-color: var(--accent, #2563EB) !important; }
.memory-accordion { border-color: var(--accent, #2563EB) !important; }
"""

with gr.Blocks(css=CSS) as demo:
    memory_store = gr.State({})
    session_id = gr.State(None)

    with gr.Row():
        settings = SettingsSidebar()

        with gr.Column(scale=1, elem_classes=["chat-col"]):
            banner = gr.HTML(value=personality_html(DEFAULT_PERSONALITY))

            memory_display = gr.JSON(value=[], label="Stored memory items", render=False)

            chatbot = gr.ChatInterface(
                respond,
                additional_inputs=[
                    settings["personality_dd"],
                    settings["system_prompt"],
                    settings["max_tokens"],
                    settings["temperature"],
                    settings["top_p"],
                    memory_store,
                    session_id,
                    settings["local_toggle"],
                    settings["min_memory_importance"],
                    settings["recent_turns"],
                ],
                additional_outputs=[memory_store, session_id, memory_display],
            )
            with gr.Accordion("Memory", open=True, elem_classes=["memory-accordion"]):
                memory_display.render()

    settings["personality_dd"].change(
        fn=update_profile,
        inputs=[settings["personality_dd"], memory_store, session_id],
        outputs=[
            settings["system_prompt"],
            chatbot.chatbot,
            chatbot.chatbot_state,
            memory_display,
            chatbot.saved_input,
            banner,
        ],
    )

if __name__ == "__main__":
    demo.launch()
