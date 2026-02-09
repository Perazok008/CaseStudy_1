import gradio as gr
from config import PERSONALITIES
from response_manager import respond, get_personality_memory
from components.SettingsSidebar import SettingsSidebar


def update_profile(personality, memory_store, session_id):
    """Switch personality: clear chat, display the new personality's stored memory."""
    system_prompt = PERSONALITIES[personality]["system_prompt"]
    memory_items = get_personality_memory(memory_store, session_id, personality)
    return system_prompt, [], [], memory_items, None


# Build the Gradio UI
with gr.Blocks() as demo:
    # Shared state
    memory_store = gr.State({})
    session_id = gr.State(None)

    # Left sidebar: Settings
    settings = SettingsSidebar()

    # Right sidebar: Memory
    with gr.Column(scale=0):
        gr.Markdown("## Memory")
        memory_display = gr.JSON(value=[], label="Memory")

    # Main: Chat
    with gr.Column(scale=1):
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

    settings["personality_dd"].change(
        fn=update_profile,
        inputs=[settings["personality_dd"], memory_store, session_id],
        outputs=[
            settings["system_prompt"],
            chatbot.chatbot,
            chatbot.chatbot_state,
            memory_display,
            chatbot.saved_input,
        ],
    )

if __name__ == "__main__":
    demo.launch()
