import gradio as gr
from config import PERSONALITIES, PERSONALITY_CHOICES

def SettingsSidebar():
    with gr.Column(scale=0):
        gr.Markdown("## Settings")
        gr.LoginButton(size="sm")

        personality_dd = gr.Dropdown(
            choices=PERSONALITY_CHOICES,
            value="Teacher",
            label="Personality",
            info="Choose how the assistant behaves.",
        )

        local_toggle = gr.Checkbox(
            label="Use local model",
            value=False,
            elem_id="local-toggle",
            info="Toggle between local and API inference.",
        )

        system_prompt = gr.Textbox(
            label="System prompt",
            value=PERSONALITIES["Teacher"]["system_prompt"],
            lines=4,
            max_lines=8,
            interactive=False,
        )

        max_tokens = gr.Slider(
            minimum=1,
            maximum=2048,
            value=512,
            step=1,
            label="Max new tokens",
        )
        temperature = gr.Slider(
            minimum=0.1,
            maximum=4.0,
            value=0.2,
            step=0.1,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            step=0.05,
            label="Top-p",
        )
        min_memory_importance = gr.Slider(
            minimum=1,
            maximum=5,
            value=2,
            step=1,
            label="Min memory importance",
            info="Only include memory items at or above this level.",
        )
        recent_turns = gr.Slider(
            minimum=1,
            maximum=20,
            value=5,
            step=1,
            label="Recent turns",
            info="Number of recent conversation turns to send.",
        )
    return {
        "personality_dd": personality_dd,
        "local_toggle": local_toggle,
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "min_memory_importance": min_memory_importance,
        "recent_turns": recent_turns,
    }
