import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from response_manager import respond


class Token:
    def __init__(self, token): self.token = token


def test_api_response():
    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token, "HF_TOKEN not set in environment"

    result = respond(
        message="Hi",
        history=[],
        personality="Teacher",
        system_message="You are a helpful assistant.",
        max_tokens=8,
        temperature=0.2,
        top_p=0.9,
        memory_store={},
        session_id=None,
        use_local=False,
        min_importance=1,
        recent_turns=5,
        hf_token=Token(hf_token),
    )
    assert isinstance(result[0], str)
    assert "please log in" not in result[0].lower()
