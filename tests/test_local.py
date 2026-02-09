import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from response_manager import respond


def test_local_response():
    result = respond(
        message="Hi",
        history=[],
        personality="Teacher",
        system_message="You are a helpful assistant.",
        max_tokens=8,
        temperature=0.7,
        top_p=0.9,
        memory_store={},
        session_id=None,
        use_local=True,
        min_importance=1,
        recent_turns=5,
    )
    assert isinstance(result[0], str)
    assert len(result[0]) > 0
