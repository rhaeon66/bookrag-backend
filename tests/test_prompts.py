from app.generation import prompts


def test_system_prompt_encodes_the_grounding_rules():
    text = prompts.build_system_prompt("In God's Path", "insufficient evidence message")
    assert "In God's Path" in text
    assert "ONLY" in text
    assert "Do not invent" in text
    assert "Do not use outside knowledge" in text
    assert "insufficient evidence message" in text


def test_user_prompt_embeds_question_and_context():
    prompt = prompts.build_user_prompt("Was Muhammad driven out of Mecca?", "[Source 1]\nsome context text")
    assert "Was Muhammad driven out of Mecca?" in prompt
    assert "[Source 1]" in prompt
    assert "some context text" in prompt


def test_query_rewrite_prompt_includes_history_and_latest_message():
    prompt = prompts.build_query_rewrite_prompt("user: what happened?\nassistant: X happened.", "why did it happen?")
    assert "what happened?" in prompt
    assert "why did it happen?" in prompt
