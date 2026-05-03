# tests/test_answer.py

from unittest.mock import patch

from agents.answer_agent import _detect_language, generate_answer


def test_language_detection_turkish():
    assert _detect_language("Bunu açıklar mısın?") == "tr"


def test_language_detection_english():
    assert _detect_language("What is machine learning?") == "en"


def test_generate_answer_returns_string():
    with patch("agents.answer_agent.llm") as mock_llm:
        mock_llm.invoke.return_value = "Machine learning is..."
        result = generate_answer("What is ML?", ["ML is a field of AI."])

    assert isinstance(result, str)
    assert result


def test_generate_answer_uses_context():
    with patch("agents.answer_agent.llm") as mock_llm:
        mock_llm.invoke.return_value = "mocked"
        generate_answer("test?", ["chunk1", "chunk2"])

        prompt = mock_llm.invoke.call_args[0][0]

    assert "chunk1" in prompt
    assert "chunk2" in prompt
    assert "QUESTION: test?" in prompt


def test_generate_answer_formats_metadata_chunks():
    with patch("agents.answer_agent.llm") as mock_llm:
        mock_llm.invoke.return_value = "mocked"
        generate_answer(
            "What is this?",
            [
                {"text": "Chunk text", "source_file": "doc.pdf", "page": 3, "chunk_index": 1},
            ],
        )

        prompt = mock_llm.invoke.call_args[0][0]

    assert "Source: doc.pdf, page 3" in prompt
    assert "Chunk text" in prompt


def test_generate_answer_supports_bullet_format():
    with patch("agents.answer_agent.llm") as mock_llm:
        mock_llm.invoke.return_value = "mocked"
        generate_answer("What is ML?", ["ML is a field of AI."], format="bullets")

        prompt = mock_llm.invoke.call_args[0][0]

    assert "STYLE: Yanıtı madde işaretleriyle ver." in prompt


def test_generate_answer_empty_chunks_returns_fallback():
    with patch("agents.answer_agent.llm") as mock_llm:
        result = generate_answer("What is X?", [])

    mock_llm.invoke.assert_not_called()
    assert result == "I don't have enough information to answer that."
