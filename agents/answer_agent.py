# agents/answer_agent.py
# ─────────────────────────────────────────────────────────────
# Responsibility: Generate a document-grounded answer to a student's
# question using a locally running Ollama LLM.
#
# How it works:
#   1. Receive the top-k chunks retrieved by the RAG pipeline.
#   2. Build a prompt that includes the chunks as "context".
#   3. Call the local Ollama model and return its response.
#
# The LLM never sees the full document — only the relevant chunks.
# This keeps answers focused and reduces hallucination.
#
# Depends on: Ollama running locally with qwen2.5:7b pulled.
#   → Start Ollama:   ollama serve
#   → Pull model:     ollama pull qwen2.5:7b
# ─────────────────────────────────────────────────────────────

import os

try:
    from langchain_ollama import OllamaLLM
except ImportError:  # pragma: no cover - makes unit tests importable without Ollama deps
    class OllamaLLM:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, prompt: str) -> str:
            raise RuntimeError("langchain_ollama is not installed")

# In Docker, OLLAMA_HOST is set to "http://ollama:11434" via docker-compose.
# When running locally without Docker it defaults to localhost.
_ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

llm = OllamaLLM(model="qwen2.5:7b", base_url=_ollama_host)

SYSTEM_PROMPT_TR = (
    "Sen akademik bir asistansın. Yalnızca aşağıdaki bağlamı kullanarak soruyu yanıtla. "
    "Yanıtın kısa, açık ve doğrudan olsun. Gerekirse madde işaretleri kullan. "
    'Bağlamda yeterli bilgi yoksa "Bu soruyu yanıtlamak için yeterli bilgiye sahip değilim." de. '
    "Uydurma bilgi ekleme ve bağlam dışına çıkma."
)

SYSTEM_PROMPT_EN = (
    "You are a helpful academic assistant. Answer the student's question using ONLY the context provided below. "
    "Keep the answer concise, clear, and well-structured. Use bullet points when helpful. "
    'If the answer is not in the context, say "I don\'t have enough information to answer that." '
    "Do not invent facts or rely on outside knowledge."
)


def _detect_language(text: str) -> str:
    """Return 'tr' for Turkish text and 'en' otherwise."""
    turkish_chars = set("çğıöşüÇĞİÖŞÜ")
    return "tr" if any(char in turkish_chars for char in text) else "en"


def _chunk_text(chunk: object) -> str:
    """Normalize chunks coming from either strings or metadata dictionaries."""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        return str(chunk.get("text", ""))
    return str(chunk)


def _format_chunk(chunk: object, index: int) -> str:
    """Render one chunk with lightweight source metadata when available."""
    if isinstance(chunk, dict):
        text = str(chunk.get("text", "")).strip()
        source_file = chunk.get("source_file", "unknown")
        page = chunk.get("page", 0)
        return f"[{index}] Source: {source_file}, page {page}\n{text}".strip()

    text = _chunk_text(chunk).strip()
    return f"[{index}] {text}"


def _build_prompt(question: str, context: str, lang: str) -> str:
    system = SYSTEM_PROMPT_TR if lang == "tr" else SYSTEM_PROMPT_EN
    return f"""{system}

---
CONTEXT:
{context}
---

QUESTION: {question}

ANSWER:"""


def _format_instruction(format: str) -> str:
    """Return a short instruction for the requested answer format."""
    if format == "bullets":
        return "Yanıtı madde işaretleriyle ver."
    if format == "paragraph":
        return "Yanıtı tek, akıcı bir paragraf olarak ver."
    return "Yanıt biçimini sorunun doğasına göre seç."


def generate_answer(question: str, chunks: list[dict], format: str = "auto") -> str:
    """
    Generate an answer grounded in the retrieved document chunks.

    Args:
        question: The student's question.
        chunks:   List of chunk dicts with keys: text, source_file, page, chunk_index.
        format:   Answer style preference: "auto", "bullets", or "paragraph".

    Returns:
        A plain-text answer string from the LLM.
    """
    normalized_chunks = [_format_chunk(chunk, index) for index, chunk in enumerate(chunks, start=1)]
    context = "\n\n".join(chunk for chunk in normalized_chunks if chunk.strip())

    if not context.strip():
        return (
            "Bu soruyu yanıtlamak için yeterli bilgiye sahip değilim."
            if _detect_language(question) == "tr"
            else "I don't have enough information to answer that."
        )

    lang = _detect_language(question)
    prompt = _build_prompt(question, context, lang)
    prompt = f"{prompt}\n\nSTYLE: {_format_instruction(format)}"
    return llm.invoke(prompt)
