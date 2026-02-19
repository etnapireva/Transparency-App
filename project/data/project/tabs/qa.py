# Q&A tab
import html
import streamlit as st
from config import GROQ_API_KEY, GROQ_MODEL, MAX_QA_DOCS, MAX_CHARS_CONTEXT
from utils.ollama_integration import build_qa_context


def render(df, init_vector_store):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_query" not in st.session_state:
        st.session_state.selected_query = ""
    if "chat_initialized" not in st.session_state:
        st.session_state.chat_initialized = False
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "pending_in_progress" not in st.session_state:
        st.session_state.pending_in_progress = False

    if not st.session_state.chat_initialized:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Përshëndetje, jam DIELLA AI. Mund të më pyesësh për përmbajtjen, tonin dhe temat e deklaratave të Diellës dhe aktorëve tjerë politikë në këtë sistem.",
        })
        st.session_state.chat_initialized = True

    _render_chat_ui()
    _handle_chat_actions(df, init_vector_store)


def _render_chat_ui():
    st.markdown(_CHAT_CSS, unsafe_allow_html=True)
    st.subheader("Bisedo me DIELLA AI")
    with st.expander("Si funksionon?", expanded=False):
        st.markdown("Pyetja kërkohet në korpusin e deklaratave përmes kërkimit vektorial (SentenceTransformer, FAISS). Deklaratat më të ngjashme dërgohen te Groq për përgjigje (RAG).")
    if not GROQ_API_KEY or not str(GROQ_API_KEY).strip():
        st.warning("Q&A nuk është i konfiguruar. Vendosni GROQ_API_KEY në .env.")
        return
    st.markdown("<div class=\"chat-container\">", unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        escaped = html.escape(str(msg["content"])).replace("\n", "<br>")
        role_class = "user" if msg["role"] == "user" else "ai"
        label = "Ti" if msg["role"] == "user" else "DIELLA AI"
        st.markdown(f"<div class=\"chat-message {role_class}\"><div><div class=\"chat-label\">{label}</div><div class=\"chat-bubble {role_class}\">{escaped}</div></div></div>", unsafe_allow_html=True)
    show_sugg = (len(st.session_state.chat_history) == 1 or (st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant")) and not st.session_state.get("pending_in_progress", False)
    if show_sugg:
        st.markdown("<div class=\"chat-suggestions-container\">", unsafe_allow_html=True)
        st.markdown("<div style=\"color:#9ca3af;font-size:0.875rem;\">Pyetje te shpeshta:</div>", unsafe_allow_html=True)
        for idx, s in enumerate(_SUGGESTIONS):
            if st.button(s[:50] + "..." if len(s) > 50 else s, key=f"chat_suggest_{idx}", use_container_width=False):
                st.session_state.pending_query = s
                st.session_state.pending_in_progress = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class=\"chat-input-container\">", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        query_shqip = st.text_input("Shkruaj pyetjen:", value=st.session_state.get("selected_query", ""), key="qa_input_chat", placeholder="Shkruaj pyetjen ose kliko sugjerim...")
        submit_button = st.form_submit_button("Dërgo", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("Pastro historine", use_container_width=True, key="clear_chat"):
        st.session_state.chat_history = []
        st.session_state.selected_query = ""
        st.session_state.chat_initialized = False
        st.rerun()
    if st.session_state.get("selected_query"):
        st.session_state.selected_query = ""
    if submit_button and query_shqip:
        st.session_state.pending_query = query_shqip
        st.session_state.pending_in_progress = False
        st.rerun()


def _handle_chat_actions(df, init_vector_store):
    if st.session_state.get("pending_query") and not st.session_state.get("pending_in_progress", False):
        q = st.session_state.pending_query
        if not any(m.get("role") == "user" and m.get("content") == q for m in st.session_state.chat_history):
            st.session_state.chat_history.append({"role": "user", "content": q})
            st.session_state.chat_history.append({"role": "assistant", "content": "DIELLA AI po mendon..."})
        st.session_state.pending_in_progress = True
        st.rerun()
    if not st.session_state.get("pending_query") or not st.session_state.get("pending_in_progress", False):
        return
    query_to_process = st.session_state.pending_query
    _model, _index = st.session_state.get("qa_model"), st.session_state.get("qa_index")
    if _model is None or _index is None:
        with st.spinner("Duke ngarkuar Q&A..."):
            _model, _index = init_vector_store(df)
            st.session_state["qa_model"] = _model
            st.session_state["qa_index"] = _index
    model, index = _model, _index
    with st.spinner("Po kerkoj..."):
        if model is None or index is None:
            err = "Baza vektoriale nuk eshte gati."
            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
                st.session_state.chat_history[-1]["content"] = err
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": err})
            st.error(err)
        else:
            context_text, sources = build_qa_context(query_to_process, model, index, df, max_docs=MAX_QA_DOCS, max_chars=MAX_CHARS_CONTEXT)
            if not sources:
                w = "Nuk u gjeten burime."
                if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
                    st.session_state.chat_history[-1]["content"] = w
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": w})
                st.warning(w)
            else:
                with st.spinner("DIELLA AI po arsyeton..."):
                    from utils.groq_integration import generate_qa_response_groq
                    response_text, sources = generate_qa_response_groq(query_to_process, context_text, sources, GROQ_API_KEY, GROQ_MODEL)
                    if response_text.startswith("Gabim"):
                        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
                            st.session_state.chat_history[-1]["content"] = response_text
                        else:
                            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                        st.error(response_text)
                    else:
                        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
                            st.session_state.chat_history[-1]["content"] = response_text
                        else:
                            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                        with st.expander("Burimet e gjetura", expanded=False):
                            for s in sources:
                                st.markdown(f"**{s['speaker']}** ({s['date']})\n\n{s['text']}\n\n---")
    st.session_state.pending_query = None
    st.session_state.pending_in_progress = False
    st.rerun()


_SUGGESTIONS = [
    "Cfare tha Diella per prokurimet publike?",
    "Cili eshte toni i deklaratave te Dielles?",
    "Cilat jane temat kryesore?",
    "Si eshte transparenca ne deklaratat e Dielles?",
    "Cilat jane deklaratat me te fundit?",
    "Si ka ndryshuar toni ne kohe?",
]

_CHAT_CSS = """
<style>
.chat-container { display: flex; flex-direction: column; gap: 1rem; padding: 1rem 0; max-height: 600px; overflow-y: auto; }
.chat-message { display: flex; margin-bottom: 1rem; }
.chat-message.user { justify-content: flex-end; }
.chat-message.ai { justify-content: flex-start; }
.chat-bubble { max-width: 75%; padding: 0.75rem 1rem; border-radius: 1rem; word-wrap: break-word; }
.chat-bubble.user { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; border-bottom-right-radius: 0.25rem; }
.chat-bubble.ai { background: #1f2937; color: #e5e7eb; border: 1px solid #374151; border-bottom-left-radius: 0.25rem; }
.chat-label { font-size: 0.75rem; color: #9ca3af; margin-bottom: 0.25rem; }
.chat-input-container { margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #374151; background: #010409; z-index: 10; }
</style>
"""
