
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from config import DATA_PATH, GROQ_API_KEY, GROQ_MODEL, MAX_QA_DOCS, MAX_CHARS_CONTEXT
from utils.data_loader import load_data
from utils.vector_store import build_vector_store
from utils.ollama_integration import build_qa_context
from utils.groq_integration import generate_qa_response_groq


QA_TEST_CASES = [
    {
        "query": "Çfarë tha Diella për tenderat publikë?",
        "keywords": ["tender", "transparenc", "korrupsion", "publik", "100"],
        "expect_no_info": False,
    },
    {
        "query": "Çfarë ka thënë Sali Berisha për Diellën?",
        "keywords": ["berisha", "jokushtetues", "korrupsion", "rama", "dekret"],
        "expect_no_info": False,
    },
    {
        "query": "Si mendon Diella ta ndalojë korrupsionin?",
        "keywords": ["korrupsion", "tender", "transparenc", "fond", "publik"],
        "expect_no_info": False,
    },
    {
        "query": "Cilat janë kritikat ndaj Diellës?",
        "keywords": ["kritik", "opozit", "kushtetut", "propagand", "jokushtetues"],
        "expect_no_info": False,
    },
    {
        "query": "Çfarë roli ka Diella në e-Albania?",
        "keywords": ["e-albania", "shërbim", "asistent", "platform", "qytetar"],
        "expect_no_info": False,
    },
    {
        "query": "Çfarë planesh ka Diella për heqjen e TVSH-së?",
        "keywords": [],
        "expect_no_info": True,
    },
    {
        "query": "Çfarë mendimi ka Diella për luftën në Ukrainë?",
        "keywords": [],
        "expect_no_info": True,
    },
]


def run_evaluation():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / DATA_PATH

    print("Duke ngarkuar të dhënat...")
    df, err = load_data(str(data_path))
    if err or df is None:
        print(f"Gabim: {err}")
        return

    print("Duke ndërtuar indeksin vektorial (FAISS)...")
    model, index = build_vector_store(df)
    if model is None or index is None:
        print("Gabim: nuk mund të ndërtohet vektori.")
        return

    if not GROQ_API_KEY or not str(GROQ_API_KEY).strip():
        print("Gabim: vendosni GROQ_API_KEY në .env.")
        return

    print("\n--- Vlerësimi i modulit Q&A ---\n")
    results = []
    for i, case in enumerate(QA_TEST_CASES, 1):
        query = case["query"]
        keywords = [k.lower() for k in case["keywords"]]
        expect_no_info = case["expect_no_info"]

        context_text, sources = build_qa_context(
            query, model, index, df, max_docs=MAX_QA_DOCS, max_chars=MAX_CHARS_CONTEXT
        )
        response_text, _ = generate_qa_response_groq(
            query, context_text, sources, GROQ_API_KEY, GROQ_MODEL
        )

        response_lower = response_text.lower()
        is_error = response_text.startswith("Gabim") or response_text.startswith("Error")

        if expect_no_info:
            hit = (
                "nuk ka informacion" in response_lower
                or "nuk u gjetën burime" in response_lower
            ) and not is_error
        else:
            hit = not is_error and any(kw in response_lower for kw in keywords)

        results.append(
            {
                "nr": i,
                "query": query[:50] + "..." if len(query) > 50 else query,
                "expect_no_info": expect_no_info,
                "hit": hit,
                "response_preview": response_text[:120].replace("\n", " ") + "...",
            }
        )
        status = "OK" if hit else "FAIL"
        print(f"{i}. [{status}] {query[:55]}...")

    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    accuracy = (hits / total * 100) if total else 0

    print("\n" + "=" * 50)
    print(f"Rezultati: {hits}/{total} pyetje të përgjigjura sipas kriterit.")
    print(f"Accuracy (e thjeshtë): {accuracy:.1f}%")
    print("=" * 50)

    summary = {
        "total_questions": total,
        "correct": hits,
        "accuracy_percent": round(accuracy, 1),
    }
    return summary, results


if __name__ == "__main__":
    run_evaluation()
