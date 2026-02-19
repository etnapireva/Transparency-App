# Evaluation tab
import json
from pathlib import Path
import pandas as pd
import streamlit as st
import run_evaluation
from config import DATA_PATH


def render(base_dir, data_path=None):
    data_path = data_path or base_dir / DATA_PATH
    st.subheader("Vlerësim i sistemit (për temë)")
    with st.expander("Metodologjia e vlerësimit", expanded=True):
        st.markdown("""
        Vlerësimi përfshin dy pjesë. E para është vlerësimi i sentimentit: një grup deklarata me etiketa të caktuara manualisht (Pozitiv / Neutral / Negativ) në skedarin evaluation_sentiment_gold.csv krahasohet me parashikimet e VADER; saktësia dhe F1 matin pajtueshmërinë e modelit me këto etiketa. E dyta është koherenca e temave (NPMI): për temat e nxirra nga NMF matet nëse fjalëkyçet e tyre shfaqen së bashku në të njëjtat dokumente; vlera më e lartë NPMI tregon tema më koherente.
        """)
    st.markdown("Ekzekutoni vlerësimin më poshtë. Rezultatet ruhen edhe në `evaluation_results.json`.")

    if st.button("Ekzekuto vlerësimin", type="primary", key="run_eval_btn"):
        eval_results = {}
        gold_path = base_dir / "evaluation_sentiment_gold.csv"

        with st.spinner("Po ekzekutohet vlerësimi i sentimentit..."):
            sent = run_evaluation.evaluate_sentiment(gold_path)
            if sent:
                eval_results.update(sent)

        with st.spinner("Po llogaritet koherenca e temave (NPMI)..."):
            coh = run_evaluation.run_topic_coherence(data_path)
            if coh:
                eval_results.setdefault("topic_coherence", {}).update(coh["topic_coherence"])

        if eval_results:
            results_file = base_dir / "evaluation_results.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)
            st.session_state["eval_results"] = eval_results
            st.success(f"Rezultatet u ruajtën në `{results_file.name}`.")
        else:
            st.session_state["eval_results"] = None
            st.warning(
                "Nuk u gjenden rezultate. Kontrolloni që ekziston "
                "`evaluation_sentiment_gold.csv` (kolona: Speech, GoldLabel) dhe që të dhënat kryesore janë të ngarkuara."
            )
        st.rerun()

    if st.session_state.get("eval_results"):
        eval_results = st.session_state["eval_results"]
        if "sentiment" in eval_results:
            s = eval_results["sentiment"]
            st.markdown("### Sentiment (VADER vs etiketa të arta)")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Saktësia", f"{s['accuracy']:.2%}")
            with c2:
                st.metric("F1 (macro)", f"{s['f1_macro']:.4f}")
            with c3:
                st.metric("Mostra", s["n_samples"])
            st.code(s["classification_report"], language=None)
            st.markdown("**Matrica e konfuzionit:**")
            st.dataframe(
                pd.DataFrame(s["confusion_matrix"], index=s["labels"], columns=s["labels"]),
                use_container_width=False,
            )
        if "topic_coherence" in eval_results:
            tc = eval_results["topic_coherence"]
            st.markdown("### Koherenca e temave (NPMI)")
            st.metric("NPMI (mesatare)", f"{tc.get('npmi_mean', 0):.4f}")
            if "n_docs_used" in tc:
                st.caption(f"Dokumente të përdorura: {tc['n_docs_used']}")
