# Metodologji tab – përshkrim i plotë për temë diplome

import streamlit as st


def render():
    st.subheader("Metodologjia e Sistemit")
    st.markdown(
        "Kjo faqe përshkruan metodologjinë e përdorur në DIELLA AI për analizën e deklaratave publike. "
        "Përshkrimi i detajuar mund të gjendet edhe në punimin e diplomës."
    )

    st.markdown("---")
    st.markdown("### 1. Të dhënat")
    st.markdown(
        "Të dhënat përfshijnë deklaratat publike të ministres Diella dhe aktorëve tjerë politikë. "
        "Çdo deklaratë ruhet në **dy gjuhë**: kolona **Speech** (anglisht) dhe **Speech_SQ** (shqip). "
        "Përfshijnë edhe datën, burimin, folësin dhe fjalëkyçet. Filtrimi në aplikacion bëhet sipas folësit dhe intervalit të datave."
    )

    st.markdown("---")
    st.markdown("### 2. Analiza e sentimentit")
    st.markdown(
        "Përdoret **VADER** (Valence Aware Dictionary and sEntiment Reasoner), një mjet **bazuar në leksik dhe rregulla** (rule-based), "
        "i krijuar për anglishten. Për çdo fjalë në leksik jepet një vlerë valencë (pozitiv/negativ); algoritmi i kombinon me rregulla "
        "(p.sh. mohimi «not», pikësimi !!!, shkronja të mëdha) dhe nxjerr një **compound score** nga -1 (shumë negativ) deri në +1 (shumë pozitiv). "
        "Sentimenti llogaritet mbi kolonën **Speech (anglisht)** që VADER të japë rezultate të besueshme. Etiketat **Pozitiv / Neutral / Negativ** "
        "caktohen sipas pragjeve (p.sh. mbi 0.05 = Pozitiv, nën -0.05 = Negativ, ndërmjet = Neutral)."
    )

    st.markdown("---")
    st.markdown("### 3. Modelimi i temave")
    st.markdown(
        "Temat nxirren me **NMF** (Non-negative Matrix Factorization) mbi paraqitjen **TF-IDF** të tekstit. "
        "Teksti shndërrohet në vektorë peshash (TF-IDF), pastaj NMF zbërthyn matricën në komponentë; çdo komponentë interpretohet si temë, "
        "me fjalëkyçet më të rëndësishme. Numri i temave dhe fjalëve për temë konfigurohet (p.sh. 5 tema, 10 fjalë). "
        "Kjo pjesë **nuk është rule-based** — është dekompozim matematikor mbi të dhënat."
    )

    st.markdown("---")
    st.markdown("### 4. Metrikat e stilit")
    st.markdown(
        "- **Gjatësia**: numri i fjalëve për deklaratë.  \n"
        "- **TTR (Type-Token Ratio)**: raporti midis fjalëve unike dhe numrit total të fjalëve; përdoret si tregues i pasurisë leksikore.  \n"
        "Llogaritjet janë formula të thjeshta, jo modele të trajnuar."
    )

    st.markdown("---")
    st.markdown("### 5. Krahasimi i folësve")
    st.markdown(
        "Të njëjtat metrika (numër deklaratash, mesatare fjalësh, TTR mesatar, sentiment mesatar) grupohen sipas **folësit**. "
        "Grafikët e krahasimit (TTR dhe shpërndarja e sentimentit) lejojnë krahasim vizual midis folësve të përzgjedhur."
    )

    st.markdown("---")
    st.markdown("### 6. Q&A (biseda me AI)")
    st.markdown(
        "Përdoret **RAG** (Retrieval-Augmented Generation): pyetja e përdoruesit kërkohet në korpusin e deklaratave me **kërkim vektorial** "
        "(SentenceTransformer për embeddings, FAISS për indeksim). Deklaratat më të ngjashme përzgjidhen si kontekst dhe dërgohen te një **model gjuhës** (Groq); "
        "modeli formulon përgjigjen vetëm në bazë të atij konteksti. Kjo pjesë bazohet në **machine learning** (embeddings, LLM), jo në rregulla me dorë."
    )

    st.markdown("---")
    st.markdown("### 7. Vlerësimi")
    st.markdown(
        "**Sentiment**: krahasohen parashikimet e VADER me etiketa të caktuara manualisht (gold labels) në një skedar të veçantë; "
        "llogariten saktësia, F1 (macro) dhe matrica e konfuzionit.  \n"
        "**Tema**: llogaritet **koherenca NPMI** e temave (nëse fjalëkyçet e një teme shfaqen së bashku në të njëjtat dokumente). "
        "Rezultatet ruhen dhe mund të përdoren në raportimin e punimit të diplomës."
    )

    st.markdown("---")
    st.caption("DIELLA AI — Sistemi i transparencës së politikave publike. Metodologjia e plotë është e detajuar në punimin e diplomës.")
