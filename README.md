# DIELLA AI — Sistemi i Transparencës së Politikave

Aplikacion Streamlit për analizën e deklaratave publike (ministria Diella, Rama, opozita). Përdor NLP për sentiment (VADER), tema (NMF), metrika stili (TTR), krahasim folësish dhe Q&A në shqip. Projekt tezë master — Etna Pireva.
Aplikacion Streamlit për analizën e deklaratave publike (ministria Diella, Rama, opozita). Përdor NLP për sentiment (VADER), tema (NMF), metrika stili (TTR), krahasim folësish dhe Q&A në shqip. Projekt— Etna Pireva.

---

## Udhëzues i shpejtë për përdorim 

### 1. Kërkesat

- **Python 3.10+**
- Paketa: Streamlit, Pandas, Plotly, Altair, scikit-learn, vaderSentiment, sentence-transformers, faiss-cpu, python-dotenv, groq (shiko **Instalimi** më poshtë).

### 2. Instalimi

```bash
# Hapni terminalin në rrënjën e projektit (Transparency-App)
cd Transparency-App/project/data/project

# (Opsional) Krijo një mjedis virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Instaloni varësitë
pip install streamlit pandas plotly altair scikit-learn vaderSentiment sentence-transformers faiss-cpu python-dotenv groq
```

Nëse ekziston skedari `requirements.txt` në të njëjtin folder:

```bash
pip install -r requirements.txt
```

### 3. Të dhënat

- Skedari kryesor i të dhënave duhet të jetë në të njëjtin folder me aplikacionin: **`diella_speeches_clean.csv`** (me kolona: Date, Speech, Keywords, Source, Title, Speaker, Speech_SQ). Ky skedar është pjesë e repo-së.
- Nuk nevojitet ndonjë konfigurim tjetër për të parë dashboard-in, sentimentin, temat dhe metrikat.

### 4. Nisja e aplikacionit

Nga folderi **`project/data/project`** (ku ndodhen `diella_ai_analysis.py` dhe `config.py`):

```bash
streamlit run diella_ai_analysis.py
```

Hapni në shfletues adresën që shfaqet (zakonisht `http://localhost:8501`).

### 5. Si përdoret

- **Sidebar (majtas):**  
  - **Zgjidh folësin:** zgjidhni **"Të gjithë"** për të parë të gjitha deklaratat, ose një folës (Diella, Edi Rama, etj.).  
  - **Data nga / Data deri:** intervali i datave.  
  - Numri i deklaratave që përputhen me filtrin shfaqet poshtë.

- **Tab-et:**  
  - **Dashboard:** total deklarata, sentiment mesatar, folësi më aktiv, grafikë, deklaratat e fundit (me zgjedhje numri dhe gjuhe Shqip/English).  
  - **Sentiment:** shpërndarja e sentimentit dhe lista e deklaratave me expander.  
  - **Topics:** tema (NMF) me fjalëkyçe dhe mundësi për të parë deklaratat për çdo temë.  
  - **Style Metrics:** gjatësi fjalësh, TTR, tema kryesore.  
  - **Krahasim Folësish:** tabelë statistikash dhe grafikë krahasimi (TTR, sentiment) për 2+ folës.  
  - **Q&A:** pyetje në shqip mbi deklaratat (kërkim vektorial + model gjuhës). Kërkon çelës Groq (shiko më poshtë).  
  - **Vlerësim:** ekzekutimi i vlerësimit të sentimentit dhe koherencës së temave (NPMI); rezultatet ruhen në `evaluation_results.json`.

### 6. Q&A (opsional)

Për të përdorur tab-in **Q&A** (pyetje në shqip), duhet një **Groq API key**:

1. Regjistrohuni / hyni në [console.groq.com](https://console.groq.com/) dhe krijoni një API key.
2. Në folderin **`project/data/project`** krijoni një skedar **`.env`** me përmbajtje:

   ```
   GROQ_API_KEY=your_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

3. Rinisni aplikacionin. Pa këtë çelës, Q&A nuk do të funksionojë, por pjesa tjetër e aplikacionit funksionon normalisht.

### 7. Vlerësimi (për tezë)

- Në tab-in **Vlerësim** klikoni **"Ekzekuto vlerësimin"**.  
- Përdoren: `evaluation_sentiment_gold.csv` (sentiment) dhe `diella_speeches_clean.csv` (koherenca e temave).  
- Rezultatet shfaqen në ekran dhe ruhen në **`evaluation_results.json`** në të njëjtin folder.

---

## Struktura e projektit (të rëndësishme për ekzekutim)

- **`project/data/project/`** — folderi nga i cili duhet të ekzekutohet aplikacioni:
  - `diella_ai_analysis.py` — skedari kryesor i aplikacionit (ky duhet të jepet te `streamlit run`).
  - `config.py` — konfigurimi (rrugë të dhënash, modele, etj.).
  - `diella_speeches_clean.csv` — të dhënat kryesore.
  - `evaluation_sentiment_gold.csv` — etiketa për vlerësimin e sentimentit.
  - `utils/` — module për të dhëna, vizualizime, NLP, vektorë, Groq.
  - `run_evaluation.py` — skript i vlerësimit (përdoret edhe nga tab-i Vlerësim në app).

---

## Teknologji

Streamlit, Pandas, Plotly, Altair, scikit-learn, VADER, SentenceTransformer, FAISS, Groq API (për Q&A).

---

## Licensë

I hapur për përdorim akademik.

Zhvilluar nga **Etna Pireva**.
