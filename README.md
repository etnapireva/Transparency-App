# Transparency-App

# DIELLA AI - Sistemi i Transparencës së Politikave

## Përshkrimi
DIELLA AI është një aplikacion Streamlit për analizën e deklaratave publike të ministres Diella. Përdor NLP për sentiment, tema, metrika stili dhe Q&A në shqip. Projekt për tezë master.

## Karakteristikat Kryesore
- **Dashboard**: Metrika, grafikë sentimenti dhe trendet ditore.
- **Analiza**: Sentiment (VADER), tema (NMF), metrika stili (TTR), krahasim folësish.
- **Q&A AI**: Pyetje në shqip me OpenAI API dhe kërkim vektorial.
- **Filtrimi**: Sipas folësi dhe datës.

## Teknologjitë
Streamlit, Pandas, Plotly, scikit-learn, VADER, SentenceTransformer, FAISS, OpenAI API.


## Përdorimi
1. Ekzekutoni: `streamlit run app.py`.
2. Eksploroni tabs dhe bëni pyetje në Q&A.

## Struktura e të Dhënave
CSV me tekste në anglisht/shqip, folës dhe datë.

## Licensa
Hapur për përdorim akademik.

Zhvilluar nga Etna Pireva. 🚀