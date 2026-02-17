# ==========================================
# GROQ INTEGRATION MODULE - DIELLA AI
# ==========================================

from groq import Groq
import pandas as pd


def generate_qa_response_groq(query, context_text, sources, api_key, model):
    """
    Generate Q&A response using Groq API.
    
    Args:
        query (str): User query in Albanian
        context_text (str): Context from documents
        sources (list): List of source documents
        api_key (str): Groq API key
        model (str): Model name
        
    Returns:
        tuple: (response_text, sources)
    """
    if not sources or not context_text:
        return "Nuk u gjetën burime të përshtatshme.", []

    try:
        client = Groq(api_key=api_key)

        system_msg = (
            "Ti je DIELLA AI — asistent ekspert që përgjigjet vetëm në shqip. "
            "Përdor VETËM informacionin që ndodhet në KONTEKSTIN e dhënë më poshtë. "
            "Përgjigju shkurt, qartë dhe jep referencat në formatin [n] ku n është numri i burimit. "
            "Nëse informacioni nuk është i mjaftueshëm në burime, thuaj qartë se 'Nuk ka informacion të mjaftueshëm në burimet e dhëna'. "
            "Mos imagjino informacion të ri jashtë kontekstit. Jep një përmbledhje 2-4 rreshtash."
        )

        user_msg = (
            f"Pyetja e përdoruesit: \"{query}\"\n\n"
            "KONTEKSTI (Burimet e Deklaratave të numëruara):\n"
            f"{context_text}\n\n"
            "Përgjigju në shqip, përdor vetëm këtë kontekst dhe cito burimin(ët) në tekst me [n]."
        )

        message = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model=model,
            temperature=0.7,
            max_tokens=500,
        )

        response_text = message.choices[0].message.content.strip()
        return response_text, sources

    except Exception as e:
        error_msg = f"Gabim gjatë komunikimit me Groq: {str(e)}"
        return error_msg, []