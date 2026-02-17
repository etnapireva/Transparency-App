import os
import re
import datetime as dt
import requests
import pandas as pd
import trafilatura
from collections import Counter

ARTICLES = [
    "https://apnews.com/article/albania-new-cabinet-program-ai-minister-diella-corruption-3aa58c801d69b5b295975cc68079a2d3",
    "https://www.abc.net.au/news/2025-09-19/ai-generated-minister-addresses-albanian-parliament/105791708",
]
OUT_CSV = "data/diella_speeches.csv"

STOPWORDS = {
    "the","a","an","and","or","but","for","with","of","to","in","on","at","by","from","as","is","are","was","were","be","been","being",
    "this","that","these","those","it","its","we","our","you","they","their","he","she","his","her","them","i",
    "dhe","te","të","ne","në","me","për","per","nga","si","qe","që","ose","por","është","eshte","janë","jane","ishte","jemi","duhet","kemi",
    "ky","kjo","këtë","kete","ai","ajo","ata","ato","pra","edhe","mund","do","po","jo","kështu","keshtu","citizens","government","minister"
}

def keywords_from_text(text: str, top_n: int = 8) -> str:
    tokens = re.findall(r"[A-Za-zËëÇçÀ-ÖØ-öø-ÿ']+", text.lower())
    words = [w for w in tokens if len(w) >= 3 and w not in STOPWORDS]
    common = [w for w, _ in Counter(words).most_common(30)]
    return ", ".join(common[:top_n])

def fetch_html(url: str):
    html = trafilatura.fetch_url(url)  # some versions don't support timeout kwarg
    if html:
        return html
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if r.ok:
            return r.text
    except Exception:
        pass
    return None

def extract_article(url: str):
    downloaded = fetch_html(url)
    if not downloaded:
        return None

    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text or len(text) < 400:
        return None

    # Metadata (robust to trafilatura version differences)
    title, date_iso = "", None
    try:
        from trafilatura.metadata import extract_metadata as tf_extract_metadata
        try:
            meta = tf_extract_metadata(downloaded)  # no 'url' kwarg in some versions
        except TypeError:
            meta = tf_extract_metadata(downloaded)
        if meta:
            title = getattr(meta, "title", "") or ""
            meta_date = getattr(meta, "date", None)
            if meta_date:
                try:
                    date_iso = pd.to_datetime(meta_date, errors="coerce").date().isoformat()
                except Exception:
                    date_iso = None
    except Exception:
        pass

    if not date_iso:
        date_iso = dt.date.today().isoformat()

    kw = keywords_from_text(text)
    return {"Date": date_iso, "Speech": text, "Keywords": kw, "Source": url, "Title": title}

def main():
    rows = []
    for url in ARTICLES:
        art = extract_article(url)
        if art:
            rows.append(art)
        else:
            print(f"Failed to extract: {url}")

    if not rows:
        print("No articles extracted.")
        return

    df_new = pd.DataFrame(rows)
    if os.path.exists(OUT_CSV):
        df_old = pd.read_csv(OUT_CSV)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.drop_duplicates(subset=["Source"], inplace=True)
    else:
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        df_all = df_new

    df_all.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df_all)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()