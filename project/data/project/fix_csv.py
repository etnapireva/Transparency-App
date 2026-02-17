import re
import os
import pandas as pd
from datetime import date

IN_PATH = "data/diella_speeches.csv"
OUT_PATH = "data/diella_speeches_clean.csv"

KNOWN_SPEAKERS = {
    r"\bdiella(\s*\(ai\))?\b": "Diella",
    r"\bedi\s+rama\b": "Edi Rama",
    r"\bsali\s+berisha\b": "Sali Berisha",
    r"\bgazment\s+bardhi\b": "Gazment Bardhi",
    r"\bsaimir\s+korreshi\b": "Saimir Korreshi",
}

ARTICLE_HOSTS = ("apnews.com", "abc.net.au", "reuters.com", "theguardian.com")

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def detect_speaker(text: str) -> str | None:
    low = text.lower()
    for pat, name in KNOWN_SPEAKERS.items():
        if re.search(pat, low):
            return name
    return None

def strip_speaker_from_keywords(kw: str) -> str:
    parts = [normalize_spaces(p) for p in kw.split(",") if p.strip()]
    keep = []
    for p in parts:
        if detect_speaker(p) is None:
            keep.append(p)
    return ", ".join(keep)

def infer_type(speech: str, source: str, title: str) -> str:
    s_low = (speech or "").lower()
    src_low = (source or "").lower()
    if "paraphrase" in s_low or "reported" in s_low:
        return "reported"
    if any(h in src_low for h in ARTICLE_HOSTS):
        return "article"
    return "statement"

def main():
    if not os.path.exists(IN_PATH):
        print(f"Missing {IN_PATH}")
        return

    df = pd.read_csv(IN_PATH)

    # Ensure expected columns exist
    for col in ["Date","Speech","Keywords","Source","Title","Speaker","Type"]:
        if col not in df.columns:
            df[col] = ""

    # Normalize text fields
    for col in ["Speech","Keywords","Source","Title","Speaker"]:
        df[col] = df[col].fillna("").astype(str)

    # Parse/clean dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df["Date"] = df["Date"].dt.date

    # Build a combined text per row to detect speaker
    def row_speaker(row):
        # Priority: explicit Speaker if already set
        if row.get("Speaker", "").strip():
            sp = detect_speaker(row["Speaker"])
            if sp:
                return sp

        blobs = " | ".join([
            row.get("Title",""),
            row.get("Keywords",""),
            row.get("Speech","")
        ])
        sp = detect_speaker(blobs)
        if sp:
            return sp

        # Default to Diella if it sounds like Diella’s own first-person lines
        if re.search(r"\bi am not here to replace people\b", row.get("Speech","").lower()):
            return "Diella"

        return "Diella"  # safe default

    df["Speaker"] = df.apply(row_speaker, axis=1)

    # Clean Keywords if they contain speaker names
    df["Keywords"] = df["Keywords"].apply(strip_speaker_from_keywords)

    # Infer Type
    df["Type"] = df.apply(lambda r: infer_type(r["Speech"], r["Source"], r["Title"]), axis=1)

    # Basic dedup: by Source (if present) else by (Date, Speech)
    before = len(df)
    if (df["Source"].str.strip() != "").any():
        df = df.sort_values(["Date"]).drop_duplicates(subset=["Source"], keep="first")
    df = df.drop_duplicates(subset=["Date","Speech"], keep="first")
    after = len(df)

    # Reorder columns
    cols = ["Date","Speaker","Type","Speech","Source","Title","Keywords"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # Final tidy
    df["Speech"] = df["Speech"].apply(lambda s: s.replace("\r\n","\n").strip())
    df["Title"] = df["Title"].apply(normalize_spaces)
    df["Keywords"] = df["Keywords"].apply(normalize_spaces)
    df = df.reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {after} rows to {OUT_PATH} (removed {before - after} duplicates).")
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()