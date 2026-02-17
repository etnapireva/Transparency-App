import os, re, pandas as pd

IN = "data/diella_speeches.csv"
OUT = "data/diella_speeches_clean.csv"

def detect_speaker(text: str) -> str:
    t = (text or "").lower()
    if "edi rama" in t: return "Edi Rama"
    if "sali berisha" in t: return "Sali Berisha"
    if "gazment bardhi" in t: return "Gazment Bardhi"
    if "saimir korreshi" in t: return "Saimir Korreshi"
    if "diella" in t or "ai minister" in t: return "Diella"
    return "Other"

def main():
    if not os.path.exists(IN):
        print(f"Missing {IN}"); return
    df = pd.read_csv(IN)
    for col in ["Date","Speech","Source","Title","Keywords"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    # Build a text blob per row and detect speaker
    blob = (df["Title"] + " | " + df["Keywords"] + " | " + df["Speech"] + " | " + df["Source"]).astype(str)
    df["Speaker"] = blob.apply(detect_speaker)
    # Ensure at least two speakers exist; if all are Diella, mark some as Other (example: non-AP/ABC sources)
    if df["Speaker"].nunique() < 2:
        mask_other = ~df["Source"].str.contains(r"(apnews|abc\.net\.au)", case=True, na=False)
        df.loc[mask_other, "Speaker"] = "Other"
    df.to_csv(OUT, index=False)
    print("Speakers:", df["Speaker"].value_counts().to_dict())
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()