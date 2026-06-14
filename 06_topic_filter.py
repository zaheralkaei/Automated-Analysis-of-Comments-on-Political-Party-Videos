import os
import csv
import json
from pathlib import Path
from dotenv import load_dotenv
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

GERMAN_STOPWORDS = stopwords.words("german")

load_dotenv()

DATASETS = [
    ("a_b", Path(os.getenv("BASE_DATA_DIR_AB", "./data_a_b"))),
    ("c_d", Path(os.getenv("BASE_DATA_DIR_CD", "./data_c_d"))),
]

OUT_DIR = Path("./06_bertopic_output")
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV    = OUT_DIR / "video_topics.csv"
MODEL_PATH = OUT_DIR / "bertopic_model"

TOP_N = 10

CSV_FIELDS = [
    "videoId", "dataset", "title", "channelName", "publishedAt",
    "primary_topic_id", "primary_topic_keywords",
    "top_topic_ids", "top_topic_keywords",
]


# helpers 

def load_metadata(dataset_dir: Path) -> dict:
    """Return {videoId: row} from combined_videos.csv."""
    csv_path = dataset_dir / "01_channel_videos" / "combined_videos.csv"
    meta = {}
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                meta[row["videoId"]] = row
    return meta


def load_transcript(path: Path) -> str:
    """Concatenate all transcript segments for one video into a single string."""
    segments = []
    for line in path.read_text(encoding="utf-8", errors="replace").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            text = obj.get("Transcript", "").strip()
            if text:
                segments.append(text)
        except json.JSONDecodeError:
            continue
    return " ".join(segments)


def collect_all_transcripts() -> tuple[list[str], list[dict]]:
    """
    Walk both datasets, load every transcript file, return:
      docs  — list of full transcript strings
      metas — list of {videoId, dataset, title, channelName, publishedAt}
    """
    docs, metas = [], []
    for label, ds_dir in DATASETS:
        t_dir = ds_dir / "02_raw_scraped"
        if not t_dir.exists():
            print(f"  [WARN] transcript dir not found: {t_dir}")
            continue
        metadata = load_metadata(ds_dir)
        files = sorted(t_dir.glob("transcripts_*.jsonl"))
        print(f"  {label}: {len(files)} transcript files")
        for f in files:
            video_id = f.stem.replace("transcripts_", "")
            text = load_transcript(f)
            if not text.strip():
                continue
            meta_row = metadata.get(video_id, {})
            docs.append(text)
            metas.append({
                "videoId":     video_id,
                "dataset":     label,
                "title":       meta_row.get("title", ""),
                "channelName": meta_row.get("channelName", ""),
                "publishedAt": meta_row.get("publishedAt", ""),
            })
    return docs, metas


def topic_keywords(topic_model: BERTopic, topic_id: int, n: int = 8) -> str:
    """Return top-n keywords for a topic as a comma-separated string."""
    if topic_id == -1:
        return "outlier"
    words = topic_model.get_topic(topic_id)
    if not words:
        return ""
    return ", ".join(w for w, _ in words[:n])


# main 

def main():
    print("Loading transcripts from all datasets...")
    docs, metas = collect_all_transcripts()
    print(f"Total documents: {len(docs)}")

    print("\nFitting BERTopic")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    vectorizer_model = CountVectorizer(
        stop_words=GERMAN_STOPWORDS,
        ngram_range=(1, 2),
        min_df=5,
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        language="multilingual",
        min_topic_size=15,
        nr_topics="auto",
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = topics.count(-1)
    print(f"\nDiscovered {n_topics} topics ({n_outliers} outliers kept as -1)")

    print("\nComputing per-document topic distributions...")
    topic_distr, _ = topic_model.approximate_distribution(docs, min_similarity=0.0)

    print(f"\nSaving results to {OUT_CSV}")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for i, (meta, primary_topic) in enumerate(zip(metas, topics)):
            # top-N topics by probability
            probs = topic_distr[i]
            top_ids = sorted(range(len(probs)), key=lambda x: probs[x], reverse=True)[:TOP_N]
            top_ids = [t for t in top_ids if probs[t] > 0]

            writer.writerow({
                **meta,
                "primary_topic_id":       primary_topic,
                "primary_topic_keywords": topic_keywords(topic_model, primary_topic),
                "top_topic_ids":          ";".join(str(t) for t in top_ids),
                "top_topic_keywords":     " | ".join(topic_keywords(topic_model, t) for t in top_ids),
            })

    print(f"\nSaving BERTopic model to {MODEL_PATH}")
    topic_model.save(str(MODEL_PATH), serialization="pickle")

    # save topic overview
    topic_overview = OUT_DIR / "topic_overview.csv"
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(topic_overview, index=False, encoding="utf-8")
    print(f"Topic overview saved to {topic_overview}")

    print("\nDone.")


if __name__ == "__main__":
    main()
