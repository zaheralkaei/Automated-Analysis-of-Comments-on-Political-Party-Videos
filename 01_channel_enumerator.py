import os
import csv
import isodate
from pathlib import Path
from dotenv import load_dotenv
from utils.api_helpers import build_youtube_client, execute_with_retry

load_dotenv()

CHANNEL_A_ID = os.getenv("CHANNEL_A_ID")
CHANNEL_B_ID = os.getenv("CHANNEL_B_ID")
BASE_DATA_DIR = Path(os.getenv("BASE_DATA_DIR", "./data"))
OUT_DIR = BASE_DATA_DIR / "01_channel_videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FIELDS = ["videoId", "title", "publishedAt", "viewCount", "likeCount",
              "commentCount", "duration", "channelId", "channelName", "url"]


def get_uploads_playlist_id(youtube, channel_id: str) -> str:
    req = youtube.channels().list(part="contentDetails", id=channel_id)
    resp = execute_with_retry(req)
    return resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def enumerate_playlist_videos(youtube, playlist_id: str) -> list:
    video_ids = []
    page_token = None
    while True:
        req = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=page_token,
        )
        resp = execute_with_retry(req)
        for item in resp.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return video_ids


def fetch_video_metadata(youtube, video_ids: list) -> list:
    records = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        req = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch),
        )
        resp = execute_with_retry(req)
        for item in resp.get("items", []):
            stats   = item.get("statistics", {})
            content = item.get("contentDetails", {})
            records.append({
                "videoId":      item["id"],
                "title":        item["snippet"]["title"],
                "publishedAt":  item["snippet"]["publishedAt"],
                "viewCount":    int(stats.get("viewCount", 0)),
                "likeCount":    int(stats.get("likeCount", 0)),   
                "commentCount": int(stats.get("commentCount", 0)),
                "duration":     content.get("duration", ""),      
                "channelId":    item["snippet"]["channelId"],
                "channelName":  item["snippet"]["channelTitle"],
                "url":          f"https://www.youtube.com/watch?v={item['id']}",
            })
    return records


def save_channel_csv(records: list, label: str) -> Path:
    path = OUT_DIR / f"channel_{label}_videos.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} videos → {path}")
    return path



def combine_csvs(path_a: Path, path_b: Path) -> Path:
    seen = set()
    combined = []
    for path in (path_a, path_b):
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["videoId"] not in seen:
                    seen.add(row["videoId"])
                    combined.append(row)
    out = OUT_DIR / "combined_videos.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(combined)
    print(f"Combined CSV: {len(combined)} unique videos → {out}")
    return out


def main():
    youtube = build_youtube_client()
    paths = {}
    for label, channel_id in [("a", CHANNEL_A_ID), ("b", CHANNEL_B_ID)]:
        print(f"\nEnumerating channel {label} ({channel_id})...")
        playlist_id = get_uploads_playlist_id(youtube, channel_id)
        video_ids = enumerate_playlist_videos(youtube, playlist_id)
        print(f"  Found {len(video_ids)} videos. Fetching metadata...")
        records = fetch_video_metadata(youtube, video_ids)
        paths[label] = save_channel_csv(records, label)
    combine_csvs(paths["a"], paths["b"])



if __name__ == "__main__":
    main()
