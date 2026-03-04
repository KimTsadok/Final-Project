import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from math import log2

from google.cloud import videointelligence_v1 as vi
from google.cloud import storage


# -----------------------------
# Configuration
# -----------------------------
# Load environment variables from a local .env file (if present).
def load_dotenv(dotenv_path: str = ".env") -> None:
    env_path = Path(dotenv_path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_dotenv()
DEFAULT_BUCKET_NAME = os.getenv("DEFAULT_BUCKET_NAME", "final-project-bucket1")
DEFAULT_GCS_PREFIX = os.getenv("DEFAULT_GCS_PREFIX", "uploads")
DEFAULT_OUT_PATH = os.getenv("DEFAULT_OUT_PATH", "VideoFeatures.json")
# -----------------------------
# Time conversion helpers
# -----------------------------
def duration_to_seconds(d) -> float:
    """
    Convert either:
    - protobuf Duration (seconds + nanos)
    - datetime.timedelta (total_seconds)
    - or numeric
    into float seconds.

    CHANGE (fix): Previously we assumed protobuf Duration always had `.nanos`.
    In your environment, offsets can come back as `datetime.timedelta`, so we
    now support both to prevent: AttributeError: 'datetime.timedelta' has no attribute 'nanos'
    """
    if d is None:
        return 0.0

    # Case 1: Python timedelta
    if hasattr(d, "total_seconds"):
        return float(d.total_seconds())

    # Case 2: Protobuf Duration-like
    if hasattr(d, "seconds"):
        nanos = getattr(d, "nanos", 0)
        return float(d.seconds) + float(nanos) / 1e9

    # Case 3: fallback numeric
    return float(d)


# -----------------------------
# GCS Upload
# -----------------------------
def build_destination_name(local_path: Path, prefix: str) -> str:
    """
    CHANGE (important): We used to add a timestamp to the destination name
    (e.g. uploads/20260228_000755_ACCEDE09230.mp4). That CAUSED duplicates
    every time you ran the script.

    NEW behavior: stable object name (no timestamp), so we can detect if the file
    already exists and skip upload:
      uploads/ACCEDE09230.mp4
    """
    prefix = prefix.strip("/")
    return f"{prefix}/{local_path.name}"


def upload_to_gcs(local_video_path: str, bucket_name: str, prefix: str) -> str:
    """
    Uploads a local file to GCS ONLY if it does NOT already exist in the bucket.
    Returns a gs:// URI.

    CHANGE (important): We added an existence check:
        if blob.exists(): skip upload
    This prevents wasting storage from duplicate uploads when rerunning the script.
    """
    path = Path(local_video_path).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    destination_blob_name = build_destination_name(path, prefix)
    blob = bucket.blob(destination_blob_name)

    # NEW: conditional upload
    if blob.exists(client=client):
        print(f"Already exists in bucket, skipping upload: gs://{bucket_name}/{destination_blob_name}")
    else:
        print(f"Uploading to bucket: gs://{bucket_name}/{destination_blob_name}")
        blob.upload_from_filename(str(path))

    return f"gs://{bucket_name}/{destination_blob_name}"


# -----------------------------
# Video Intelligence Analysis
# -----------------------------
def analyze_video_uri(video_uri: str, timeout_seconds: int = 600):
    client = vi.VideoIntelligenceServiceClient()
    features = [
        vi.Feature.LABEL_DETECTION,
        vi.Feature.OBJECT_TRACKING,
        vi.Feature.SHOT_CHANGE_DETECTION,
    ]

    print("\nSending video for analysis...")
    operation = client.annotate_video(
        request={"input_uri": video_uri, "features": features}
    )

    print("Waiting for results...")
    result = operation.result(timeout=timeout_seconds)  # timeout in seconds
    return result.annotation_results[0]


# -----------------------------
# Feature Engineering (Week 2 deliverable)
# -----------------------------
def merge_intervals(intervals):
    """
    Merge overlapping intervals [(start, end), ...] and return merged list.
    """
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]

    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1][1] = max(last_e, e)
        else:
            merged.append([s, e])

    return [(s, e) for s, e in merged]


def intervals_total_length(intervals):
    return sum(max(0.0, e - s) for s, e in intervals)


def shannon_entropy_from_counts(counts: Counter) -> float:
    """
    Shannon entropy H = -sum(p_i * log2(p_i))
    where p_i is the probability of each object type in the frequency distribution.
    """
    total = sum(counts.values())
    if total <= 0:
        return 0.0

    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * log2(p)

    return entropy


def compute_video_duration_seconds(annotations) -> float:
    """
    Prefer duration from shots; fallback to object segments.
    """
    shot_ends = [
        duration_to_seconds(s.end_time_offset)
        for s in getattr(annotations, "shot_annotations", [])
    ]
    if shot_ends:
        return max(shot_ends)

    obj_ends = []
    for obj in getattr(annotations, "object_annotations", []):
        obj_ends.append(duration_to_seconds(obj.segment.end_time_offset))

    return max(obj_ends) if obj_ends else 0.0


def compute_features(annotations, video_uri: str) -> dict:
    """
    Computes Week 2 derived features and returns a dict that will be saved to JSON:
      - avg_shot_length
      - shot_variance
      - object_frequency_distribution
      - human_presence_ratio
      - object_entropy
      - interaction_density
    """
    # ---- Duration ----
    duration_s = compute_video_duration_seconds(annotations)

    # ---- Shots: avg + variance ----
    shot_lengths = []
    for shot in getattr(annotations, "shot_annotations", []):
        s = duration_to_seconds(shot.start_time_offset)
        e = duration_to_seconds(shot.end_time_offset)
        if e > s:
            shot_lengths.append(e - s)

    if shot_lengths:
        avg_shot_length = sum(shot_lengths) / len(shot_lengths)
        shot_variance = sum((x - avg_shot_length) ** 2 for x in shot_lengths) / len(shot_lengths)
    else:
        avg_shot_length = 0.0
        shot_variance = 0.0

    # ---- Labels (unique) ----
    labels = []
    for lab in getattr(annotations, "segment_label_annotations", []):
        desc = getattr(lab.entity, "description", None)
        if desc:
            labels.append(desc)

    # ---- Objects: distribution + person intervals ----
    obj_descs = []
    person_intervals = []
    for obj in getattr(annotations, "object_annotations", []):
        desc = getattr(obj.entity, "description", None)
        if not desc:
            continue

        obj_descs.append(desc)

        if desc.lower() == "person":
            s = duration_to_seconds(obj.segment.start_time_offset)
            e = duration_to_seconds(obj.segment.end_time_offset)
            if e > s:
                person_intervals.append((s, e))

    obj_counts = Counter(obj_descs)

    # ---- Human presence ratio: union(person intervals) / duration ----
    if duration_s > 0 and person_intervals:
        merged = merge_intervals(person_intervals)
        human_presence_ratio = intervals_total_length(merged) / duration_s
        human_presence_ratio = max(0.0, min(1.0, human_presence_ratio))
    else:
        human_presence_ratio = 0.0

    # ---- Object entropy ----
    object_entropy = shannon_entropy_from_counts(obj_counts)

    # ---- Interaction density ----
    # Definition: number of tracked object segments per second.
    total_object_tracks = len(getattr(annotations, "object_annotations", []))
    interaction_density = (total_object_tracks / duration_s) if duration_s > 0 else 0.0

    # ---- JSON-friendly dicts ----
    object_frequency_distribution = dict(obj_counts.most_common())

    features = {
        "video_uri": video_uri,
        "duration_seconds": round(duration_s, 6),

        "shots": {
            "count": len(shot_lengths),
            "avg_shot_length": round(avg_shot_length, 6),
            "shot_variance": round(shot_variance, 6),
            "shot_lengths": [round(x, 6) for x in shot_lengths],
        },

        "labels": {
            "unique": sorted(set(labels)),
            "count_unique": len(set(labels)),
        },

        "objects": {
            "total_tracks": total_object_tracks,
            "object_frequency_distribution": object_frequency_distribution,
            "top_objects": [k for k, _ in obj_counts.most_common(10)],
            "object_entropy": round(object_entropy, 6),
            "human_presence_ratio": round(human_presence_ratio, 6),
            "interaction_density_tracks_per_sec": round(interaction_density, 6),
        },
    }

    return features


def save_json(data: dict, out_path: str):
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved features to: {out_file.resolve()}")


# -----------------------------
# CLI / Main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload a local video to GCS (only if not already there), analyze it, and output VideoFeatures.json"
    )
    parser.add_argument("--video", type=str, help="Local path to video file (mp4, etc.)")
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET_NAME, help="GCS bucket name")
    parser.add_argument("--prefix", type=str, default=DEFAULT_GCS_PREFIX, help="GCS folder/prefix")
    parser.add_argument("--timeout", type=int, default=600, help="Analysis timeout in seconds")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_PATH, help="Output JSON file path")
    parser.add_argument("--print_summary", action="store_true", help="Print a short summary to console")
    return parser.parse_args()


def print_summary(features: dict):
    print("\n=== Feature Summary ===")
    print(f"Duration: {features['duration_seconds']}s")
    print(
        f"Shots: {features['shots']['count']} | "
        f"avg={features['shots']['avg_shot_length']} | "
        f"var={features['shots']['shot_variance']}"
    )
    print(f"Top objects: {features['objects']['top_objects']}")
    print(f"Human presence ratio: {features['objects']['human_presence_ratio']}")
    print(f"Object entropy: {features['objects']['object_entropy']}")
    print(f"Interaction density (tracks/sec): {features['objects']['interaction_density_tracks_per_sec']}")


def main():
    args = parse_args()

    local_video_path = args.video
    if not local_video_path:
        local_video_path = input('Enter local path to video (e.g. C:\\path\\video.mp4): ').strip().strip('"')

    # 1) Upload (conditional)
    video_uri = upload_to_gcs(local_video_path, args.bucket, args.prefix)
    print(f"Using URI: {video_uri}")

    # 2) Analyze
    annotations = analyze_video_uri(video_uri, timeout_seconds=args.timeout)

    # 3) Compute derived features
    features = compute_features(annotations, video_uri)

    # 4) Save JSON deliverable
    save_json(features, args.out)

    if args.print_summary:
        print_summary(features)


if __name__ == "__main__":
    main()
