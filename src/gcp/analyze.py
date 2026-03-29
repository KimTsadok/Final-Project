# src/gcp/analyze.py
"""
this file knows how to:
* connect to Video Intelligence
* request the relevant features
* return the annotation result
"""

from google.cloud import videointelligence_v1 as vi


def analyze_video_uri(video_uri: str, timeout_seconds: int = 600):
    """
    Send a GCS video URI to Google Video Intelligence and return the
    first annotation result.
    """
    client = vi.VideoIntelligenceServiceClient()

    features = [
        vi.Feature.LABEL_DETECTION,
        vi.Feature.OBJECT_TRACKING,
        vi.Feature.SHOT_CHANGE_DETECTION,
    ]

    print("\nSending video for analysis...")
    operation = client.annotate_video(
        request={
            "input_uri": video_uri,
            "features": features,
        }
    )

    print("Waiting for results...")
    result = operation.result(timeout=timeout_seconds)
    return result.annotation_results[0]