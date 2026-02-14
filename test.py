from google.cloud import videointelligence_v1 as vi
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Kim\OneDrive - Afeka College Of Engineering\Desktop\פרויקט גמר חדש\FinalProject\FinalProject\.venv\green-calling-481021-r2-23cfc870ab9c.json"

VIDEO_URI = "gs://final-project-bucket1/ACCEDE09230.mp4"

def main():
    client = vi.VideoIntelligenceServiceClient()

    features = [
        vi.Feature.LABEL_DETECTION,
        vi.Feature.OBJECT_TRACKING,
        vi.Feature.SHOT_CHANGE_DETECTION,
    ]

    print("Sending video for analysis...")
    operation = client.annotate_video(
        request={
            "input_uri": VIDEO_URI,
            "features": features,
        }
    )

    print("Waiting for results...")
    result = operation.result(timeout=600) #600 seconds/milliseconds
    annotations = result.annotation_results[0]

    print("\n=== Scene Shots ===")
    for shot in annotations.shot_annotations:
        print(
            shot.start_time_offset.total_seconds(),
            "->",
            shot.end_time_offset.total_seconds()
        )

    print("\n=== Labels ===")
    for label in annotations.segment_label_annotations[:10]:
        print(label.entity.description)

    print("\n=== Objects ===")
    for obj in annotations.object_annotations[:10]:
        print(
            obj.entity.description,
            obj.segment.start_time_offset.ToNanoseconds(),
            "->",
            obj.segment.end_time_offset.ToNanoseconds(),
            "confidence:",
            obj.confidence #How sure the AI is at telling this detection (0-1)
        )

if __name__ == "__main__":
    main()
