import os
import argparse
import pandas as pd

OUTPUT_CSV_NAME = "finetune_dataset.csv"

def process_cataract(cataract_video_dir, cataract_annotations_dir, output_dir):
    """
    Processes Cataract dataset annotations for phase recognition and outputs a CSV 
    with video paths, start index, duration, and label for fine-tuning.
    """
    rows = []

    for case_folder in os.listdir(cataract_annotations_dir):
        case_path = os.path.join(cataract_annotations_dir, case_folder)
        if not os.path.isdir(case_path):
            continue

        # Define annotation files
        phase_csv = os.path.join(case_path, f"{case_folder}_annotations_phases.csv")
        video_csv = os.path.join(case_path, f"{case_folder}_video.csv")

        # Check if necessary files exist
        if not os.path.exists(phase_csv):
            print(f"Warning: Missing phase annotations for {case_folder}. Skipping.")
            continue
        if not os.path.exists(video_csv):
            print(f"Warning: Missing video metadata for {case_folder}. Skipping.")
            continue

        # Read FPS from video metadata
        try:
            video_metadata = pd.read_csv(video_csv)
            fps = float(video_metadata["fps"].iloc[0])
        except Exception as e:
            print(f"Warning: Failed to read FPS from {video_csv}: {e}. Skipping.")
            continue

        # Read phase annotations
        try:
            phase_data = pd.read_csv(phase_csv)
        except Exception as e:
            print(f"Warning: Failed to read phase data from {phase_csv}: {e}. Skipping.")
            continue

        if "frame" not in phase_data.columns or "endFrame" not in phase_data.columns or "comment" not in phase_data.columns:
            print(f"Error: Missing required columns in {phase_csv}. Skipping.")
            continue

        # Construct video path
        video_filename = f"{case_folder}.mp4"
        video_path = os.path.abspath(os.path.join(cataract_video_dir, video_filename))

        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}. Skipping.")
            continue

        # Extract phase segments
        for _, row in phase_data.iterrows():
            try:
                start_frame = int(row["frame"])
                end_frame = int(row["endFrame"])
                label = str(row["comment"]).strip().lower().replace(" ", "_")

                if end_frame <= start_frame:
                    continue

                rows.append([
                    video_path,
                    start_frame,
                    end_frame,
                    label
                ])
            except Exception as e:
                print(f"Warning: Error parsing annotation row in {case_folder}: {e}")
                continue

    # Save to output CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, OUTPUT_CSV_NAME)
    df = pd.DataFrame(rows, columns=["video_path", "start_frame", "end_frame", "label"])
    df.to_csv(output_path, sep=' ', header=False, index=False)
    print(f"Preprocessing complete. CSV saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cataract_video_dir",
        type=str,
        required=True,
        help="Path to directory containing the video files (e.g. case_4687.mp4)"
    )
    parser.add_argument(
        "--cataract_annotations_dir",
        type=str,
        required=True,
        help="Path to annotation folders containing per-case CSVs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the generated CSV"
    )

    args = parser.parse_args()
    process_cataract(args.cataract_video_dir, args.cataract_annotations_dir, args.output_dir)