import os
import argparse
import pandas as pd

OUTPUT_CSV_NAME = "finetune_dataset.csv"

# Function to create CSV rows for Cataract Phase Recognition Fine-tuning
def process_cataract(cataract_video_dir, cataract_annotations_dir, output_dir):
    """
    Processes Cataract dataset including annotations for phase recognition and generates CSV file with video paths, annotation and metadata.
    """
    rows = []

    for case_folder in os.listdir(cataract_annotations_dir):
        case_path = os.path.join(cataract_annotations_dir, case_folder)
        if not os.path.isdir(case_path):
            continue

        # Define paths for annotation files
        phase_csv = os.path.join(case_path, f"{case_folder}_annotations_phases.csv")
        video_csv = os.path.join(case_path, f"{case_folder}_video.csv")

        # Check if required annotation files exist
        if not os.path.exists(phase_csv):
            print(f"Warning: Missing phase annotations for {case_folder}. Skipping.")
            continue
        if not os.path.exists(video_csv):
            print(f"Warning: Missing video metadata for {case_folder}. Skipping.")
            continue

        # Load video metadata to get FPS
        video_metadata = pd.read_csv(video_csv)
        try:
            fps = float(video_metadata["fps"].iloc[0])  # Ensure FPS is correctly read
        except (KeyError, ValueError, IndexError) as e:
            print(f"Warning: Error reading FPS for {case_folder}. Skipping. Error: {e}")
            continue

        # Load phase annotations
        phase_data = pd.read_csv(phase_csv)

        # Debugging: Ensure 'frame' column exists
        if "frame" not in phase_data.columns:
            print(f"Error: Expected 'frame' column missing in {phase_csv}")
            print(f"Available columns: {list(phase_data.columns)}")
            continue

        # Construct the video path
        video_path = os.path.abspath(os.path.join(cataract_video_dir, f"{case_folder}.mp4"))

        if not os.path.exists(video_path):
            print(f"Warning: Video {video_path} not found for {case_folder}. Skipping.")
            continue

        # Add rows for each phase annotation
        for _, row in phase_data.iterrows():
            try:
                start_frame = int(row["frame"])  # Fixed column name
                total_frames = -1  # Placeholder for full video

                # Append data
                rows.append([
                    os.path.abspath(video_path),
                    start_frame,
                    total_frames,
                    -1  # Placeholder label (can be updated if needed)
                ])
            except (ValueError, KeyError) as e:
                print(f"Warning: Error processing phase data for {case_folder}. Skipping row. Error: {e}")
                continue

    # Define output CSV path
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, OUTPUT_CSV_NAME)

    # Save data to CSV
    df = pd.DataFrame(rows, columns=["video_path", "start_index", "total_frames", "label"])
    df.to_csv(output_csv_path, index=False, sep=' ', header=False)
    print(f"CSV file created: {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cataract_video_dir",
        type=str,
        required=True,
        help="Path to the directory containing all Cataract trimmed phase videos."
    )
    parser.add_argument(
        "--cataract_annotations_csv",
        type=str,
        required=True,
        help="Path to the Cataract CSV file containing video phase annotations."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the output CSV file will be saved."
    )
    
    args = parser.parse_args()
    
    process_cataract(args.cataract_video_dir, args.cataract_annotations_csv, args.output_dir)