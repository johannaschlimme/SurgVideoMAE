import os
import argparse
import pandas as pd

# Generic filename for the output CSV
OUTPUT_CSV_NAME = "pretrain_dataset.csv"

def process_ophnet(ophnet_video_dir, ophnet_annotations_csv, output_dir):
    """
    Processes the OphNet dataset and generates a CSV file with video paths and metadata.
    """
    rows = []

    # Load annotation CSV
    loca_data = pd.read_csv(ophnet_annotations_csv)

    for _, row in loca_data.iterrows():
        video_id = row["video_id"]
        start_time = row["start"]
        end_time = row["end"]

        # Skip rows with missing or invalid start/end times
        if pd.isna(start_time) or pd.isna(end_time):
            print(f"Warning: Missing start or end time for video {video_id}. Skipping row.")
            continue

        # Set total_frames to -1 for raw video
        total_frames = -1

        # Handle multiple videos per case folder
        case_folder = os.path.join(ophnet_video_dir, video_id)
        if os.path.exists(case_folder) and os.path.isdir(case_folder):
            for video_file in os.listdir(case_folder):
                if video_file.endswith(".mp4"):
                    video_path = os.path.abspath(os.path.join(case_folder, video_file))
                    rows.append([video_path, 0, total_frames, -1])
        else:
            print(f"Warning: Folder {case_folder} not found in OphNet dataset.")

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
        "--ophnet_video_dir",
        type=str,
        required=True,
        help="Path to the directory containing all OphNet trimmed phase videos."
    )
    parser.add_argument(
        "--ophnet_annotations_csv",
        type=str,
        required=True,
        help="Path to the OphNet CSV file containing video phase annotations."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the output CSV file will be saved."
    )
    
    args = parser.parse_args()
    
    process_ophnet(args.ophnet_video_dir, args.ophnet_annotations_csv, args.output_dir)