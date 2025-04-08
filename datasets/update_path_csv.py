import pandas as pd

def update_video_paths(input_csv_path, output_csv_path, old_path, new_path):
    """
    Update video paths in a CSV file by replacing a specific substring in the paths.

    Parameters:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the modified CSV file.
        old_path (str): The substring in the path to be replaced.
        new_path (str): The new substring to replace the old one.

    Returns:
        None
    """
    try:
        # Load the CSV file
        df = pd.read_csv(input_csv_path, sep="\t", header=None, names=["path", "label"])
    except Exception as e:
        print(f"Error loading the CSV file: {e}")
        return

    # Check for required columns
    if "path" not in df.columns or "label" not in df.columns:
        print("The CSV file does not contain the expected columns ('path', 'label').")
        return

    try:
        # Replace the paths
        df["path"] = df["path"].str.replace(old_path, new_path, regex=False)
    except Exception as e:
        print(f"Error updating the paths: {e}")
        return

    try:
        # Save the modified CSV file
        df.to_csv(output_csv_path, sep="\t", index=False, header=False)
        print(f"Modified file saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving the modified file: {e}")

# Example usage
if __name__ == "__main__":
    #update_train_kinetics400 ; val_kinetics400 ; train_tiny
    input_csv = "val_hmdb51.csv"
    output_csv = "s_val_hmdb51.csv"
    old_path = "/home/ens/Knguetche/hmdb51_video/"
    new_path = "/storage/Knguetche/hmdb51_video/"
	
    update_video_paths(input_csv, output_csv, old_path, new_path)

