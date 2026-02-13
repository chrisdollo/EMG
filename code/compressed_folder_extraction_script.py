import os
import zipfile
import shutil

# =====================
# EDIT PATHS HERE
# =====================
HEAD_FOLDER = r"E:/Chris/EMG/Data/raw data"          # folder that contains the outer zip files
OUTPUT_FOLDER = r"E:/Chris/EMG/Data/cvs_data_per_subject"  # where you want ALL csv files to end up

def extract_all_zips(root):
    """
    Keep extracting zip files until none remain.
    Handles nested zips automatically.
    """
    found_zip = True

    while found_zip:
        found_zip = False

        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith(".zip"):
                    zip_path = os.path.join(dirpath, name)
                    extract_folder = os.path.join(dirpath, name[:-4])

                    os.makedirs(extract_folder, exist_ok=True)

                    print("Extracting:", zip_path)

                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extractall(extract_folder)

                    os.remove(zip_path)
                    found_zip = True


def move_all_csv(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".csv"):
                src = os.path.join(dirpath, name)
                dst = os.path.join(out_dir, name)

                shutil.move(src, dst)
                count += 1

    print(f"Moved {count} CSV files.")


def main():
    extract_all_zips(HEAD_FOLDER)
    move_all_csv(HEAD_FOLDER, OUTPUT_FOLDER)
    print("DONE")


if __name__ == "__main__":
    main()
