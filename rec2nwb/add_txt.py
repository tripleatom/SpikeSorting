import os
import glob
import argparse

def create_txt_for_rec_files(folder):
    rec_files = glob.glob(os.path.join(folder, "*", "*.rec"))
    if not rec_files:
        print(f"No .rec files found in {folder}")
        return
    for rec_path in rec_files:
        txt_path = rec_path + ".txt"
        if not os.path.exists(txt_path):
            open(txt_path, "w").close()
            print(f"Created: {txt_path}")
        else:
            print(f"Already exists: {txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create .txt files for each .rec file in a folder.")
    parser.add_argument("folder", help="Path to the folder to search")
    args = parser.parse_args()
    create_txt_for_rec_files(args.folder)
