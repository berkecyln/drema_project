import os
import shutil
import glob
import datetime


def merge_input_folders(input_folders, output_folder):
    subfolders = ["images", "depth", "depth_scaled", "poses", "object_mask"]
    os.makedirs(output_folder, exist_ok=True)
    for sub in subfolders:
        os.makedirs(os.path.join(output_folder, sub), exist_ok=True)

    idx = 1
    for folder in input_folders:
        print(f"Merging from: {folder}")
        img_dir = os.path.join(folder, "images")
        if not os.path.exists(img_dir):
            print(f"  Skipping missing {img_dir}")
            continue

        img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
        for img_path in img_files:
            ext = os.path.splitext(img_path)[1]
            shutil.copy2(img_path, os.path.join(output_folder, "images", f"{idx:04d}{ext}"))

            base_idx = os.path.splitext(os.path.basename(img_path))[0]
            for sub, sub_ext in [("depth", ".npy"), ("depth_scaled", ".npy"), ("poses", ".txt")]:
                src = os.path.join(folder, sub, f"{base_idx}{sub_ext}")
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(output_folder, sub, f"{idx:04d}{sub_ext}"))

            obj_mask_src = os.path.join(folder, "object_mask", f"{base_idx}.png")
            if os.path.exists(obj_mask_src):
                shutil.copy2(obj_mask_src, os.path.join(output_folder, "object_mask", f"{idx:04d}.png"))

            idx += 1

    print(f"Merge complete. Output: {output_folder}")


if __name__ == "__main__":
    # Adjust these paths to your input folders
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_folders = [
        os.path.join(project_dir, "input/line_scan"),
        os.path.join(project_dir, "input/orbit"),
    ]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(project_dir, f"input/merged_{timestamp}")
    merge_input_folders(input_folders, output_folder)
