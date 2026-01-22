import os
import shutil

raw_dir = "dataset_raw"
target_dir = "dataset"

classes = {
    "aeromonas": "aeromonas",
    "saprolegnia": "saprolegnia"
}

os.makedirs(target_dir, exist_ok=True)

for folder in os.listdir(raw_dir):
    folder_path = os.path.join(raw_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    # Cari kelas berdasarkan nama folder
    folder_lower = folder.lower()

    target_class = None
    for key in classes:
        if key in folder_lower:
            target_class = classes[key]
            break

    if target_class is None:
        print(f"Skip folder (tidak dikenali): {folder}")
        continue

    class_dir = os.path.join(target_dir, target_class)
    os.makedirs(class_dir, exist_ok=True)

    for img in os.listdir(folder_path):
        src = os.path.join(folder_path, img)
        dst = os.path.join(class_dir, img)
        shutil.copy(src, dst)

    print(f"Folder '{folder}' → '{target_class}' selesai dipindahkan.")
