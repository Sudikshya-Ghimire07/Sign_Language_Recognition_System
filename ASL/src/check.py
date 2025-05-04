import os

# Root folder that contains subfolders A-Z, nothing, space, del
root_folder = r"E:\Sign_Language_Recognition_System\ASL\input\asl_alphabet_train\asl_alphabet_train"

# Include extra folders
extra_folders = {"nothing", "space", "del"}

# Loop over folders
for folder in sorted(os.listdir(root_folder)):
    folder_path = os.path.join(root_folder, folder)

    # Process A-Z or explicitly allowed folders
    if os.path.isdir(folder_path) and (
        (len(folder) == 1 and folder.isalpha()) or folder.lower() in extra_folders
    ):
        folder_label = folder.upper() if len(folder) == 1 else folder.lower()
        print(f"\n📂 Processing folder: {folder}")

        # Get all image files
        files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        # Find existing valid-format files
        existing_numbers = set()
        for f in files:
            name, ext = os.path.splitext(f)
            if name.startswith(folder_label) and name[len(folder_label):].isdigit():
                existing_numbers.add(int(name[len(folder_label):]))

        count = 1
        for file in sorted(files):
            old_path = os.path.join(folder_path, file)
            name, ext = os.path.splitext(file)

            # Skip if already correctly named
            if name.startswith(folder_label) and name[len(folder_label):].isdigit():
                continue

            # Find next unused number
            while count in existing_numbers:
                count += 1

            new_filename = f"{folder_label}{count}{ext.lower()}"
            new_path = os.path.join(folder_path, new_filename)

            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"✅ Renamed: {file} → {new_filename}")
                existing_numbers.add(count)
                count += 1
            else:
                print(f"⏩ Skipped (exists): {new_filename}")
