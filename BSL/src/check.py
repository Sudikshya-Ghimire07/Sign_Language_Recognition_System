import os

# Root folder that contains subfolders A-Z
root_folder = r"E:\Sign_Language_Recognition_System\BSL\input\train"

# Loop over folders A-Z
for folder in sorted(os.listdir(root_folder)):
    folder_path = os.path.join(root_folder, folder)

    if os.path.isdir(folder_path) and len(folder) == 1 and folder.isalpha():
        letter = folder.upper()
        print(f"\n📂 Processing folder: {folder}")

        # Get all files in the folder
        files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        # Find existing A1, A2... format files
        existing_numbers = set()
        for f in files:
            name, ext = os.path.splitext(f)
            if name.startswith(letter) and name[1:].isdigit():
                existing_numbers.add(int(name[1:]))

        count = 1
        for file in sorted(files):
            old_path = os.path.join(folder_path, file)
            name, ext = os.path.splitext(file)

            # Skip if file is already in correct format
            if name.startswith(letter) and name[1:].isdigit():
                continue

            # Find next available number
            while count in existing_numbers:
                count += 1

            new_filename = f"{letter}{count}{ext.lower()}"
            new_path = os.path.join(folder_path, new_filename)

            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"✅ Renamed: {file} → {new_filename}")
                existing_numbers.add(count)
                count += 1
            else:
                print(f"⏩ Skipped (exists): {new_filename}")
