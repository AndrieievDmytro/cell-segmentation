# from PIL import Image
# from pathlib import Path

# input_folder = Path("E:/diploma_proj_latest/cell-segmentation/.data/01_raw/livecell_test")
# for file in input_folder.glob("*.tif"):
#     try:
#         with Image.open(file) as img:
#             print(f"Successfully opened {file.name}")
#     except Exception as e:
#         print(f"Error opening {file.name}: {e}")


# import os

# def check_user():
#     print("Checking user access...")
#     print(f"Current user: {os.getlogin()}")

# if __name__ == "__main__":
#     check_user()


# from pathlib import Path

# def test_folder_access():
#     path = Path("E:/diploma_proj_latest/cell-segmentation/.data/01_raw/livecell_test")
#     try:
#         print(f"Checking folder access for: {path}")
#         if path.exists() and path.is_dir():
#             print(f"Folder exists and is a directory: {path}")
#             for file in path.glob("*.tif"):
#                 print(f"File found: {file}")
#         else:
#             print(f"Folder not accessible: {path}")
#     except PermissionError as e:
#         print(f"Permission error: {e}")
#     except Exception as e:
#         print(f"Unexpected error: {e}")

# if __name__ == "__main__":
#     test_folder_access()


# import os
# from pathlib import Path

# def compare_directories(dir1, dir2):
#     dir1_path = Path(dir1)
#     dir2_path = Path(dir2)

#     print(f"Files in {dir1_path}: {list(dir1_path.iterdir())}")
#     print(f"Files in {dir2_path}: {list(dir2_path.iterdir())}")

#     # Get list of files in each directory (case-insensitive extensions)
#     dir1_files = {file.name: file for file in dir1_path.glob("*.tif")}
#     dir1_files.update({file.name: file for file in dir1_path.glob("*.TIF")})

#     dir2_files = {file.name: file for file in dir2_path.glob("*.tif")}
#     dir2_files.update({file.name: file for file in dir2_path.glob("*.TIF")})

#     # Compare files
#     missing_in_dir2 = dir1_files.keys() - dir2_files.keys()
#     missing_in_dir1 = dir2_files.keys() - dir1_files.keys()

#     common_files = dir1_files.keys() & dir2_files.keys()
#     size_mismatched_files = []

#     for file_name in common_files:
#         if dir1_files[file_name].stat().st_size != dir2_files[file_name].stat().st_size:
#             size_mismatched_files.append(file_name)

#     # Count total files in each directory
#     total_files_dir1 = len(dir1_files)
#     total_files_dir2 = len(dir2_files)

#     # Print results
#     print(f"\nDirectory: {dir1}")
#     print(f"Total number of files: {total_files_dir1}")

#     print(f"\nDirectory: {dir2}")
#     print(f"Total number of files: {total_files_dir2}")

#     print(f"\nFiles in {dir1} but missing in {dir2} ({len(missing_in_dir2)}):")
#     for file_name in missing_in_dir2:
#         print(f"  - {file_name}")

#     print(f"\nFiles in {dir2} but missing in {dir1} ({len(missing_in_dir1)}):")
#     for file_name in missing_in_dir1:
#         print(f"  - {file_name}")

#     print(f"\nFiles present in both but with size mismatch ({len(size_mismatched_files)}):")
#     for file_name in size_mismatched_files:
#         dir1_size = dir1_files[file_name].stat().st_size
#         dir2_size = dir2_files[file_name].stat().st_size
#         print(f"  - {file_name}: {dir1} size={dir1_size}, {dir2} size={dir2_size}")

# if __name__ == "__main__":
#     raw_dir = "E:/diploma_proj_latest/cell-segmentation/.data/01_raw"
#     intermediate_dir = "E:/diploma_proj_latest/cell-segmentation/.data/02_intermediate"

#     compare_directories(raw_dir, intermediate_dir)
