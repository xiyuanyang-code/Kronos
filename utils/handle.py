import os
import re
import shutil


def consolidate_folder_contents_by_pattern(
    source_directory="/GPFS/rhome/xiyuanyang/Quan/Factor_Mining/out/tensorboard/",
    destination_folder_name="/GPFS/rhome/xiyuanyang/Quan/Factor_Mining/out/tensorboard/1_2025-07-17-00_2",
):
    """
    Finds directories matching the pattern '1_{number}_2025-07-17-00' in the source directory,
    then moves the *contents* of these matching directories into a single new destination folder.
    After moving contents, the empty matched directories are removed.

    Args:
        source_directory (str): The directory to search for subdirectories. Defaults to the current directory.
        destination_folder_name (str): The name of the folder where all consolidated contents will be moved.
    """

    # Create the destination folder if it doesn't exist
    destination_path = os.path.join(source_directory, destination_folder_name)
    os.makedirs(destination_path, exist_ok=True)

    # Define the regular expression pattern for directory names
    pattern = re.compile(r"1_(\d+)_2025-07-17-00_2")
    # 1_18030_2025-07-17-00_1
    # 1_2025-07-16-09_single_mode_
    # 1_2025-07-17-12_single_mode
    # 1_2025-07-16-11_single_mode_
    # 1_2025-07-17-13_single_mode
    #   1_24513_2025-07-17-10_1
    # 1_21264_2025-07-17-00_2
    # pattern = re.compile(r"1_2025-07-17-13_single_mode_(\d+)")

    print(
        f"Searching for directories matching '1_<number>_2025-07-17-10' in '{os.path.abspath(source_directory)}'..."
    )
    print(f"Contents will be moved to '{os.path.abspath(destination_path)}'\n")

    found_directories_count = 0
    moved_items_count = 0
    removed_empty_dirs_count = 0

    # Iterate over all items (files and directories) in the source directory
    for item_name in os.listdir(source_directory):
        item_path = os.path.join(source_directory, item_name)

        # Check if it's a directory and matches our pattern
        if os.path.isdir(item_path) and pattern.search(item_name):
            print(f"Found matching directory: '{item_name}'")
            found_directories_count += 1

            # Move contents of the matched directory to the destination folder
            for content_item in os.listdir(item_path):
                source_content_path = os.path.join(item_path, content_item)
                destination_content_path = os.path.join(destination_path, content_item)

                try:
                    # If an item with the same name already exists in the destination,
                    # you might want to handle conflicts (e.g., rename, overwrite, skip).
                    # For simplicity, this script will try to move; if the name clashes
                    # and it's a file, it might overwrite. For directories, it will error
                    # if target directory exists and isn't empty.
                    # Consider adding more sophisticated conflict resolution if needed.

                    shutil.move(source_content_path, destination_content_path)
                    print(
                        f"  Moved '{content_item}' from '{item_name}/' to '{destination_folder_name}/'"
                    )
                    moved_items_count += 1
                except shutil.Error as e:
                    print(
                        f"  Warning: Could not move '{content_item}' from '{item_name}/' to '{destination_folder_name}/'. "
                        f"Reason: {e}. It might already exist or be a non-empty directory."
                    )
                except Exception as e:
                    print(f"  Error moving '{content_item}' from '{item_name}/': {e}")

            # After moving contents, remove the now empty source directory
            try:
                os.rmdir(item_path)
                print(f"  Removed empty directory: '{item_name}'")
                removed_empty_dirs_count += 1
            except OSError as e:
                print(
                    f"  Error removing directory '{item_name}': {e} (It might not be empty)"
                )
            except Exception as e:
                print(
                    f"  An unexpected error occurred while removing '{item_name}': {e}"
                )

    if found_directories_count == 0:
        print("No matching directories found.")
    else:
        print(f"\n--- Consolidation Summary ---")
        print(f"Processed {found_directories_count} matching directorie(s).")
        print(f"Moved {moved_items_count} item(s) into '{destination_folder_name}'.")
        print(f"Removed {removed_empty_dirs_count} empty source directorie(s).")
        print(
            f"Consolidation complete. All relevant contents are now in '{os.path.abspath(destination_path)}'."
        )


if __name__ == "__main__":
    consolidate_folder_contents_by_pattern()
