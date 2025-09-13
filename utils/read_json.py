import os
import json
import glob
from collections import defaultdict
from typing import Optional


def read_single_mode(save_path: str):
    """
    Reads and merges JSON files from a specified directory.
    Files are expected to be named '{index}_{steps}_steps_pool.json'.
    It groups files by 'steps', sorts them by 'index', and merges 'exprs' and 'weights'.
    Merged data is saved to a new 'integrate' subdirectory as '{steps}_steps_pool.json'.

    Args:
        save_path (str): The directory containing the JSON files.
    """
    integrate_dir = os.path.join(save_path, "integrate")
    os.makedirs(integrate_dir, exist_ok=True)
    print(f"Created/Ensured directory: {integrate_dir}")

    grouped_files = defaultdict(list)

    # Search for files matching the pattern {index}_{steps}_steps_pool.json
    search_pattern = os.path.join(save_path, "*_steps_pool.json")
    all_json_files = glob.glob(search_pattern)

    print(f"Found {len(all_json_files)} potential JSON files in {save_path}.")

    for filepath in all_json_files:
        filename = os.path.basename(filepath)
        # Attempt to parse filename: {index}_{steps}_steps_pool.json
        # E.g., "0_1000_steps_pool.json"

        # Remove "_steps_pool.json" suffix and then split from the right by '_'
        name_without_suffix = filename.replace("_steps_pool.json", "")
        parts = name_without_suffix.rpartition("_")

        if len(parts) == 3 and parts[1] == "_":
            try:
                index = int(parts[0])
                steps = int(parts[2])

                # ! 这里使用defaultdict本质是用字典的键值对来存储，键是特定的迭代更新轮数，值是那个迭代更新轮数下的挖出来的所有单因子的汇总（一个列表，或者说是一个元组）
                grouped_files[steps].append((index, filepath))
                print(f"  Parsed file: {filename} -> index={index}, steps={steps}")
            except ValueError:
                print(f"  Skipping file with invalid index or steps format: {filename}")
        else:
            print(
                f"  Skipping file not matching expected pattern '{{index}}_{{steps}}_steps_pool.json': {filename}"
            )

    if not grouped_files:
        print(
            "No files matching the specified pattern and filters were found for merging."
        )
        return

    # Process each group of files (by steps)
    for steps, files_for_steps in grouped_files.items():
        print(f"\nProcessing files for steps: {steps}")
        # Sort by index to maintain the order as per your request

        # 添加去重逻辑
        files_for_steps.sort(key=lambda x: x[0])

        merged_exprs = []
        merged_weights = []
        expr_set = set()

        for index, filepath in files_for_steps:
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    exprs = data.get("exprs", [])
                    weights = data.get("weights", [])
                    for expr, weight in zip(exprs, weights):
                        # 用json.dumps保证表达式唯一性（适用于dict或复杂结构）
                        expr_key = json.dumps(expr, sort_keys=True)
                        if expr_key in expr_set:
                            print(
                                f"  Warning: Duplicate expr found and removed: {expr}"
                            )
                            continue
                        expr_set.add(expr_key)
                        merged_exprs.append(expr)
                        merged_weights.append(weight)
                print(
                    f"  Successfully loaded and merged data from: {os.path.basename(filepath)}"
                )
            except json.JSONDecodeError as e:
                print(
                    f"  Error decoding JSON from {os.path.basename(filepath)}: {e}. Skipping file."
                )
            except FileNotFoundError:
                print(f"  File not found: {os.path.basename(filepath)}. Skipping file.")
            except Exception as e:
                print(
                    f"  An unexpected error occurred while processing {os.path.basename(filepath)}: {e}. Skipping file."
                )

        if merged_exprs and merged_weights:
            output_data = {"exprs": merged_exprs, "weights": merged_weights}
            output_filename = os.path.join(integrate_dir, f"{steps}_steps_pool.json")
            with open(output_filename, "w") as f:
                json.dump(output_data, f, indent=4)
            print(f"Merged data for steps {steps} written to {output_filename}")
        else:
            print(f"No valid data merged for steps {steps}. Skipping output file.")


if __name__ == "__main__":
    read_single_mode("/GPFS/rhome/xiyuanyang/Quan/Factor_Mining/out/results/10_2025-07-17-21_single_mode")
    # read_single_mode("/GPFS/rhome/xiyuanyang/Quan/Factor_Mining/out/results/10_2025")

