#!/usr/bin/env python3
import os

def list_files_by_size(root_path, output_file="file_sizes.txt"):
    file_info = []

    # Walk through the directory tree
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                size = os.path.getsize(file_path)
                file_info.append((file_path, size))
            except (OSError, PermissionError):
                # Skip files that cannot be accessed
                continue

    # Sort files by size (descending)
    file_info.sort(key=lambda x: x[1], reverse=True)

    # Write results to file
    with open(output_file, "w") as f:
        for path, size in file_info:
            f.write(f"{path} | {size} bytes\n")

    print(f"Done! Results written to {output_file}")

   
list_files_by_size("/cluster/home/kristiac/")
