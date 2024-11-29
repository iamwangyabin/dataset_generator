

#show structure
import os

def print_directory_tree(directory, prefix=""):
    contents = os.listdir(directory)
    subdirs = [d for d in contents if os.path.isdir(os.path.join(directory, d))]
    for i, subdir in enumerate(subdirs):
        is_last = (i == len(subdirs) - 1)
        connector = "└── " if is_last else "├── "
        path = os.path.join(directory, subdir)
        file_count = sum([len(files) for r, d, files in os.walk(path)])
        print(f"{prefix}{connector}{subdir} ({file_count} files)")
        extension = "    " if is_last else "│   "
        print_directory_tree(path, prefix + extension)

# directories = ["flux", "pg", "pixart", "pixart_sigma", "sc", "sd15", "sd21", "sd3", "sdxl", "uni"]
directories = ["sd15"]

root = os.getcwd()

for directory in directories:
    path = os.path.join(root, directory)
    if os.path.exists(path):
        print(f"{directory}")
        print_directory_tree(path)
    else:
        print(f"{directory} (directory not found)")
    print()




