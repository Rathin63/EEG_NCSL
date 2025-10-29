"""
File System Utilities Module

This module provides utility functions for file system operations.

Functions:
    create_folder: Create a directory if it doesn't exist
    create_folders: Create a directory and all its parent directories if they don't exist
"""

from os import makedirs
from os.path import exists, dirname


def create_folder(directory):
    """
    Create a folder if it does not already exist.

    This function checks if a directory exists at the specified path.
    If it doesn't exist, it creates the directory (including any necessary
    parent directories). If it already exists, it simply reports this fact.

    Parameters
    ----------
    directory : str
        The path of the directory to be created. Can be either absolute
        or relative path.

    Returns
    -------
    None

    Side Effects
    ------------
    - Creates directory on file system if it doesn't exist
    - Prints status message to stdout

    Examples
    --------
    ### create_folder("data/processed")
    Directory data/processed created successfully.

    ### create_folder("data/processed")  # Called again
    Directory data/processed already exists.

    Notes
    -----
    - Uses os.makedirs which creates intermediate directories if needed
    - Safe to call multiple times on the same directory
    """
    if not exists(directory):
        makedirs(directory)
        print(f"Directory {directory} created successfully.")
    else:
        print(f"Directory {directory} already exists.")


def create_folders(path):
    """
    Create a directory path and all parent directories if they don't exist.

    This function ensures that an entire directory path exists, creating
    all parent directories as needed. It's essentially an enhanced version
    of create_folder that explicitly handles deep directory structures.

    Parameters
    ----------
    path : str
        The full path to create. Can be a deep nested path like
        "data/processed/2024/january/reports"

    Returns
    -------
    bool
        True if any directories were created, False if all already existed

    Side Effects
    ------------
    - Creates all directories in the path if they don't exist
    - Prints status message for each created directory

    Examples
    --------
    ### # Create a deep nested structure
    ### create_folders("data/processed/2024/january")
    Directory data created successfully.
    Directory data/processed created successfully.
    Directory data/processed/2024 created successfully.
    Directory data/processed/2024/january created successfully.

    ### # Run again - all exist
    ### create_folders("data/processed/2024/january")
    Directory data/processed/2024/january already exists.

    ### # Create path for a file (creates parent directories)
    ### create_folders("output/figures/exp1/results.png")
    # Creates: output/, output/figures/, output/figures/exp1/

    ### # Works with absolute paths too
    ### create_folders("/home/user/projects/ml/data/raw")

    Notes
    -----
    - If the path appears to be a file (has an extension), creates parent directories only
    - Handles both forward slashes (/) and backslashes (\\)
    - Safe to call multiple times on the same path
    - More explicit than create_folder about creating parent directories
    """
    created_any = False

    # Check if this might be a file path (has an extension)
    # If so, get the directory part only
    if '.' in path.split('/')[-1] or '.' in path.split('\\')[-1]:
        # Likely a file path, get directory
        directory = dirname(path)
        if directory and directory != '':
            path = directory
        else:
            # No directory part
            return False

    # Build up the path incrementally to show each created directory
    parts = path.replace('\\', '/').split('/')
    current_path = ''

    for i, part in enumerate(parts):
        if not part:  # Skip empty parts (e.g., from double slashes)
            continue

        if i == 0 and ':' in part:  # Windows drive letter
            current_path = part
        elif i == 0 and path.startswith('/'):  # Absolute Unix path
            current_path = '/' + part
        else:
            current_path = current_path + '/' + part if current_path else part

        if not exists(current_path):
            makedirs(current_path, exist_ok=True)
            print(f"Directory {current_path} created successfully.")
            created_any = True

    if not created_any:
        print(f"Directory {path} already exists.")

    return created_any

if __name__ == "__main__":
    create_folders("C:\\Users\\adaraie\\Desktop\\NCSL_Desk\\Prediction\\Codes\\Python\\11\\22\\33")