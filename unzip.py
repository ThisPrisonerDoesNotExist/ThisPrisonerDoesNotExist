import os
import zipfile


def unzip_file(zip_file_path: str, extract_path: str) -> None:
    """
    Unzips the given file to the given path.

    Args:
    - zip_file_path (str): the path to the zip file.
    - extract_path (str): the path to extract the zip file to.

    Returns:
    - None
    """
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
