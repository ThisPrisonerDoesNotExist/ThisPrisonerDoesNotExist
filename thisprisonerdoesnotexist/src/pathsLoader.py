import os


def load_paths(directory: str) -> list:
    """
    Loads the paths of all files in the given directory.

    Args:
    - directory (str): the directory to load the paths from.

    Returns:
    - list: a list of paths to the files in the given directory.
    """
    paths = os.listdir(directory)
    paths = [os.path.join(directory, i) for i in paths]
    return paths


def load_paths_prisoners_dataset() -> list:
    """
    Loads the paths of all images in the prisoners dataset.

    Args:
    - None

    Returns:
    - list: a list of paths to the images in the prisoners dataset.
    """
    return load_paths("Prisonersi/front/")
