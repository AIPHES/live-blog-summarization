import os

def mkdirp(path):
    """Checks if a path exists otherwise creates it
    Each line in the filename should contain a list of URLs separated by comma.
    Args:
        path: The path to check or create
    """
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise