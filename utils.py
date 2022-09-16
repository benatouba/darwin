from glob import glob
from constants import basepath


def normalize_whitespace(text):
    """Remove redundant whitespace from a string."""
    return " ".join(text.split())


def isnt_str_or_basestring(obj):
    """
    check if the passed in object is a str
    """

    return isinstance(obj, str)


def remove_nonalphanumerics(string: str) -> str:
    return "".join(ch for ch in string if ch.isalnum())


def transform_k_to_c(ds):
    return ds[ds.VARNAME].data - 273.15


def glob_from_str(*args, **kwargs):
    return glob(*args, **kwargs)


def get_basename(path, suffix=False):
    name = path.split("/")[-1]
    return name if suffix else name.split(".")[0]


def glob_files(path, *args, **kwargs):
    if type(path) == str:
        return glob_from_str(pathname=path, *args, **kwargs)
    if not path.is_absolute():
        path = basepath / path
    return list(path.glob(*args, **kwargs))
