"""
Preprocessing module
"""

import re
import numpy as np


def rm_missing_values(values: list) -> list:
    """
    Removes missing values from a list
    """
    values_np = np.array(values)
    mask = np.logical_or(
        np.logical_or(values_np == None, values_np == ""), values_np == "nan"
    )
    return values_np[np.logical_not(mask)].tolist()


def fill_missing_values(values: list, filler: int = 0) -> list:
    """
    Fills missing values from a list
    """
    values_np = np.array(values)
    mask = np.logical_or(
        np.logical_or(values_np == None, values_np == ""), values_np == "nan"
    )
    values_np[mask] = filler
    print(mask, values_np is None, values_np == "", values_np == "nan")
    return values_np.tolist()


def rm_duplicated_values(values: list) -> list:
    """
    Removes duplicated values from a list
    """
    values_np = np.array(values)
    return np.unique(values_np).tolist()


def min_max_normalization(
    values: list, new_min: float = 0.0, new_max: float = 1.0
) -> list:
    """
    Normalizes the values from a list
    """
    values_np = np.array(values)
    old_min = values_np.min()
    old_max = values_np.max()
    normalized_0_1 = (values - old_min) / (old_max - old_min)
    normalized = (normalized_0_1 * (new_max - new_min)) + new_min
    return normalized.tolist()


def standarization(values: list) -> list:
    """
    Standarizes the values from a list
    """
    values_np = np.array(values)
    standarized = (values_np - values_np.mean()) / values_np.std()
    return standarized.tolist()


def clip_values(values: list, min_v: float, max_v: float) -> list:
    """
    Clips the values from a list
    """
    values_np = np.array(values)
    values_np[values_np < min_v] = min_v
    values_np[values_np > max_v] = max_v
    return values_np.tolist()


def values_to_ints(values: list) -> list:
    """
    Changes the values from a list to ints
    """
    values_np = np.array(values)
    values_np = values_np.astype(int)
    return values_np.tolist()


def log_scale(values: list) -> list:
    """
    Changes the scale of the values from a list to logarithmic
    """
    values_np = np.array(values)
    return np.log(values_np).tolist()


def tokenize(text: str) -> str:
    """
    Tokenizes a text
    """
    text = re.sub(r"[^0-9a-z]", "", text.lower())
    return text


def select_alphanumerical(text: str) -> str:
    """
    Deletes non-alphanumerical characters of a text
    """
    text = re.sub(r"[^0-9a-z\s]", "", text)
    return text


def rm_stopwords(text: str, stopwords: list[str]) -> str:
    """
    Removes the given stopwords from a text
    """
    for stopword in stopwords:
        pattern = re.compile(r"\b" + stopword + r"\b")
        text = pattern.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text


def flatten(list_of_lists: list[list]) -> list:
    """
    Flattens a list of lists
    """
    flatenned = []
    for l in list_of_lists:
        flatenned.extend(l)
    return flatenned


def shuffle(values: list, seed: int) -> list:
    """
    Shuffles the values from a list
    """
    values_np = np.array(values)
    rng = np.random.default_rng(seed)
    rng.shuffle(values_np)
    return values_np.tolist()
