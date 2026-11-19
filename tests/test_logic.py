"""
Tests of the preprocessing module
"""
import src.preprocessing as preprocessing

import pytest
import numpy as np

@pytest.fixture
def num_values():
    return [1, 2, 3, 4]

def test_rm_missing_values_with_none_missing(num_values):
    ret_values = preprocessing.rm_missing_values(num_values)
    assert ret_values == num_values

@pytest.mark.parametrize("values, expected", [
    (["1", None, None, "4"], ["1", "4"]),
    (["", "2", "3", ""], ["2", "3"]),
    (["1", "2", None, ""], ["1", "2"]),
])
def test_rm_missing_values_with_some_missing(values, expected):
    ret_values = preprocessing.rm_missing_values(values)
    assert ret_values == expected
    
def test_fill_missing_values_with_none_missing(num_values):
    ret_values = preprocessing.fill_missing_values(num_values)
    assert ret_values == num_values

def test_fill_missing_values_with_some_missing(num_values):
    values = [str(v) for v in num_values]
    values.append("")
    expected = values[:-1].copy()
    expected.append("0")
    ret_values = preprocessing.fill_missing_values(values)
    assert ret_values == expected
    
def test_rm_duplicated_values_with_none_duplicated(num_values):
    ret_values = preprocessing.rm_duplicated_values(num_values)
    assert ret_values == num_values

def test_rm_duplicated_values_with_some_duplicated(num_values):
    values = num_values.copy()
    values.append(num_values[2])
    ret_values = preprocessing.rm_duplicated_values(values)
    assert ret_values == num_values

def test_min_max_normalization(num_values):
    ret_values = preprocessing.min_max_normalization(num_values, 0, 6)
    assert ret_values == [0, 2, 4, 6]

def test_standarization(num_values):
    ret_values = preprocessing.standarization(num_values)
    assert ret_values == pytest.approx([-1.34164079, -0.4472136, 0.4472136, 1.34164079])

@pytest.mark.parametrize("min_v, max_v, expected", [
    (2, 3, [2, 2, 3, 3]),
    (5, 10, [5]*4),
    (1, 5, [1, 2, 3, 4]),
])
def test_clip_values(num_values, min_v, max_v, expected):
    ret_values = preprocessing.clip_values(num_values, min_v, max_v)
    assert ret_values == expected


def test_values_to_ints(num_values):
    values = [str(v) for v in num_values]
    ret_values = preprocessing.values_to_ints(values)
    assert ret_values == num_values


def test_log_scale(num_values):
    ret_values = preprocessing.log_scale(num_values)
    assert ret_values == [np.log(v) for v in num_values]


def test_tokenize():
    ret_values = preprocessing.tokenize("Str That Needs To Be Tokenized!")
    assert ret_values == "strthatneedstobetokenized"


def test_select_alphanumerical():
    ret_values = preprocessing.select_alphanumerical("Str with **many!** symbols & characters?")
    assert ret_values == "Str with many symbols  characters"


def test_rm_stopwords():
    ret_values = preprocessing.rm_stopwords("string of the stopwords to remove", ["of","a","the","to"])
    assert ret_values == "string stopwords remove"


def test_flatten():
    ret_values = preprocessing.flatten([["a"],["b", "c"],["d"]])
    assert ret_values == ["a", "b", "c", "d"]


def test_shuffle(num_values):
    seed = 42
    ret_values = preprocessing.shuffle(num_values, seed)
    assert ret_values == num_values[::-1]

