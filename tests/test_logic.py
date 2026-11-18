"""
Tests of the preprocessing module
"""
import src.preprocessing as preprocessing

import pytest
import numpy as np

@pytest.fixture
def num_values():
    return [1, 2, 3, 4]

# @pytest.mark.parametrize("a, b, expected", [
#     (1, 2, 3),
#     (5, 5, 10),
#     (-1, 1, 0),
# ])
def test_rm_missing_values_with_none_missing(num_values):
    ret_values = preprocessing.rm_missing_values(num_values)
    assert ret_values == num_values

def test_rm_missing_values_with_some_missing(num_values):
    values = [str(v) for v in num_values]
    values.append("")
    ret_values = preprocessing.rm_missing_values(values)
    assert ret_values == values[:-1]
    
def test_fill_missing_values_with_none_missing(num_values):
    ret_values = preprocessing.fill_missing_values(num_values)
    assert ret_values == num_values

def test_fill_missing_values_with_some_missing(num_values):
    values = [str(v) for v in num_values]
    values.append("")
    expected = values.copy()
    ret_values = preprocessing.fill_missing_values(values)
    assert ret_values == values[:-1]
    
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
    assert ret_values == num_values

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
    pass


def test_select_alphanumerical():
    pass


def test_rm_stopwords():
    pass


def test_flatten():
    ret_values = preprocessing.flatten([["a"],["b", "c"],["d"]])
    assert ret_values == ["a", "b", "c", "d"]


def test_shuffle(num_values):
    seed = 42
    ret_values = preprocessing.shuffle(num_values, seed)
    assert ret_values == num_values[::-1]

