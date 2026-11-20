"""
Tests of the cli module
"""
from src.cli import cli

import pytest
import numpy as np
from click.testing import CliRunner

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def num_values():
    return ["1", "2", "3", "4"]
    
def test_rm_missing_values_with_none_missing(runner, num_values):
    result = runner.invoke(cli, ["clean", "rm_missing_values"] + num_values)
    
    assert result.exit_code == 0
    assert str(num_values) in result.output

@pytest.mark.parametrize("values, expected", [
    (["1", "", "", "4"], ["1", "4"]),
    (["", "2", "3", ""], ["2", "3"]),
    (["1", "2", "", ""], ["1", "2"]),
])
def test_rm_missing_values_with_some_missing(runner, values, expected):
    result = runner.invoke(cli, ["clean", "rm_missing_values"] + values)
    
    assert result.exit_code == 0
    assert str(expected) in result.output
    
def test_fill_missing_values_with_none_missing(runner, num_values):
    result = runner.invoke(cli, ["clean", "fill_missing_values"] + num_values)
    
    assert result.exit_code == 0
    assert str(num_values) in result.output

def test_fill_missing_values_with_some_missing(runner, num_values):
    values = [str(v) for v in num_values]
    values.append("")
    expected = values[:-1].copy()
    expected.append("0")
    result = runner.invoke(cli, ["clean", "fill_missing_values"] + values)
    assert result.exit_code == 0
    assert str(expected) in result.output
    
def test_rm_duplicated_values_with_none_duplicated(runner, num_values):
    result = runner.invoke(cli, ["struct", "unique_values"] + num_values)
    
    assert result.exit_code == 0
    assert str(num_values) in result.output

def test_rm_duplicated_values_with_some_duplicated(runner, num_values):
    values = num_values.copy()
    values.append(num_values[2])
    result = runner.invoke(cli, ["struct", "unique_values"] + values)
    
    assert result.exit_code == 0
    assert str(num_values) in result.output

def test_min_max_normalization(runner, num_values):
    result = runner.invoke(cli, ["numeric", "normalization"] + num_values + ["--min", "0", "--max", "6"])
    print(["numeric", "normalization"] + num_values + ["--min", "0", "--max", "6"])
    
    assert result.exit_code == 0
    assert str([0.0, 2.0, 4.0, 6.0]) in result.output

def test_standarization(runner, num_values):
    result = runner.invoke(cli, ["numeric", "standarization"] + num_values)
    
    assert result.exit_code == 0
    ret_values = eval(result.output)
    assert ret_values == pytest.approx([-1.34164079, -0.4472136, 0.4472136, 1.34164079])

@pytest.mark.parametrize("min_v, max_v, expected", [
    (2, 3, [2.0, 2.0, 3.0, 3.0]),
    (5, 10, [5.0]*4),
    (1, 5, [1.0, 2.0, 3.0, 4.0]),
])
def test_clip_values(runner, num_values, min_v, max_v, expected):
    result = runner.invoke(cli, ["numeric", "clip"] + num_values + ["--min", f"{min_v}", "--max", f"{max_v}"])
    
    assert result.exit_code == 0
    assert str(expected) in result.output


def test_values_to_ints(runner, num_values):
    result = runner.invoke(cli, ["numeric", "values_to_ints"] + num_values)
    
    assert result.exit_code == 0
    assert str([int(v) for v in num_values]) in result.output


def test_log_scale(runner, num_values):
    values = [int(v) for v in num_values]
    print((["numeric", "log_scale"] + num_values))
    result = runner.invoke(cli, ["numeric", "log_scale"] + num_values)
    print(result.output)
    
    assert result.exit_code == 0
    assert str([float(np.log(v)) for v in values]) in result.output


def test_tokenize(runner):
    result = runner.invoke(cli, ["text", "tokenize", "Str That Needs To Be Tokenized!"])
    
    assert result.exit_code == 0
    assert "strthatneedstobetokenized" in result.output


def test_select_alphanumerical(runner):
    result = runner.invoke(cli, ["text", "select_alphanumerical", "Str with **many!** symbols & characters?"])
    
    assert result.exit_code == 0
    assert "Str with many symbols  characters" in result.output


def test_rm_stopwords(runner):
    result = runner.invoke(cli, ["text", "rm_stopwords", "string of the stopwords to remove"] + ["of","a","the","to"])
    
    assert result.exit_code == 0
    assert "string stopwords remove" in result.output


def test_flatten(runner):
    result = runner.invoke(cli, ["struct", "flatten"] + ["a", "b,c", "d"])
    
    assert result.exit_code == 0
    assert str(["a", "b", "c", "d"]) in result.output


def test_shuffle(runner, num_values):
    result = runner.invoke(cli, ["struct", "shuffle"] + num_values + ["--seed", "42"])
    
    assert result.exit_code == 0
    assert str(num_values[::-1]) in result.output

