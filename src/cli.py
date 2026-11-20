"""
Command line module
"""

import click
from . import preprocessing


@click.group()
def cli():
    """
    Cli group
    """


@cli.group()
def clean():
    """
    Clean group
    """


@clean.command("rm_missing_values")
@click.argument("values", nargs=-1, required=True, help="List with values.")
def rm_missing_values(values):
    """
    Removes missing values from a list

    >> rm_missing_values a b
    """
    values_preprocessed = preprocessing.rm_missing_values(values)
    print(values_preprocessed)


@clean.command("fill_missing_values")
@click.argument("values", nargs=-1, required=True, help="List with values.")
def fill_missing_values(values):
    """
    Fills missing values from a list

    >> fill_missing_values a b
    """
    values_preprocessed = preprocessing.fill_missing_values(values)
    print(values_preprocessed)


@cli.group()
def numeric():
    """
    Numeric group
    """


@numeric.command("normalization")
@click.argument("values", nargs=-1, required=True, help="List with values.")
@click.option(
    "--min",
    "min_value",
    default=0,
    type=click.FLOAT,
    required=False,
    help="Min, default = 0.",
)
@click.option(
    "--max",
    "max_value",
    default=1,
    type=click.FLOAT,
    required=False,
    help="Max, default = 1.",
)
def normalization(values, min_value, max_value):
    """
    Normalizes the values from a list

    >> normalization 0 1 2 --min_value 0 --max_value 1
    """
    values = [float(v) for v in values]
    values_preprocessed = preprocessing.min_max_normalization(
        values, min_value, max_value
    )
    print(values_preprocessed)


@numeric.command("standarization")
@click.argument("values", nargs=-1, required=True, help="List with values.")
def standarization(values):
    """
    Standarizes the values from a list

    >> standarization 0 1 2
    """
    values = [float(v) for v in values]
    values_preprocessed = preprocessing.standarization(values)
    print(values_preprocessed)


@numeric.command("clip")
@click.argument("values", nargs=-1, required=True, help="List with values.")
@click.option(
    "--min",
    "min_value",
    default=0,
    type=click.FLOAT,
    required=False,
    help="Min, default = 0.",
)
@click.option(
    "--max",
    "max_value",
    default=1,
    type=click.FLOAT,
    required=False,
    help="Max, default = 1.",
)
def clip(values, min_value, max_value):
    """
    Clips the values from a list

    >> clip 0 1 2 --min_value 0 --max_value 1
    """
    values_preprocessed = preprocessing.clip_values(
        [float(v) for v in values], min_value, max_value
    )
    print(values_preprocessed)


@numeric.command("values_to_ints")
@click.argument("values", nargs=-1, required=True, help="List with values.")
def values_to_ints(values):
    """
    Changes the values from a list to ints

    >> values_to_ints 0 1 2
    """
    values_preprocessed = preprocessing.values_to_ints(values)
    print(values_preprocessed)


@numeric.command("log_scale")
@click.argument("values", nargs=-1, required=True, help="List with values.")
def log_scale(values):
    """
    Changes the scale of the values from a list to logarithmic

    >> log_scale 0 1 2
    """
    values = [float(v) for v in values]
    values_preprocessed = preprocessing.log_scale(values)
    print(values_preprocessed)


@cli.group()
def text():
    """
    Text group
    """


@text.command("tokenize")
@click.argument("in_text", required=True, help="Text.")
def tokenize(in_text):
    """
    Tokenizes a text

    >> tokenize "Hello wold"
    """
    text_preprocessed = preprocessing.tokenize(in_text)
    print(text_preprocessed)


@text.command("select_alphanumerical")
@click.argument("in_text", required=True, help="Text.")
def select_alphanumerical(in_text):
    """
    Deletes non-alphanumerical characters of a text

    >> select_alphanumerical "Hello wold"
    """
    text_preprocessed = preprocessing.select_alphanumerical(in_text)
    print(text_preprocessed)


@text.command("rm_stopwords")
@click.argument("in_text", required=True, help="Text.")
@click.argument(
    "stopwords",
    nargs=-1,
    required=True,
    help="List of stopwords to remove from the text.",
)
def rm_stopwords(in_text, stopwords):
    """
    Removes the given stopwords from a text

    >> rm_stopwords "Hello wold" "word1" "word2"
    """
    text_preprocessed = preprocessing.rm_stopwords(in_text, stopwords)
    print(text_preprocessed)


@cli.group()
def struct():
    """
    Struct group
    """


@struct.command("flatten")
@click.argument(
    "values",
    nargs=-1,
    required=True,
    help="List with values (comma-separated for sublists).",
)
def flatten(values):
    """
    Flattens a list of lists

    >> flatten "a,b" "c,d"
    """
    values_preprocessed = preprocessing.flatten([v.split(",") for v in values])
    print(values_preprocessed)


@struct.command("shuffle")
@click.argument("values", nargs=-1, required=True, help="List with values.")
@click.option(
    "--seed", type=click.INT, default=None, required=False, help="Seed for rng."
)
def shuffle(values, seed):
    """
    Shuffles the values from a list

    >> shuffle a b
    """
    values_preprocessed = preprocessing.shuffle(values, seed)
    print(values_preprocessed)


@struct.command("unique_values")
@click.argument("values", nargs=-1, required=True, help="List with values.")
def unique_values(values):
    """
    Removes duplicated values from a list

    >> unique_values a b
    """
    values_preprocessed = preprocessing.rm_duplicated_values(values)
    print(values_preprocessed)


if __name__ == "__main__":
    cli()
