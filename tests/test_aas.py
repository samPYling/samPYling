from pytest import mark, raises

from sampyling.aas import AAS


@mark.parametrize(
    "pop_size, conf, margin_error, expected",
    [
        (100, 0.95, 0.02, 97),
        (1000, 0.95, 0.02, 706),
        (1000, 0.99, 0.02, 807),
        (1000, 0.9, 0.02, 628),
        (1000, 0.95, 0.03, 517),
        (1000, 0.95, 0.05, 278),
    ],
)
def test_calculate_sample_size_without_pop_var(
    pop_size, conf, margin_error, expected
):
    aas = AAS(pop_size, conf, margin_error)

    result = aas.calculate_sample_size()

    assert result == expected


@mark.parametrize(
    "pop_size, conf, margin_error, pop_var, expected",
    [
        (100, 0.95, 0.02, 0.3, 97),
        (1000, 0.90, 0.02, 0.4, 729),
        (1000, 0.99, 0.02, 0.5, 893),
    ],
)
def test_calculate_sample_size_with_pop_var(
    pop_size, conf, margin_error, pop_var, expected
):
    aas = AAS(pop_size, conf, margin_error, pop_var)

    result = aas.calculate_sample_size()

    assert result == expected


@mark.parametrize(
    "pop_size, conf, margin_error, p, expected",
    [
        (100, 0.95, 0.02, 0.5, 0.25),
        (1000, 0.90, 0.02, 0.9, 0.09),
        (1000, 0.99, 0.02, 0.3, 0.21),
    ],
)
def test_calculate_population_variance(
    pop_size, conf, margin_error, p, expected
):
    aas = AAS(pop_size, conf, margin_error)

    result = aas.calculate_population_variance(p)

    assert round(result, 2) == expected


def test_confidence_level_greater_than_1_must_return_error():
    error_message = "`conf` precisa estar entre 0 e 1."

    with raises(ValueError) as error:
        AAS(100, 2.0, 0.02)

    assert error.value.args[0] == error_message


def test_confidence_level_smaller_than_0_must_return_error():
    error_message = "`conf` precisa estar entre 0 e 1."

    with raises(ValueError) as error:
        AAS(100, -0.95, 0.02)

    assert error.value.args[0] == error_message


def test_confidence_level_not_float_must_return_error():
    error_message = "`conf` precisa ser um float: <class 'list'>"

    with raises(ValueError) as error:
        AAS(100, [0.95, 0.9], 0.02)  # pyright: ignore

    assert error.value.args[0] == error_message
