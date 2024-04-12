from pytest import mark

from sampyling.constants import SamplingType
from sampyling.design import PopulationDesign, SampleDesign
from sampyling.size import SampleSizeCalculator


@mark.parametrize(
    "design, expected",
    [
        (
            SampleDesign(
                population_design=PopulationDesign(pop_size=100),
                sample_type=SamplingType.srs,
                conf_level=0.95,
                margin_error=0.02,
            ),
            97,
        ),
        (
            SampleDesign(
                population_design=PopulationDesign(pop_size=1000),
                sample_type=SamplingType.srs,
                conf_level=0.95,
                margin_error=0.02,
            ),
            706,
        ),
        (
            SampleDesign(
                population_design=PopulationDesign(pop_size=1000),
                sample_type=SamplingType.srs,
                conf_level=0.99,
                margin_error=0.02,
            ),
            807,
        ),
        (
            SampleDesign(
                population_design=PopulationDesign(pop_size=1000),
                sample_type=SamplingType.srs,
                conf_level=0.95,
                margin_error=0.03,
            ),
            517,
        ),
    ],
)
def test_calculate_sample_size_without_pop_var(design, expected):
    aas = SampleSizeCalculator(design)

    result = aas.calculate()

    assert result.sample_size == expected


@mark.parametrize(
    "design, expected",
    [
        (
            SampleDesign(
                population_design=PopulationDesign(pop_size=100, pop_var=0.3),
                sample_type=SamplingType.srs,
                conf_level=0.95,
                margin_error=0.02,
            ),
            97,
        ),
        (
            SampleDesign(
                population_design=PopulationDesign(pop_size=1000, pop_var=0.4),
                sample_type=SamplingType.srs,
                conf_level=0.90,
                margin_error=0.02,
            ),
            729,
        ),
        (
            SampleDesign(
                population_design=PopulationDesign(pop_size=1000, pop_var=0.5),
                sample_type=SamplingType.srs,
                conf_level=0.99,
                margin_error=0.02,
            ),
            893,
        ),
    ],
)
def test_calculate_sample_size_with_pop_var(design, expected):
    aas = SampleSizeCalculator(design)

    result = aas.calculate()

    assert result.sample_size == expected
