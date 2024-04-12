"""Este módulo define os designs de amostra e população para serem utilizados
em outros métodos de estimação.

Os designs reúnem informações necessárias para que possam ser utilizadas
técnicas estatísticas de estimação.

Nesse módulo você encontrará:

- PopulationDesign: Design da sua população alvo.
- SampleDesign: Design utilizado para a sua amostragem.


Exemplos:

>>> PopulationDesign(pop_size=1000)
PopulationDesign(pop_size=1000, pop_var=0.25)
>>> PopulationDesign(pop_size=500, pop_var=0.3)
PopulationDesign(pop_size=500, pop_var=0.3)
>>> SampleDesign(
...    population_design=PopulationDesign(pop_size=100),
...    sample_type="srs",
...    conf_level=0.95,
...    margin_error=0.02
... )
SampleDesign(population_design=PopulationDesign(pop_size=100, pop_var=0.25), sample_type=<SamplingType.srs: 'srs'>, conf_level=0.95, margin_error=0.02, z_score=1.96, alpha=0.05, sample_size=None)
>>> SampleDesign(
...    population_design=PopulationDesign(pop_size=100),
...    sample_type="srs",
...    conf_level=0.95,
...    margin_error=0.02,
...    sample_size=100,
... )
SampleDesign(population_design=PopulationDesign(pop_size=100, pop_var=0.25), sample_type=<SamplingType.srs: 'srs'>, conf_level=0.95, margin_error=0.02, z_score=1.96, alpha=0.05, sample_size=100)

"""

from typing import Annotated, Optional, Union

import numpy as np
from pydantic import Field, PositiveFloat
from pydantic.dataclasses import dataclass
from scipy.stats import norm

from sampyling.constants import SamplingType


@dataclass
class PopulationDesign:
    """Design da população alvo do estudo.

    Parameters
    ----------
    pop_size: int
        Tamanho da população.
    pop_var: PositiveFloat, padrão 0.25
        Variância populacional. Caso não seja especificado, será utilizado a
        variância máxima.

    Attributes
    ----------
    pop_size: int
        Tamanho da população.
    pop_var: PositiveFloat, padrão 0.25
        Variância populacional. Caso não seja especificado, será utilizado a
        variância máxima.

    Examples
    --------
    >>> PopulationDesign(pop_size=100)
    PopulationDesign(pop_size=100, pop_var=0.25)
    >>> PopulationDesign(pop_size=500, pop_var=0.3)
    PopulationDesign(pop_size=500, pop_var=0.3)
    """

    pop_size: int
    pop_var: PositiveFloat = 0.25


@dataclass
class SampleDesign:
    """Representa um design de amostra para amostragem estatística.

    Parameters
    ----------
    population_design : PopulationDesign
        Os detalhes do design da população da qual está sendo feita a amostragem.
    sample_type : SamplingType
        O tipo de técnica de amostragem utilizada.
    conf_level : Annotated[float, Field(ge=0, le=1)]
        O nível de confiança para o design da amostra, um valor float entre 0 e 1.
    margin_error : Union[int, float]
        A margem de erro permitida no design da amostra.

    Attributes
    ---------
    population_design : PopulationDesign
        Os detalhes do design da população da qual está sendo feita a amostragem.
    sample_type : SamplingType
        O tipo de técnica de amostragem utilizada.
    conf_level : Annotated[float, Field(ge=0, le=1)]
        O nível de confiança para o design da amostra, um valor float entre 0 e 1.
    margin_error : Union[int, float]
        A margem de erro permitida no design da amostra.

    z_score : float
        O valor Z-score derivado do nível de confiança (inicializado automaticamente).
    alpha : float
        O nível de significância derivado do nível de confiança (inicializado automaticamente).

    Notes
    -----
    O Z-score é arredondado para duas casas decimais.

    Examples
    --------
    >>> population = PopulationDesign(pop_size=100)
    >>> sample_design = SampleDesign(population, "srs", conf_level=0.95, margin_error=0.05)
    >>> sample_design.z_score
    1.96
    >>> sample_design.alpha
    0.05
    """

    population_design: PopulationDesign
    sample_type: SamplingType
    conf_level: Annotated[float, Field(ge=0, le=1)]
    margin_error: Union[int, float]
    z_score: float = Field(init=False)
    alpha: float = Field(init=False)
    sample_size: Optional[int] = None

    def __get_z_score(self) -> float:
        """Busca o valor Z de uma Normal Padrão a partir do nível de confiança."""

        probability = 1 - (self.alpha / 2)
        z_score = np.round(float(norm.ppf(probability)), 2)

        return z_score

    def __post_init__(self):
        self.alpha = np.round(1 - self.conf_level, 2)
        self.z_score = self.__get_z_score()
