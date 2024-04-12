# TODO: Construir docstrings
from typing import Callable

import numpy as np

from sampyling.design import SampleDesign, SamplingType


class SampleSizeCalculator:
    """Calcula o tamanho da amostra a partir de um design amostral.

    Esta classe permite calcular o tamanho adequado da amostra com base no design de amostragem fornecido.

    Parameters
    ----------
    sample_design: SampleDesign
        O design da amostra utilizado para calcular o tamanho da amostra.

    Attributes
    ----------
    sample_design: SampleDesign
        Obtém o design da amostra com o tamanho de amostra calculado, realizando o cálculo se ainda não tiver sido feito.

    Examples
    --------
    """

    # TODO: Docstring com os exemplos

    _sample_size_calculators: dict[SamplingType, Callable[[], int]]

    def __init__(self, sample_design: SampleDesign) -> None:
        self._sample_size_calculators = {
            SamplingType.srs: self._calculate_srs_size
        }

        self._sample_design = sample_design

    def _calculate_srs_size(self) -> int:
        """Calcula o tamanho de amostra para uma amostragem aleatória simples.

         O cálculo do tamanho da amostra para uma AAS segundo Lohr é:

         n = n_0 / (1 + (n_0 / N)),

         Onde:

         - n = tamanho da amostra
         - n_0 = o tamanho de amostra inicial calculado como:
             n_0 = (Z_alpha/2² * S²)/e²,
             - Z_alpha/2 = Z Score da Normal padrão para o nível de confiança
             (1 - alpha)
             - e = margem de erro desejada
         - N = tamanho da população alvo

        References
         ----------
         - Sampling Design and Analysis 2nd. Ed., Sharon L. Lohr - 2010
        """

        n_0 = (
            self._sample_design.z_score**2
            * self._sample_design.population_design.pop_var
        ) / self._sample_design.margin_error**2

        sample_size = n_0 / (
            1 + (n_0 / self._sample_design.population_design.pop_size)
        )

        return int(np.ceil(sample_size))

    def calculate(self) -> SampleDesign:
        """Calcula e atribui o tamanho da amostra ao design de amostra fornecido.

        Examples
        --------

        Returns
        -------
        SampleDesign:
            O design da amostra com o tamanho de amostra calculado.
        """

        # TODO: Construir docstring com exemplos
        size = self._sample_size_calculators[self._sample_design.sample_type]()

        self._sample_design.sample_size = size

        return self._sample_design

    @property
    def sample_design(self) -> SampleDesign:
        """Obtém o design da amostra com o tamanho de amostra calculado.

        Se o tamanho da amostra ainda não foi calculado, este método
        calculará automaticamente antes de retornar o design da amostra.
        """
        if not self._sample_design.sample_size:
            self.calculate()

        return self._sample_design
