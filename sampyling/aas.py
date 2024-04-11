# TODO: Construir docstring para o módulo
import typing as T

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import norm


class SampleMetadata(BaseModel):
    """Metadados da amostra."""

    conf: T.Annotated[float, Field(ge=0, le=1)]
    margin_error: T.Union[int, float]
    z_score: float
    pop_var: float
    pop_size: int
    sample_size: int


class AAS:
    """Classe responsável por gerenciar os métodos de amostra aleatória
    simples.

    Com essa classe é possível:

    - Estimar o tamanho de amostra considerando amostragem aleatória simples

    Parameters
    ----------
    pop_size: int
        Número total de elementos na população alvo.
    conf: Annotated[float, Field(ge=0, le=1)]
        Nível de confiança desejado para as estimativas, deve estar entre 0 e 1.
    margin_error: int or float
        Margem de erro desejada para as estimativas.
    pop_var: float, optional
        Variância populacional. Caso não seja especificada, será considerado a
        variância máxima utilizando p = 1/2 = 0.5.

    Attributes
    ----------
    pop_size: int
        Número total de elementos na população alvo.
    pop_var: float
        Variância populacional.
    conf: float
        Nível de confiança desejado para as estimativas.
    margin_error: int or float
        Margem de erro desejada para as estimativas.
    z_score: float
        Z Score da Normal Padrão.


    Examples
    --------
    >>> aas = AAS(pop_size=100, conf=0.95, margin_error=0.02)
    >>> aas.pop_size
    100
    >>> aas.conf
    0.95
    >>> aas.sample_size
    97
    >>> aas.calculate_sample_size()
    97
    >>> aas.pop_var
    0.25
    >>> aas.calculate_population_variance(p=0.5)
    0.25
    >>> aas.z_score
    1.96
    >>> aas.sample_metadata
    SampleMetadata(conf=0.95, margin_error=0.02, z_score=1.96, pop_var=0.25, pop_size=100, sample_size=97)
    """

    _alpha: T.Annotated[float, Field(ge=0, le=1)]
    _z_score: float
    _sample_metadata: SampleMetadata
    _sample_size: int

    def __init__(
        self,
        pop_size: int,
        conf: T.Annotated[float, Field(ge=0, le=1)],
        margin_error: T.Union[int, float],
        pop_var: T.Optional[float] = None,
    ) -> None:
        self.pop_size = pop_size
        self._pop_var = pop_var
        self.margin_error = margin_error

        if isinstance(conf, float):
            if 0 <= conf <= 1:
                self.conf = conf
            else:
                raise ValueError("`conf` precisa estar entre 0 e 1.")
        else:
            raise ValueError(f"`conf` precisa ser um float: {type(conf)}")

        self._alpha = 1 - self.conf

    def calculate_population_variance(
        self, p: T.Annotated[float, Field(ge=0, le=1)] = 0.5
    ) -> float:
        """Calcula a variância populacional segundo uma distribuição Bernoulli.

        Caso a probabilidade de sucesso não seja especificada, será utilizado a
        abordagem conservadora calculando a variância máxima dada por:

        S² = p(1 - p), p = 1/2 = 0.5

        Parameters
        ----------
        p: float, default 0.5
            Probabilidade de sucesso para o cálculo da variância.
        """

        self._pop_var = p * (1 - p)

        return self._pop_var

    @property
    def pop_var(self) -> float:
        """Variância populacional."""

        if self._pop_var is None:
            self.calculate_population_variance()

        # TODO: Resolver erro de tipos do pyright -> Expression of type "float | None" cannot be assigned to return type "float"
        return self._pop_var

    def get_z_score(self) -> float:
        """Busca o valor Z de uma Normal Padrão a partir do nível de confiança.

        O Z Score é arredondado para duas casas decimais.
        """

        probability = 1 - (self._alpha / 2)
        self._z_score = np.round(float(norm.ppf(probability)), 2)

        return self._z_score

    @property
    def z_score(self) -> float:
        """Z Score da Normal Padrão."""

        if not hasattr(self, "_z_score"):
            self.get_z_score()

        return self._z_score

    def calculate_sample_size(self) -> int:
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

        Examples
        --------
        >>> aas = AAS(100, 0.95, 0.02)
        >>> aas.calculate_sample_size()
        97
        >>> aas.sample_size
        97

        References
        ----------
        - Sampling Design and Analysis 2nd. Ed., Sharon L. Lohr - 2010
        """

        n_0 = (self.z_score**2 * self.pop_var) / self.margin_error**2

        n = n_0 / (1 + (n_0 / self.pop_size))

        self._sample_size = int(np.ceil(n))

        return self._sample_size

    @property
    def sample_size(self) -> int:
        """Tamanho da amostra."""

        if not hasattr(self, "_sample_size"):
            self.calculate_sample_size()

        return self._sample_size

    @property
    def sample_metadata(self) -> SampleMetadata:
        """Metadados da amostra."""

        if not hasattr(self, "_sample_metadata"):
            self._sample_metadata = SampleMetadata(
                conf=self.conf,
                margin_error=self.margin_error,
                z_score=self.z_score,
                pop_var=self.pop_var,
                pop_size=self.pop_size,
                sample_size=self.sample_size,
            )

        return self._sample_metadata
