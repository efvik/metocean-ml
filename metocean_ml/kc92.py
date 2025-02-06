"""This module contains functions to use the composite growth curves of Kahma & Calkoen (1992)

All transformations to spatial domain (wavelength etc.) usees DEEP water dispersion relation

Reference: Kahma, K. K., and C. J. Calkoen, 1992: Reconciling Discrepancies in the Observed Growth of
Wind-generated Waves. J. Phys. Oceanogr., 22, 1389-1405,
https://doi.org/10.1175/1520-0485(1992)022<1389:RDITOG>2.0.CO;2."""

import numpy as np


G = 9.81  # m/s^2


def variance(x: float, u: float) -> float:
    """Calculates variance (m^2) from wind speed (m/s) and fetch (m) given Kahma & Calkoen (1992) composite curves

    Reference: Kahma, K. K., and C. J. Calkoen, 1992: Reconciling Discrepancies in the Observed Growth of
    Wind-generated Waves. J. Phys. Oceanogr., 22, 1389-1405,
    https://doi.org/10.1175/1520-0485(1992)022<1389:RDITOG>2.0.CO;2.
    """

    xhat = G * x / u / u  # Dimensonless fetch

    # page 1404
    ehat = (5.2 * 10**-7) * xhat**0.9  # dimensonless energy

    return ehat / G / G * u**4


def hs(x: float, u: float) -> float:
    """Calculates significant wave height (m) from wind speed (m/s) and fetch (m) given Kahma & Calkoen (1992) composite curves

    Reference: Kahma, K. K., and C. J. Calkoen, 1992: Reconciling Discrepancies in the Observed Growth of
    Wind-generated Waves. J. Phys. Oceanogr., 22, 1389-1405,
    https://doi.org/10.1175/1520-0485(1992)022<1389:RDITOG>2.0.CO;2.
    """
    return 4 * variance(x=x, u=u) ** 0.5


def wp(x: float, u: float) -> float:
    """Calculates angular pead frequency (rad/s) from wind speed (m/s) and fetch (m) given Kahma & Calkoen (1992) composite curves

    Reference: Kahma, K. K., and C. J. Calkoen, 1992: Reconciling Discrepancies in the Observed Growth of
    Wind-generated Waves. J. Phys. Oceanogr., 22, 1389-1405,
    https://doi.org/10.1175/1520-0485(1992)022<1389:RDITOG>2.0.CO;2.
    """
    xhat = G * x / u / u  # Dimensonless fetch

    # page 1404
    omegahat = 13.7 * xhat ** (-0.27)  # dimensonless peak frequency

    return omegahat * G / u


def fp(x: float, u: float) -> float:
    """Calculates peak frequency (Hz) from wind speed (m/s) and fetch (m) given Kahma & Calkoen (1992) composite curves

    Reference: Kahma, K. K., and C. J. Calkoen, 1992: Reconciling Discrepancies in the Observed Growth of
    Wind-generated Waves. J. Phys. Oceanogr., 22, 1389-1405,
    https://doi.org/10.1175/1520-0485(1992)022<1389:RDITOG>2.0.CO;2.
    """

    return wp(x=x, u=u) / 2 / np.pi


def tp(x: float, u: float) -> float:
    """Calculates peak period (s) from wind speed (m/s) and fetch (m) given Kahma & Calkoen (1992) composite curves

    Reference: Kahma, K. K., and C. J. Calkoen, 1992: Reconciling Discrepancies in the Observed Growth of
    Wind-generated Waves. J. Phys. Oceanogr., 22, 1389-1405,
    https://doi.org/10.1175/1520-0485(1992)022<1389:RDITOG>2.0.CO;2.
    """

    return 1 / fp(x=x, u=u)


def lp(x: float, u: float) -> float:
    """Calculates peak wavelenght (m) from wind speed (m/s) and fetch (m) given Kahma & Calkoen (1992) composite curves

    Uses deep water linear dispersion

    Reference: Kahma, K. K., and C. J. Calkoen, 1992: Reconciling Discrepancies in the Observed Growth of
    Wind-generated Waves. J. Phys. Oceanogr., 22, 1389-1405,
    https://doi.org/10.1175/1520-0485(1992)022<1389:RDITOG>2.0.CO;2.
    """

    return G / np.pi / 2 * tp(x=x, u=u) ** 2


def kp(x: float, u: float) -> float:
    """Calculates peak wavenumber (rad/m) from wind speed (m/s) and fetch (m) given Kahma & Calkoen (1992) composite curves

    Uses deep water linear dispersion

    Reference: Kahma, K. K., and C. J. Calkoen, 1992: Reconciling Discrepancies in the Observed Growth of
    Wind-generated Waves. J. Phys. Oceanogr., 22, 1389-1405,
    https://doi.org/10.1175/1520-0485(1992)022<1389:RDITOG>2.0.CO;2.
    """

    return 2 * np.pi / lp(x=x, u=u)
