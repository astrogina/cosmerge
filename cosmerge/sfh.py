"""Contains cosmic star formation information"""

import numpy as np
import astropy.units as u
from scipy.stats import norm as NormDist


def md_14(z):
    """The Madau & Dickinson (2014) star formation rate
    per comoving volume as a function of redshift

    Parameters
    ----------
    z : float or numpy.array
        redshift

    Returns
    -------
    sfr : float or numpy.array
        star formation rate per comoving volume at redshift z
        with astropy units
    """
    sfr = (0.015 * (1 + z)**2.7 / (1 + ((1 + z) / (1 + 1.9))**5.6) * u.Msun * u.Mpc**(-3) * u.yr**(-1))

    return sfr


def mf_17(z):
    """The Madau & Fragos (2017) star formation rate
    per comoving volume as a function of redshift

    Parameters
    ----------
    z : float or numpy.array
        redshift

    Returns
    -------
    sfr : float or numpy.array
        star formation rate per comoving volume at redshift z
        with astropy units
    """

    sfr = 0.01 * (1 + z)**2.6 / (1 + ((1 + z) / 3.2)**6.2) * u.Msun/(u.Mpc**3 * u.yr)

    return sfr

def mean_metal_log_z(z, Zsun=0.017):
    """
    Mass-weighted average log(metallicity) as a function of redshift
    From Madau & Fragos (2017)

    Parameters
    ----------
    z : float or numpy.array
        redshift

    Zsun : float or numpy.array
        metallicity of the sun
        NOTE: Madau & Fragos assume Zsun = 0.017

    Returns
    -------
    log_Z : float or numpy.array
        log(mean metallicity)
    """

    log_Z_Zsun = 0.153 - 0.074 * z ** 1.34
    log_Z = np.log(10 ** log_Z_Zsun * Zsun)

    return log_Z


def log_p_Z_z(Z, z, sigma_log10Z):
    """Computes the metallicity and redshift log probability
    distribution function assuming a log normal metallicity
    distribution with sigma at each redshift

    Parameters
    ----------
    Z : numpy.array
        metallicities

    z : numpy.array
        redshifts

    sigma_log10Z : numpy.array
        standard deviation of metallicity in dex (convert to log)

    Returns
    -------
    log_pi : numpy.array
        log probability of the metallicity/redshift distribution at Z,z
    """
    mu = mean_metal_log_z(z)
    sigma = np.ones_like(z) * sigma_log10Z * np.log(10)

    return -np.log(Z) - np.log(sigma) - 0.5 * np.square((np.log(Z) - mu) / sigma)


# Below are all taken from van Son 2022 (The Redshift Evolution...)
# Trying my best to make them flexible but I'm also prioritizing
# getting my project finished :D 
def van_son_22(z, a=-0.02, b=1.48, c=4.45, d=5.9):
    """Generalized version of the star formation rate. Default values
    correspond to the star formation rate given in van Son (2022), which
    was based on the TNG simulation. In units of mass per comoving volume 
    as a function of redshift.
    
    Parameters
    ----------
    z : float or numpy.array
        redshift
    a : float
    b : float
    c : float
    d : float
    
    Returns
    -------
    sfr : float or numpy.array
        star formation rate per comoving volume at redshift z
        with astropy units"""
    
    sfr = a * (1 + z)**b / (1 + ((1 + z) / c)**d) * u.Msun/(u.Mpc**3 * u.yr)
    
    return sfr


# TODO: check what the assumption is for Zsun here
def mu_z(z, mu0, muz):
    """Redshift dependence of mean metallicity assuming 0 skew.
    
    Parameters
    ----------
    z: float or numpy.array
        redshift
    mu0 : float
    muz : float
    
    Returns
    -------
    mu : float or numpy.array
        mean metallicity at specified redshift for 0 skew distribution"""
    
    return mu0 * 10**(muz * z)


def omega_z(z, omega0, omegaz)
    """Redshift dependence of scale of metallicity distribution.
    
    Parameters
    ----------
    z : float or numpy.array
        redshift
    omega0 : float
    omegaz : float
    
    Returns
    -------
    omega : float or numpy.array
        scale of metallicity distribution at specified metallicity"""
    return omega0 * 10**(omegaz * z)


def mean_Z_z(z, mu0, muz, alpha):
    """Redshift dependence of mean metallicity for skewed distribution.
    
    Parameters
    ----------
    z : float or numpy.array
        redshift
    mu0 : float
    muz : float
    
    Returns
    -------
    xi : float or numpy.array
        updated mean of metallicity distribution assuming a skewed
        log-normal distribution
    """
    omega = omega_z(z, omega0, omegaz)
    mu = mu_z(z, mu0, muz)
    beta = alpha/(np.sqrt(1 + alpha**2))
    return -omega**2 / 2 * np.log(mu / (2 * NormDist.cdf(beta * omega)))


def log_p_Z_z_skewed(Z, z, mu0=1.125, muz=-0.048, omega0=1.125, omegaz=0.048, alpha=-1.77):
    """The metallicity and redshift log probability distribution function. 
    Default values of constants correspond to the star formation rate given 
    in van Son (2022), which was based on the TNG simulation. 
    
    Parameters
    ----------
    Z : float or numpy.array
        metallicity
    z : float or numpy.array
        redshift
    mu0 : float
    muz : float
    omega0 : float
    omegaz : float
    alpha : float
        skew parameter
    
    Returns
    -------
    log_pi : numpy.array
        log probability (ln(dP/dlnZ)) of metallicity/redshift distribution at
        specified metallicity and redshift values"""
    
    omega = omega_z(z, omega0, omegaz)
    xi = mean_Z_z(z, mu0, muz, alpha)
    dPdlnZ = 2 / omega * NormDist.pdf((np.log(Z) - xi) / omega) * NormDist.cdf(alpha * (np.log(Z) - xi) / omega)
    
    return np.log(dPdlnZ)