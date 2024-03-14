import numpy as np
import scipy.optimize as opt
import astropy.constants as const

def search_store(file: np.lib.npyio.NpzFile, string: str) -> np.ndarray:
    """
    Search an npz file for all files containing a certain string and return an array of them.
    
    Parameters
    ----------
    file : numpy .NpzFile
        The npz file to be searched.
    string : str
        The string being searched for in the file names.

    Returns
    -------
    arr : numpy array
        The files containing the string.
    """
    lst = []
    for i in file:
        if string in i:
            lst.append(file[i])
    arr = np.asarray([lst])
    return arr[0]

def chi2(p, x, y, func) -> float:
    """
    Produce a chi squared distribution as a merit function for the purposes of fitting.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the function func.
    x : array_like
        The x data to be fit.
    y : array_like
        The y data to be fit.
    func : callable
        The function to be fit.

    Returns
    -------
    float
        The chi squared value for this fit.
    """
    return np.sum((y - func(p, x))**2)

def gauss(p, x) -> np.ndarray:
    """
    Produce a Gaussian function.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the Gaussian function.
    x : array_like
        The x range over which the Gaussian function is to be produced.

    Returns
    -------
    numpy.ndarray
        The Gaussian function.
    """
    return p[0] * np.exp(-(x - p[1])**2 / (2 * p[2]**2)) + p[3]

def sin(p, x, period=1.2410229) -> np.ndarray:
    """
    Produce a sin function for a given period.
    
    Keyword arguments:
    p : array_like
        Coefficients of the sin function.
    x : array_like
        The x range over which the sin function is to be produced.
    period : float, default 1.2410229

    Returns:
    numpy.ndarray
        The sin function.
    """
    # The default period is the orbital of the planet in this assignment
    return p[0] * np.sin(2 * np.pi/period * x + p[1]) + p[2]


def residuals(p, func, x, y, s=1) -> np.ndarray:
    """
    Find the residuals from fitting data to a function.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the function func.
    x : array_like
        The x data to be fit.
    y : array_like
        The y data to be fit.
    func : callable
        The function to be fit.
    s : float, default 1
        The standard deviation.

    Returns
    -------
    numpy.ndarray
        The residuals of the fit function.
    """
    return (y - func(p, x)) / s

def fitting(p, x, y, func, s=1):
    """
    Fit data to a function.
    
    Parameters
    ----------
    p : array_like
        Initial guess at the values of the coefficients to be fit
    x : array_like
        The x data to be fit.
    y : array_like
        The y data to be fit.
    func : callable
        The function to fit the data to.
    s : float, default 1
        The standard deviation.

    Returns
    -------
    p_fit : numpy.ndarray
        The array of the coefficients after fitting the function to the data.
    chi : float
        The chi squared value of this fit.
    unc_fit : numpy.ndarray
        The array of the uncertainties in the values of p_fit.
    """
    # Fit the data and find the uncertainties
    r = opt.least_squares(residuals, p, args=(func, x, y, s))
    p_fit = r.x
    hessian = np.dot(r.jac.T, r.jac) #estimate the hessian matrix
    K_fit = np.linalg.inv(hessian) #covariance matrix
    unc_fit = np.sqrt(np.diag(K_fit)) #stdevs
    
    # rescale
    beta = np.sqrt(np.sum(residuals(p_fit, func, x, y)**2) / (x.size - len(p_fit)))
    unc_fit = unc_fit * beta
    K_fit = K_fit * beta
    
    # find the chi2 score
    chi = np.sum(residuals(p_fit, func, x, y, beta)**2) / (x.size - len(p_fit))
    
    return p_fit, chi, unc_fit

def transit(p, t, period=1.2410229) -> np.ndarray:
    """
    Produce a model of a transit light curve.
    
    Parameters
    ----------
    p : array_like 
        The parameters of the transit light curve: central transit time, semi-major axis (in units of stellar radii), planet-star radius ratio, inclination, flux out of transit.
    t : array_like
        The time series data.
    period : float, default 1.2410229
        The orbital period of the planet in days.

    Returns
    -------
    flux : numpy.ndarray
        The simulated light curve.
    """
    # split up parameters
    t0, a, rho, i, foot = p
    
    # define phi, z, k0 and k1
    phi = 2 * np.pi/period * (t - t0)
    z = a * np.sqrt(np.sin(phi)**2 + (np.cos(i) * np.cos(phi))**2)
    
    k0 = np.arccos((rho**2 + z**2 - 1) / (2 * rho * z))
    k1 = np.arccos((1 - rho**2 + z**2) / (2 * z))
    
    # find the flux
    flux = np.ones_like(t) * foot
    
    index = (z <= (1 + rho)) * (z > (1 - rho))
    
    flux[index] = foot - 1/np.pi * (rho**2 * k0[index] + k1[index] - np.sqrt((4 * z[index]**2 - (1 + z[index]**2 - rho**2)**2) / 4))
    flux[z<=(1 - rho)] = foot - rho**2

    return flux

def eclipse(p, t, period=1.2410229, a=6.378313425315667, rho=0.12621724555605157, i=1.4511148254237418) -> np.ndarray:
    """
    Produce a model of an eclipse light curve.
    
    Parameters
    ----------
    p : array_like
        The parameters of the eclipse light curve: central eclipse time, flux out of transit, scaling, eclipse depth.
    t : array_like
        The time series data.
    period : float, default 1.2410229
        The orbital period of the planet in days.
    a : float, default 6.378313425315667
        The semi major axis in units of stellar radii.
    rho : float, default 0.12621724555605157
        Planet-star radius ratio.
    i : float, default 1.4511148254237418
        The planet's orbital inclination in radians.

    Returns
    -------
    flux : numpy.ndarray
        The simulated light curve.
    """
    # default parameters were calculated in this assignment, except for the period which was given
    
    # split up parameters
    t0, foot, depth = p
    
    # define phi, z, k0 and k1
    phi = 2 * np.pi/period * (t - t0)
    z = a * np.sqrt(np.sin(phi)**2 + (np.cos(i) * np.cos(phi))**2)
    
    k0 = np.arccos((rho**2 + z**2 - 1) / (2 * rho * z))
    k1 = np.arccos((1 - rho**2 + z**2) / (2 * z))
    
    # find the flux
    flux = np.ones_like(t) * foot
    
    index = (z <= (1 + rho)) * (z > (1 - rho))

    flux[index] = foot - 1/np.pi * (depth * k0[index] + k1[index] - np.sqrt((4 * z[index]**2 - (1 + z[index]**2 - rho**2)**2) / 4))
    flux[z<=(1 - rho)] = foot - depth
    
    #scaling = scale * t

    return flux#, k0, k1, index, (depth + z**2 - 1) / (2*z*np.sqrt(depth)), (rho**2 + z**2 - 1) / (2*z*rho)

def round_sig_fig_uncertainty(value, uncertainty):
    """
    Round to the first significant figure of the uncertainty.
    
    Parameters
    ----------
    value : float or array_like
        The value(s) to be rounded.
    uncertainty : float or array_like
        The uncertaint(y/ies) in this value, must be the same size as value.

    Returns
    -------
    value_out : numpy.ndarray or float
        The rounded array of values.
    uncertainty_out : numpy.ndarray or float
        The rounded array of uncertainties.

    See Also
    --------
    round_sig_fig : Round to a given number of significant figures.
    """
    # check if numpy array/list or float/int
    if isinstance(value, np.ndarray) or isinstance(value, list):
        value_out = np.array([])
        uncertainty_out = np.array([])
        for i in range(len(value)):
            # Check if some of the values are 0
            if uncertainty[i] == 0:
                value_out = np.append(value_out, value[i])
                uncertainty_out = np.append(uncertainty_out, uncertainty[i])
            # Check if the leading digit in the error is 1, and if so round to an extra significant figure
            elif np.floor(uncertainty[i] / (10**np.floor(np.log10(uncertainty[i])))) != 1.0:
                uncertainty_rnd = np.round(uncertainty[i], int(-(np.floor(np.log10(uncertainty[i])))))
                value_rnd = np.round(value[i], int(-(np.floor(np.log10(uncertainty_rnd)))))

                value_out = np.append(value_out, value_rnd)
                uncertainty_out = np.append(uncertainty_out, uncertainty_rnd)
            else:
                uncertainty_rnd = np.round(uncertainty[i], int(1 - (np.floor(np.log10(uncertainty[i])))))
                value_rnd = np.round(value[i], int(1 - (np.floor(np.log10(uncertainty_rnd)))))

                value_out = np.append(value_out, value_rnd)
                uncertainty_out = np.append(uncertainty_out, uncertainty_rnd)
        return value_out, uncertainty_out
   
    elif isinstance(value, float) or isinstance(value, int):
        if uncertainty == 0:
            return value, uncertainty
        elif np.floor(uncertainty / (10**np.floor(np.log10(uncertainty)))) != 1.0:
            uncertainty_out = np.round(uncertainty, int(-(np.floor(np.log10(uncertainty)))))
            value_out = np.round(value, int(-(np.floor(np.log10(uncertainty_out)))))

            return value_out, uncertainty_out
        else:
            uncertainty_out = np.round(uncertainty, int(1 - (np.floor(np.log10(uncertainty)))))
            value_out = np.round(value, int(1 - (np.floor(np.log10(uncertainty_out)))))

            return value_out, uncertainty_out
    else:
        return value, uncertainty

def round_sig_fig(value, n: int) -> float:
    """
    Round to a given number of significant figures.
        
    Parameters
    ----------
    value : float or array_like
        The value(s) to be rounded.
    n : positive int
        The number of significant figures to round value to.
    
    Returns
    -------
    out : float
        The rounded value(s).

    See Also
    --------
    round_sig_fig_uncertainty : Round to the first significant figure of the uncertainty.
    """
    # Check if float/int or array/list
    if isinstance(value, float) or isinstance(value, int):
        # Turn the value into a string in scientific notation
        val_str = np.format_float_scientific(value, n)

        # Split this into the base and exponent of 10, drop the base
        expo = int(val_str.split('e')[1])

        # Round the value to this number of decimal places (minus means it goes to the left of the decimal point, also add 1 because numpy is weird in the way it rounds for some reason, it tends to round to n+1 sig fig in format_float_scientific, and in round (at least when going negative for round). Not sure why, but this corrects for it)
        out = np.round(value, -(expo - n + 1))
        return out
    elif isinstance(value, np.ndarray) or isinstance(value, list):
        out = np.array([])
        for i in range(len(value)):
            # Turn the value into a string in scientific notation
            val_str = np.format_float_scientific(value[i], n)

            # Split this into the base and exponent of 10, drop the base
            expo = int(val_str.split('e')[1])

            # Round the value to this number of decimal places
            out_val = np.round(value[i], -(expo - n + 1))
            out = np.append(out, out_val)
        return out

def kstar_func(planet_mass: float, star_mass: float, i: float, period: float) -> float:
    """
    Return the amplitude of the radial velocity curve.
  
    Parameters
    ----------
    planet_mass : float
        The mass of the planet.
    star_mass : float
        The mass of the star.
    i : float
        The orbital inclination of the planet.
    period : float
        The oribital period of the planet.
    
    Returns
    -------
    float
        The amplitude of the radial velocity curve.
    """
    return planet_mass * np.sin(i) /((star_mass + planet_mass)**(2/3)) * (2 * np.pi * const.G.value / period)**(1/3)

def kstar_uncertainty_func(planet_mass_uncertainty: float, planet_mass: float, star_mass: float, i: float, period: float) -> float:
    """
    Return the uncertainty in the amplitude of the radial velocity curve.

    Parameters
    ----------
    planet_mass_uncertainty : float
        The uncertainty in the mass of the planet.
    planet_mass : float
        The mass of the planet.
    star_mass : float
        The mass of the star.
    i : float
        The orbital inclination of the planet.
    period : float
        The oribital period of the planet.
    
    Returns
    -------
    float
        The uncertainty in the amplitude of the radial velocity curve.
    """
    return np.sqrt((np.sin(i) * (2 * np.pi * const.G.value / period)**(1/3) * (planet_mass + 3 * star_mass)/(3 * (planet_mass + star_mass)**(5/3)))**2 * planet_mass_uncertainty**2)

def find_planet_mass(star_mass: float, i: float, period: float, kstar_value: float, kstar_value_uncertainty: float, planet_mass_guess: float, planet_mass_uncertainty_guess: float) -> float:
    """
    Numerically find the planet's mass given a value for kstar

    Parameters
    ----------
    star_mass : float
        The mass of the star.
    i : float
        The orbital inclination of the planet.
    period : float
        The oribital period of the planet.
    kstar : float
        The value of kstar to be solved for.
    Kstar_uncertainty : float
        The value of the uncertainty in kstar to be solved for.
    planet_mass_guess : float
        The initial guess at the mass of the planet
    planet_mass_uncertainty_guess : float
        The initial guess at the uncertainty in the mass of the planet.

    Returns
    -------
    mass : float
        The mass of the planet.
    """
    f = lambda planet_mass: kstar_func(planet_mass, star_mass, i, period) - kstar_value
    mass = opt.fsolve(f, planet_mass_guess)

    f = lambda planet_mass_uncertainty: kstar_uncertainty_func(planet_mass_uncertainty, mass, star_mass, i, period) - kstar_value_uncertainty
    mass_uncertainty = opt.fsolve(f, planet_mass_uncertainty_guess)
    return mass[0], mass_uncertainty[0]