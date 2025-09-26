import numpy as np
import pandas as pd
import astropy.units as u
from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord

def mag(zp, Flux):
    '''
    Transform the flux to magnitude
    inputs
    zp: zero point
    Flux: vector that contains the lightcurve flux
    '''
    return zp - 2.5 * np.log10(abs(Flux))

def deviation_from_constant(pyLIMA_parameters, pyLIMA_telescopes):
    '''
     There at least four points in the range
     $[t_0-tE, t_0+t_E]$ with the magnification deviating from the
     constant flux by more than 3$sigma$
    '''
    ZP = {'W149': 27.615, 'u': 27.03, 'g': 28.38, 'r': 28.16,
          'i': 27.85, 'z': 27.46, 'y': 26.68}
    t0 = pyLIMA_parameters['t0']
    tE = pyLIMA_parameters['tE']
    satis_crit = {}
    for telo in pyLIMA_telescopes:
        if not len(telo.lightcurve['mag']) == 0:
            mag_baseline = ZP[telo.name] - 2.5 * np.log10(pyLIMA_parameters['ftotal_' + f'{telo.name}'])
            x = telo.lightcurve['time'].value
            y = telo.lightcurve['mag'].value
            z = telo.lightcurve['err_mag'].value
            mask = (t0 - tE < x) & (x < t0 + tE)
            consec = []
            if len(x[mask]) >= 3:
                combined_lists = list(zip(x[mask], y[mask], z[mask]))
                sorted_lists = sorted(combined_lists, key=lambda item: item[0])
                sorted_x, sorted_y, sorted_z = zip(*sorted_lists)
                for j in range(len(sorted_y)):
                    if sorted_y[j] + 3 * sorted_z[j] < mag_baseline:
                        consec.append(j)
                result = has_consecutive_numbers(consec)
                if result:
                    satis_crit[telo.name] = True
                else:
                    satis_crit[telo.name] = False
            else:
                satis_crit[telo.name] = False
        else:
            satis_crit[telo.name] = False
    return any(satis_crit.values())

def filter5points(pyLIMA_parameters, pyLIMA_telescopes):
    '''
    Check that at least one light curve
    have at least 5 pts in the t0+-tE
    '''
    t0 = pyLIMA_parameters['t0']
    tE = pyLIMA_parameters['tE']
    crit5pts = {}
    for telo in pyLIMA_telescopes:
        if not len(telo.lightcurve['mag']) == 0:
            x = telo.lightcurve['time'].value
            mask = (t0 - tE < x) & (x < t0 + tE)
            if len(x[mask]) >= 5: #cambiar a 10, 15
                crit5pts[telo.name] = True
            else:
                crit5pts[telo.name] = False
    return any(crit5pts.values())

def FilterNpoints(n, pyLIMA_parameters, pyLIMA_telescopes):
    '''
    Check that at least one light curve
    have at least N pts in the t0+-tE
    '''
    t0 = pyLIMA_parameters['t0']
    tE = pyLIMA_parameters['tE']
    critNpts = {}
    for telo in pyLIMA_telescopes:
        if not len(telo.lightcurve['mag']) == 0:
            x = telo.lightcurve['time'].value
            mask = (t0 - tE < x) & (x < t0 + tE)
            if len(x[mask]) >= n: #cambiar a 10, 15
                critNpts[telo.name] = True
            else:
                critNpts[telo.name] = False
    return any(critNpts.values())

def filter_band(lightcurve, m5, fil):
    '''
    * Save the points of the lightcurve greater and smaller than
      1sigma fainter and brighter that the saturation and 5sigma_depth
    * check that the lightcurve have more than 10 points
    * check if the lightcurve have at least 1 point at 5 sigma from the 5sigma_depth
    '''
    # print(len(lightcurve))
    mag_sat = {'W149': 14.8, 'u': 14.7, 'g': 15.7, 'r': 15.8, 'i': 15.8, 'z': 15.3, 'y': 13.9}
    lightcurve['m5'] =  m5

    b1 = lightcurve['mag'].value - lightcurve['err_mag'].value > mag_sat[fil]
    b2 = lightcurve['mag'].value + lightcurve['err_mag'].value < lightcurve['m5']
    lc_fil1 = lightcurve[b1&b2]
    # display(lc_fil1)
    return lc_fil1

def has_consecutive_numbers(lst):
    """
    check if there at least 3 consecutive numbers in a list lst
    """
    sorted_lst = sorted(lst)
    for i in range(len(sorted_lst) - 2):
        if sorted_lst[i] + 1 == sorted_lst[i + 1] == sorted_lst[i + 2] - 1:
            return True
    return False
