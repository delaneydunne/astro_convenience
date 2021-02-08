import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import scipy.odr as odr



def coordsearch(searchra, searchdec, catra, catdec, catval=None, tol=10):
    """
    Function to search for objects in a catalogue based on their coordinates. Returns
    -99 if a given object isn't found within the tolerance and automatically picks the
    best match if there are several objects found within the tolerance
    INPUTS:
        searchra: right ascension of the object(s) you want to find in the catalogue, in
                  degrees (but passed unitless)
        searchdec: declination of the object(s) you want to find in the catalogue, in
                  degrees (but passed unitless)
        catra:    list/array containing the ra of each object in the catalogue, in degrees
                  (but passed unitless)
        catdec:   list/array containing the dec of each object in the catalogue, in
                  degrees (but passed unitless)
        catval:   (optional) list/array containing the value you want to retrieve from the
                  catalogue for each object
        tol:      tolerance (in arcseconds) for the search

    OUTPUTS:
        indices:   if catval wasn't passed, the indices of the object(s) in the passed
                   catalogue
        foundvals: if catval was passed, the values found for each object(s) in the passed
                   catalogue
    """

    tol = tol*u.arcsec
    searchra, searchdec = np.array(searchra), np.array(searchdec)
    searchcoord = SkyCoord(ra=searchra*u.deg, dec=searchdec*u.deg, frame='fk5')

    catra, catdec = np.array(catra), np.array(catdec)
    catcoords = SkyCoord(ra=catra*u.deg, dec=catdec*u.deg, frame='fk5')

    foundidx = []
    if catval:
        foundval = []

    for coord in searchcoord:
        linenumbers = np.where(catcoords.separation(coord) < tol)[0]

        seps = catcoords[linenumbers].separation(coord).value

        if seps.size == 0:
            foundidx.append(-99)
            if catval:
                foundval.append(-99)
        else:
            best = np.min(seps)
            bestidx = linenumbers[np.where(seps == best)]

            foundidx.append(bestidx[0])
            if catval:
                foundval.append(catval[bestidx[0]])

    if catval:
        return np.array(foundval)
    else:
        return np.array(foundidx)



def bandwidth_vel_to_freq(restfreq, delta_v, z):
    '''
        Calculates the frequency width of a band with velocity width delta_v.
        INPUTS:
            restfreq:  The rest-frame frequency of the center of the band (if CO(2-1), then
                       this should be 230.538GHz). Whatever units are used here are the units
                       that will be returned
            delta_v:   the velocity width of the desired band (km/s)
            z:         Target redshift
        OUTPUTS:
            delta_nu:  The bandwidth as a frequency value
    '''
    obsfreq = restfreq / (1 + z)
    delta_nu = obsfreq * delta_v / 2.99792e5

    return delta_nu




# orthogonal distance reduction fitting
def odrstraightline(B, x):
    '''
        y = mx+b function
        INPUTS:
            B: a 2-entry array where B[0] is the slope and B[1] is the y-intercept
            x: the x-values
    '''
    return B[0]*x + B[1]


def odr_fit(xvals, yvals, dxvals, dyvals, beta0, logx=True):
    '''
        Runs through all the steps to do a scipy orthogonal distance reduction fit to the
        values LOGGED. For details check out:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Output.html#scipy.odr.Output
        INPUTS:
            xvals:  the array of x values to be fit (assumed to be in linspace if logx=True,
                    else assumed to be in logspace)
            yvals:  the array of y values to be fit (should always be in linspace)
            dxvals: uncertainties in the x-axis (should be in linspace, the function will
                    log them properly on its own)
            dyvals: uncertainties in the y-axis (should be in linspace, the function will
                    log them properly on its own)
            beta0:  initial guess at the fitting parameters. 2-entry array of the form
                    [slope, y-intercept]
            logx:   optional, default=True: if true, will assume the x-values are linear and
                    take the log10. If false, x-values should be passed logged

        OUTPUTS:
            lineodr.run: a scipy ODR object. output.beta will give the fit parameters and
                         output.sd_beta will give the standard deviation in those fitting parameters
    '''

    if logx:
        lxvals = np.log10(xvals)
        dlxvals = np.abs(dxvals / (xvals * np.log(10)))
    else:
        lxvals = xvals
        dlxvals = dxvals

    lyvals = np.log10(yvals)
    dlyvals = np.abs(dyvals / (yvals * np.log(10)))

    def odrstraightline(B, x):
        return B[0]*x + B[1]

    linemodel = odr.Model(odrstraightline)
    linedata = odr.RealData(lxvals, lyvals, sx=dlxvals, sy=dlyvals)
    lineodr = odr.ODR(linedata, linemodel, beta0=beta0)

    return lineodr.run()
