import numpy as np
import astropy.units as u
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import os

os.chdir('/Users/dee/Documents/ALMA/chary_elbaz_python')
import chary_elbaz as ce


def luminosity(z, flux, cosmol):
    '''flux to luminosity conversion'''
    return flux * 4 * np.pi * cosmol.luminosity_distance(z).to(u.cm) ** 2


def nulnu(z, flux, lam, cosmol):
    lum = luminosity(z, flux*u.Jy, cosmol)
    nulnu = lum*(lam*u.um).to(u.Hz, equivalencies=u.spectral())
    return nulnu.to(u.erg/u.s).to(u.Lsun)


def get_mstar(redshift, irac, dirac, templatefile, cosmo, corr, tol=0.005):
    ''' Function to return the stellar mass of a sample of galaxies given their 3.6um fluxes and a Bruzual+Charlot
        (2003) stellar population template. Assumes a Salpeter IMF (the * 1.65) and the uncertainty is based on the
        scatter in a comparison w the FAST SED fit. corr is the scaling correction to match the FAST SED fit
    '''

    # make sure everything's an array in case you're passing values from a dataframe
    redshift = np.array(redshift, dtype=np.float64)
    irac = np.array(irac, dtype=np.float64)
    dirac = np.array(dirac, dtype=np.float64)

    # rough guess (no K-corr)
    roughmstar = (nulnu(redshift, irac, 3.6, cosmo).value)*u.Msun
    # restframe wavelengths
    lamrest = 3.6/(1+redshift)*u.um

    # load in the template
    spectemp = np.genfromtxt(templatefile)
    # get the relative fluxes from the template
    lam = (spectemp[:,0]*u.angstrom).to(u.um)
    tempflux = spectemp[:,1]

    # emitted relative fluxes
    tempfluxrel = []
    tol = tol*u.um
    for lamemval in lamrest:
        guesses = tempflux[np.where(np.logical_and(lam > lamemval-tol, lam < lamemval+tol))]
        tempfluxrel.append(np.mean(guesses))
    tempfluxrel = np.array(tempfluxrel)

    kfluxrestframe = tempflux[np.where(np.logical_and(lam > 2.1*u.um, lam < 2.15*u.um))[0]][0]

    # k-corrections
    kcorrection = kfluxrestframe / tempfluxrel

    mstar = roughmstar*kcorrection*1.65
    if np.any(np.isnan(dirac)):
        dmstar = (1-10**-0.3)*mstar
    else:
        dmstar = (1-10**-0.3)*mstar + (dirac/irac)*mstar

    return (mstar*corr).value, (dmstar*corr).value


def sfr_ce_ken(redshift, mips, dmips=-999):
    # mips should be in uJy

    # make sure everything's an array in case you're passing values from dataframes
    redshift = np.array(redshift)
    mips = np.array(mips)
    dmips = np.array(dmips)

    # first get the ir luminosity with the CE01 code
    lumvals = [ce.fit_SED(z=redshift[0], S_nu=mips[0]*u.uJy, wavelength=24*u.um, H0=70, use_IRAS=False)]
    for i in np.arange(1, len(redshift)):
        lum = ce.fit_SED(z=redshift[i], S_nu=mips[i]*u.uJy, wavelength=24*u.um, H0=70, use_IRAS=False)
        lumvals.extend([lum])

    # then the SFR from kennicutt 98 equation
    kensfr = np.array([(lumvals*u.L_sun).to(u.erg/u.s) * 4.5e-44])[0]

    # uncertainty (based on 0.15dex scatter from kennicutt 98)
    if np.all(dmips == -999):
        dkensfr = kensfr*(1-10**-0.15)
    elif np.any(np.isnan(dmips)):
        dkensfr = kensfr*(1-10**-0.15)
    else:
        dkensfr = kensfr*(1-10**-0.15) + (dmips/mips)*kensfr

    return kensfr, dkensfr



def sfr_rieke(redshift, mips, cosmo, dmips=-999):
    # mips needs to be in uJy

    # make sure everything's an array in case you're passing values from dataframes
    redshift = np.array(redshift)
    mips = np.array(mips)
    dmips = np.array(dmips)

    # table of A and B coefficients from rieke 2009
    riekez = np.array([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.0, 2.2])
    A = np.array([0.417, 0.502, 0.528, 0.573, 0.445, 0.358, 0.505, 0.623, 0.391, 0.072, 0.013, 0.029])
    B = np.array([1.032, 1.169, 1.272, 1.270, 1.381, 1.565, 1.745, 1.845, 1.716, 1.642, 1.639, 1.646])

    # find the A and B coefficients for the passed redshifts by interpolation
    Acoeffs = np.interp(redshift, riekez, A)
    Bcoeffs = np.interp(redshift, riekez, B)

    # flux at 24um
    fnu = (mips*u.uJy).to(u.Jy).value
    DL = (cosmo.luminosity_distance(redshift)).to(u.cm).value

    sfrrieke = np.power(10, (Acoeffs + Bcoeffs * (np.log10(4*np.pi*DL**2*fnu) - 53)))

    if np.all(dmips == -999):
        # scatter of 0.2 dex from Rieke 09
        dsfrrieke = (1-10**-0.2)*sfrrieke

    else:
        # scatter of 0.2 dex from rieke 09 plus the uncert in the mips measurements
        dsfrrieke = (1-10**-0.2)*sfrrieke + (dmips/mips)*sfrrieke

    return sfrrieke, dsfrrieke


def get_properties(frame, templatefile, cosmo, corr, tol=0.005):
    '''
    Function to, when passed a pandas dataframe containing mips/irac/redshift info for a given
    galaxy population, append the stellar mass, (CE/Ken) SFR, and sSFR values for each galaxy
    onto the dataframe

    depends on the chary and elbaz python code and a bruzual and charlot template
    assumes MIPS values are in Jy
    '''

    # first get the star formation rate
    sfrvals, dsfrvals = sfr_ce_ken(frame.z, frame.mips*1e6, frame.dmips*1e6)

    # then the stellar mass
    mstarvals, dmstarvals = get_mstar(frame.z, frame.irac, frame.dirac, templatefile, cosmo, corr,
                                      tol=tol)

    # then find ssfrs
    ssfrvals = sfrvals / mstarvals * 1e9
    dssfrvals = ssfrvals * np.sqrt((dsfrvals/sfrvals)**2 + (dmstarvals/mstarvals)**2)

    # then gas fractions
    fgasvals = frame.mgas / mstarvals
    dfgasvals = fgasvals * np.sqrt((frame.dmgas/frame.mgas)**2 + (dmstarvals/mstarvals)**2)

    # and star formation efficiencies
    sfevals = sfrvals / frame.mgas * 1e9
    dsfevals = sfevals * np.sqrt((dsfrvals/sfrvals)**2 + (frame.dmgas/frame.mgas)**2)

    newframe = frame.copy()

    # add everything into the dataframe as new columns
    newframe['sfr'] = sfrvals
    newframe['dsfr'] = dsfrvals
    newframe['mstar'] = mstarvals
    newframe['dmstar'] = dmstarvals
    newframe['ssfr'] = ssfrvals
    newframe['dssfr'] = dssfrvals
    newframe['fgas'] = fgasvals
    newframe['dfgas'] = dfgasvals
    newframe['sfe'] = sfevals
    newframe['dsfe'] = dsfevals

    return newframe
