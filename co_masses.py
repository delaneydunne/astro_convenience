import numpy as np
from photutils.centroids import fit_2dgaussian
from astropy.modeling.functional_models import Ellipse2D
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from astropy.constants import si
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import quad


def get_beam_crtf(file, nbeam):
    '''
        files (cube (fits) file)

    '''

    # open the cube file to get the size of the beam
    cubehdu = fits.open(file)[0]
    cubedata = cubehdu.data[0, 0, :, :]
    cubewcs = wcs.WCS(cubehdu.header).sub(['celestial'])

    # major and minor axes
    bmaj, bmin = cubehdu.header['BMAJ'] / 2, cubehdu.header['BMIN'] / 2  # ***CONVERT FROM ARCSEC?
    # center pixel
    xcent, ycent = cubehdu.header['CRPIX1'], cubehdu.header['CRPIX2']
    # beam angle
    theta = cubehdu.header['BPA']

    # open a file to store the region information
    fname = 'beam_region_' + str(nbeam) + 'sigma.crtf'
    f = open(fname, "w")
    f.write("#CRTF\n")
    f.write('ellipse[[' + str(xcent) + 'pix, ' + str(ycent) + 'pix], [' + str(bmaj * nbeam) + 'deg, '
            + str(bmin * nbeam) + 'deg], ' + str(theta) + 'deg]')
    return


def get_mom0_chans(specfile, nstd, rechanfile=None, p0=[0.003, 67200, 50, 0], centmean=True, linefree=(0, 50),
                   zvals=(90, 160), z=None, centline=True, n_gauss=1, p02=[0.003, 0.003, -100, 100, 100, 100, 0],
                   plot=1, bounds=(np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
                                   np.array([np.inf, np.inf, np.inf, np.inf]))):
    # if len(specfile) > 1, first make and fit a spectral profile using the file with larger channel widths (which
    # will come first), and then use this standard deviation to get the number of channels from the 25km/s file
    # (which will come second) - don't need to fit the 25 km/s file, just have the values to get vel from

    if plot == 1:
        plotmgas = True
        plotchans = False
    elif plot == 0:
        plotmgas = False
        plotchans = True
    else:
        plotmgas = True
        plotchans = True

        # specmax, rms, opt, stdmat, hzmean, uncert

    # create and fit a spectral profile
    specmax, rms, opt, stdmat, centfreq, uncert = get_vel_from_txt(specfile, p0=p0, centmean=centmean,
                                                                   linefree=linefree,
                                                                   n_gauss=1, zvals=zvals, z=z, centline=centline,
                                                                   plot=plotchans,
                                                                   p02=p02)

    # from the standard deviation and mean returned from the fit in opt, define a range in km/s about the mean
    # from which to take the moment 0 channels
    mean, std = opt[1], opt[2]
    xmin, xmax = mean - nstd * std, mean + nstd * std
    print(mean)

    print("1-gaussian FWHM: {:.3e}".format(std))

    if plotchans == True:
        plt.axvline(x=xmin, zorder=20, color='k', ls='--')
        plt.axvline(x=xmax, zorder=20, color='k', ls='--')

    # turn the xmin and xmax into channel values somehow

    if rechanfile:
        # load in the spectral profile from the file with wider channels
        specprof = np.genfromtxt(rechanfile)
        vel, flux = specprof[:, 0], specprof[:, 1]
    else:
        # load in the spectral profile from the file
        specprof = np.genfromtxt(specfile)
        vel, flux = specprof[:, 0], specprof[:, 1]

    print(xmin, xmax)

    if xmin > -1000 and xmax < 1000:

        if xmin > xmax:
            c = xmin
            xmin = xmax
            xmax = c
        # if the center of the channel is included inside the n sigma region, include it. otherwise, don't
        chanmin = np.min(np.where(vel > xmin))
        chanmax = np.max(np.where(vel < xmax))

        print("'{}~{}'".format(chanmin, chanmax))

        print(vel[chanmin], vel[chanmax])

        if plotchans == True:
            plt.axvline(x=vel[chanmin], zorder=20, color='k', ls=':')
            plt.axvline(x=vel[chanmax], zorder=20, color='k', ls=':')

    else:
        print('WARNING: region very large. Theres probably something wrong with the specprof file')

    if n_gauss == 2:
        # create and fit a spectral profile
        specmax, rms, opt, centfreq, uncert = get_vel_from_txt(specfile, p0=p0, centmean=centmean, linefree=linefree,
                                                               n_gauss=2, zvals=zvals, z=z, centline=centline,
                                                               plot=False)

        print("2-gaussian stds: {}, {}".format(opt[4], opt[5]))

    mgas = get_M_gas_txt(specfile, z, linefree=linefree, n_gauss=n_gauss, zvals=zvals, centline=centline,
                         p0=p0, center_mean=centmean, p02=p02, plot=plotmgas)
    print("Gas mass: {:.3E}".format(mgas[0]))


def get_FWHMa(data, cutout):
    ''' Use photutils' fit_2dgaussian function to determine the FWHM of an elliptical signal. Data should be the whole
        that is to be fit, and cutout is [xmin, xmax] in native pixels around the signal. returns an array
        of [x FHWM, y FHWM]
    '''
    gfit = fit_2dgaussian(data[cutout[0]:cutout[1], cutout[0]:cutout[1]])
    # get the Gaussian FWHM from the fitted model: FWHM = sigma * sqrt(8 ln(2))
    stds = np.array([gfit.x_stddev.value, gfit.y_stddev.value])
    FWHMa = stds * np.sqrt(8 * np.log(2))

    return FWHMa


def plot_gauss_fit(data, scicoords, cutout):
    '''plots the gaussian fitted to the signal using photutils' fit_2dgaussian function, as well as residuals and the 
       original signal. data should be the entire frame, and cutout is [xmin, xmax] in native pixels around the signal
    '''

    xmin, xmax = cutout[0], cutout[1]

    scifit = fit_2dgaussian(data[xmin:xmax, xmin:xmax])

    # fitted gaussian
    scieval = scifit.evaluate(scicoords[:, :, 0], scicoords[:, :, 1], scifit.constant.value, scifit.amplitude.value,
                              scifit.x_mean.value + xmin, scifit.y_mean.value + xmin, scifit.x_stddev.value,
                              scifit.y_stddev.value, scifit.theta.value)

    # colormap extrema
    vmin, vmax = np.min(data), np.max(data)

    # residuals
    resids = data - scieval

    # plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    # original
    im = axs[0].pcolormesh(data[xmin:xmax, xmin:xmax], cmap=plt.cm.inferno, vmin=vmin, vmax=vmax)
    axs[0].set_aspect(aspect=1)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.07, ax=axs[0])

    # fitted gaussian
    im = axs[1].pcolormesh(scieval[xmin:xmax, xmin:xmax], cmap=plt.cm.inferno, vmin=vmin, vmax=vmax)
    axs[1].set_aspect(aspect=1)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.07, ax=axs[1])

    # residuals
    resids = data - scieval
    im = axs[2].pcolormesh(resids[xmin:xmax, xmin:xmax], cmap=plt.cm.gist_rainbow)
    axs[2].set_aspect(aspect=1)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.07, ax=axs[2])
    return


def line_mask(data, cutout, use_stds=False, nstds=4, ycutout=None):
    ''' Function to, when passed a frame with a spectral line signal, determine a 4 FWHM elliptical region about
        the center of the source. returns a 2D array with the ellipse pixels=1 and all other pixels=0. Data
        should be a single frame and cutout should be a region of the form [xmin,xmax] around the center of 
        the desired signal. if use_stds is true, 4 sigma will be used instead of 4 FWHMa
    '''

    xmin, xmax = cutout[0], cutout[1]

    if ycutout:
        ymin, ymax = ycutout[0], ycutout[1]

    # to get theta and the mean values, fit
    datafit = fit_2dgaussian(data[xmin:xmax, xmin:xmax])
    if ycutout:
        datafit = fit_2dgaussian(data[xmin:xmax, ymin:ymax])

    if use_stds == True:
        # use the standard deviation values to calculate the limits of the ellipse
        FWHMa = (datafit.x_stddev.value, datafit.y_stddev.value)
    else:
        # fit the cutout data to a 2D gaussian ellipse to determine the FWHMa. call get_FWHMa to do this
        FWHMa = get_FWHMa(data, cutout)

    xcent = datafit.x_mean.value + xmin
    if ycutout:
        ycent = datafit.y_mean.value + ymin
    else:
        ycent = datafit.y_mean.value + xmin

    # ellipse2d object with axes 4*fwhma
    ell = Ellipse2D(1, xcent, ycent, nstds * FWHMa[0], nstds * FWHMa[1],
                    datafit.theta.value)

    meanvals = (datafit.x_mean.value, datafit.y_mean.value)

    # x and y coords for evaluation
    datx = np.arange(np.shape(data)[0])
    x, y = np.meshgrid(datx, datx)
    return ell(x, y), FWHMa, meanvals, datafit.theta.value


def int_gaussian(amp, mu, sig, C):
    '''integral of a gaussian function assuming the bounds are essentially infinite.
       Ignores the constant offset because it's assumed to be part of the continuum, but it's included
       in the input parameters to make it easy to just unpack opt
       (This equation is just taken from wikipedia)'''
    # NOT CURRENTLY BEING USED
    return np.sqrt(2 * np.pi) * amp * np.abs(sig)


def gaussian(x, amp, mu, sig, C):
    '''simple 1d gaussian function with included offset'''
    return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))  # + C


def twogaussian(x, amp1, amp2, mu1, mu2, sig1, sig2, C):
    gauss1 = amp1 * np.exp(-np.power(x - mu1, 2.) / (2 * np.power(sig1, 2.)))  # + C
    gauss2 = amp2 * np.exp(-np.power(x - mu2, 2.) / (2 * np.power(sig2, 2.)))  # + C
    return gauss1 + gauss2


def gauss_area_uncert(optmat, stdmat):
    # area under a gaussian is A = sqrt(2*pi)*a*|c|
    #  => \sigma_A = sqrt(2*pi)*|A|*sqrt((da/a)^2 + (dc/c)^2 + 2(dac/a*c))

    optmat = np.abs(optmat)
    stdmat = np.abs(stdmat)

    A = np.sqrt(2 * np.pi) * optmat[0] * optmat[2]

    sigA = np.sqrt((stdmat[0, 0] / optmat[0]) ** 2 + (stdmat[2, 2] / optmat[2]) ** 2) * np.abs(
        A)  # + 2*(stdmat[0,2]/optmat[0]/optmat[2])

    return sigA, A


def twogauss_area_uncert(optmat, stdmat):
    #  amp1, amp2, mu1, mu2, sig1, sig2, C

    optmat = np.abs(optmat)
    stdmat = np.abs(stdmat)

    A1 = np.sqrt(2 * np.pi) * (optmat[0] * optmat[4])
    A2 = np.sqrt(2 * np.pi) * (optmat[1] * optmat[5])
    A = A1 + A2

    sigA1_2 = ((stdmat[0, 0] / optmat[0]) ** 2 + (stdmat[4, 4] / optmat[4]) ** 2) * A1 ** 2
    sigA2_2 = ((stdmat[1, 1] / optmat[1]) ** 2 + (stdmat[5, 5] / optmat[5]) ** 2) * A2 ** 2
    sigA = np.sqrt(sigA1_2 + sigA2_2)

    return sigA, A


def rmsfunc(obs, linefree=(0, 50)):

    rmsval = np.sqrt(np.sum(obs[linefree[0]:linefree[1]] ** 2) / len(obs[linefree[0]:linefree[1]]))

    return rmsval


def chi(obs, exp, nfit, linefree=(0, 50)):

    rmsval = rmsfunc(obs, linefree)
    chisquare = np.sum((obs - exp) ** 2) / (rmsval ** 2 * (len(obs) - nfit))
    return chisquare


def get_crtf(momfile, cutout, use_stds=True, nstds=3, ycutout=None):
    # open the momentfile
    hdu = fits.open(momfile)[0]
    mom = hdu.data
    momwcs = wcs.WCS(hdu.header).sub(['celestial'])
    mom = mom[0, 0, :, :]
    mom[np.where(np.isnan(mom))] = 0.

    # get the elliptical mask by fitting
    window, FWHMa, means, theta = line_mask(mom, cutout, use_stds=use_stds, nstds=nstds, ycutout=ycutout)
    theta = (theta*u.rad).to(u.deg)
    if ycutout:
        means = np.add(means, (cutout[0], ycutout[0]))
    else:
        means = np.add(means, cutout[0])
    FWHMa = np.multiply(FWHMa, nstds)

    # plot the masked moment 0 as a sanity check
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=momwcs)
    im = plt.pcolormesh(window * mom)
    fig.colorbar(im)

    # open a file to store the region information
    fname = 'elliptical_mask_' + str(nstds) + 'sigma.crtf'
    f = open(fname, "w")
    f.write("#CRTF\n")
    f.write('ellipse[[' + str(means[0]) + 'pix, ' + str(means[1]) + 'pix], [' + str(FWHMa[0]) + 'pix, '
            + str(FWHMa[1]) + 'pix], ' + str(theta.value + 90) + 'deg]')

    return


from scipy.integrate import quad


def get_vel_from_txt(specfile, p0=[0.3, 67200, 50, 0], centmean=True, p02=None, linefree=(0, 50), n_gauss=None,
                     zvals=(90, 160), z=None, centline=False, plot=True,
                     bounds=(np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
                             np.array([np.inf, np.inf, np.inf, np.inf]))):
    # load in the spectral profile from the file
    with open(specfile) as file:
        specprof = np.genfromtxt(file)
    vel, flux = specprof[:, 0], specprof[:, 1]

    zmin, zmax = zvals[0], zvals[1]
    velmin, velmax = np.min(vel), np.max(vel)

    # xarray to fit
    x = np.linspace(velmin, velmax, num=500)

    if plot:
        fig, axs = plt.subplots(1, figsize=(10, 4))

        # plot the intensity data to make sure the fit worked
        axs.bar(vel, flux * 1e3, width=np.abs(vel[1] - vel[0]))
        axs.set_xlabel('Velocity (km/s), LSRK)')
        axs.set_ylabel('Integrated Intensity (mJy)')

    # adjust the guess fit parameters to match the data
    if centmean:
        meanval = np.mean(x)
        p0[1] = meanval
    if not p02:
        # if no fit parameters are passed, guess based off the fit parameters for one gaussian
        p02 = [p0[0], p0[0], p0[1] - 50, p0[1] + 50, p0[2], p0[2], 0]

    rmssigval = rmsfunc(flux, linefree=linefree)
    print("rmssigval: {:.3f}".format(rmssigval))
    rmssig = np.ones(len(vel)) * rmssigval

    # fit to a gaussian
    opt1, cov1 = curve_fit(gaussian, vel, flux, p0=p0, method='trf', sigma=rmssig, bounds=bounds)
    # fit to two gaussians
    opt2, cov2 = curve_fit(twogaussian, vel, flux, p0=p02, method='trf', sigma=rmssig)

    # find the reduced chi-squared value for each fit
    chi1 = chi(flux[zmin:zmax], gaussian(vel[zmin:zmax], *opt1), 4, linefree=linefree)
    chi2 = chi(flux[zmin:zmax], twogaussian(vel[zmin:zmax], *opt2), 8, linefree=linefree)

    # print out the chi-squared values for each fit
    print("chi-squared for one Gaussian: {:.3f}".format(chi1))
    print("chi-squared for two Gaussians: {:.3f}".format(chi2))

    if not n_gauss:
        if chi1 < chi2:
            use_twog = False
            rms = rmssigval
            opt = opt1
            cov = cov1
        elif chi1 > chi2:
            use_twog = True
            rms = rmssigval
            opt = opt2
            cov = cov2
    else:
        if n_gauss == 1:
            use_twog = False
            rms = rmssigval
            opt = opt1
            cov = cov1
        elif n_gauss == 2:
            use_twog = True
            rms = rmssigval
            opt = opt2
            cov = cov2
        else:
            print("Warning: Calling for a number other than 1 or 2 Gaussians")
            return

    print("using a two-gaussian fit: {}".format(use_twog))

    if plot:
        if use_twog == False:
            axs.plot(x, gaussian(x, *opt1) * 1e3, color='orange')
        else:
            axs.plot(x, twogaussian(x, *opt2) * 1e3, color='orange')

    # find area under gaussian fit - integration means -> Jy*km/s
    specmax1 = quad(gaussian, np.min(vel), np.max(vel),
                    args=(opt1[0], opt1[1], opt1[2], 0))[0]
    print("One-gaussian integrated flux: {:.3e} Jy km/s".format(specmax1))

    specmax2 = quad(twogaussian, np.min(vel), np.max(vel),
                    args=(opt2[0], opt2[1], opt2[2], opt2[3], opt2[4], opt2[5], opt2[6]))[0]
    print("Two-gaussian integrated flux: {:.3e} Jy km/s".format(specmax2))

    centfreq = 230.538 * u.GHz / (1 + z)
    freq_to_vel = u.doppler_radio(230.528 * u.GHz)
    centvel = centfreq.to(u.km / u.s, equivalencies=freq_to_vel)

    if not use_twog:
        specmax = specmax1
        hzmean = (opt1[1] * u.km / u.s + centvel).to(u.Hz, equivalencies=freq_to_vel)
        print('hzmean (usetwog false): {}'.format(hzmean))
    else:
        specmax = specmax2
        hzmean = (np.mean((opt2[2], opt2[3])) * u.km / u.s + centvel).to(u.Hz, equivalencies=freq_to_vel)
        print('hzmean (usetwog true) : {}'.format(hzmean))

    if centline:
        if plot:
            axs.axvline(x=0, zorder=20, color='k')

    stdmat = np.sqrt(cov)

    if n_gauss == 1:
        uncert, funcA = gauss_area_uncert(opt, stdmat)
    else:
        uncert, funcA = twogauss_area_uncert(opt, stdmat)

    print("area from calculation: {:.3e}".format(funcA))
    print("uncertainty on area: {:.3e}".format(uncert))

    return specmax, rms, opt, stdmat, hzmean, uncert


def lco_uncert(A, z, nuobs, Lco, dA, dz, dnuobs, cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):
    # ignoring uncertainty in the luminosity distance for now (should come from dz?)

    # Lco = 3.25e7 * A * D_L^2 / ((1+z)^3 * nuobs^2) -> calling the top part C and the bottom part B
    print("dnuobs, nuobs: {}, {}".format(dnuobs, nuobs))

    dL = np.sqrt((2 * dnuobs / nuobs) ** 2 + (3 * dz / (1 + z)) ** 2 + (dA / A) ** 2)
    #     dB = np.sqrt((2*dnuobs/nuobs)**2 + (3*dz/(1+z))**2)

    #     f = co.co_line_luminosity(A, nuobs, z, cosmo)

    #     dL = f * np.sqrt((dA / A)**2 + (dB)**2)

    return dL * Lco

'''
def get_velocity_int_flux(moment_file, cube_file, cutout, p0, use_stds=True, nstds = 4, center_mean=True, p02=None):
    '''''' Function to, when passed an image cube and the Moment-0 map calculated from the cube, calculate and
        return the CO intensity in two different ways: firstly, by finding the point of maximum flux in the
        moment 0 map and returning that value as Jy/km/s, and secondly by defining a 4-sigma region about the
        center of a 2d elliptical gaussian fit to the signal, integrating over the region in each channel to
        determine the spectral profile of the signal, and fitting the profile to a gaussian function. The
        area under this second Gaussian is another measure of the maximum intensity.

        INPUTS: moment_file: file pointer to the moment 0 plot
                cube_file: file pointer to the image cube
                cutout: (xmin, xmax) region about the center of the signal in the moment file
                p0 = [amplitude, mean, standard deviation, offset]: best-guess parameters for spectral gaussian
                                                                    fit. will automatically adjust mean to be the
                                                                    center of the velocity axis and automatically
                                                                    adjust for two gaussians
        OUTPUTS: mommax: value of the most intense pixel in the moment 0 plot
                 specmax: area under a gaussian fit to the spectral profile of the 4sigma region about the
                          signal in Jy
                 opt: the fit parameters for the equation that best fits the data (either entries for gaussian or
                      for twogaussian)'''
'''

    # get data from the moment 0 plot
    hdul = fits.open(moment_file)
    sci = hdul['PRIMARY'].data
    momwcs = wcs.WCS(hdul[0].header)
    sci = sci[0, 0, :, :].astype('float64')

    # pass the nans in the sci array to zeros
    sci[np.where(np.isnan(sci))] = 0.

    # beam area in arcseconds^2
    abeam = (np.pi * hdul[0].header['BMAJ'] * hdul[0].header['BMIN'] * u.deg ** 2).to(u.arcsec ** 2).value / 4

    mommax = np.nanmax(sci)  # * abeam  # value of the most intense pixel in Jy*km/s

    # prepare coordinates to get the fitted mask
    scix = np.arange(np.shape(sci)[0])
    scicoords = np.stack(np.meshgrid(scix, scix), axis=2).astype('float64')

    # elliptical 4sigma mask about the mean of the moment signal
    window, FWHMa = line_mask(sci, cutout, use_stds=use_stds, nstds=nstds)

    # sigma used to determine the mask
    sigx, sigy = FWHMa

    # get data from the image cube
    hdul = fits.open(cube_file)
    cube = hdul['PRIMARY'].data
    cubewcs = wcs.WCS(hdul[0].header)
    cubecelwcs = cubewcs.sub(['celestial'])
    # first axis is empty so discard it
    cube = cube[0, :, :, :]
    # pass NaNs to zeros
    cube[np.where(np.isnan(cube))] = 0.0

    # sometimes the moment 0 map and the image cube have different spatial dimensions - this adjusts the
    # mask to agree with the cube
    dimdiff = int((np.shape(sci)[0] - cube.shape[1]) / 2)
    if dimdiff != 0:
        window = window[dimdiff:-dimdiff, dimdiff:-dimdiff]

    # apply mask to cube
    masked_cube = cube * window

    # get spectral profile
    total_i = np.sum(masked_cube, axis=(1, 2)) * u.Jy

    # change the bandwidth hertz units in the header to km/s
    # ref: https://keflavich-astropy.readthedocs.io/en/latest/units/equivalencies.html#a-slightly-more-complica
    # ted-example-spectral-doppler-equivalencies

    restfreq = (hdul['PRIMARY'].header['RESTFRQ'] * u.Hz).to(u.GHz)  # rest frequency of 12 CO 2-1 in GHz

    # *** THIS DOESN'T WORK IN RELATIVE UNITS - need to get the vel value of each channel and subtract them
    # from each other

    freq_to_vel = u.doppler_radio(restfreq)
    nchans = masked_cube.shape[0]
    chan_idx = np.stack((np.zeros(nchans), np.zeros(nchans), np.arange(nchans), np.zeros(nchans)), axis=1)
    chan_idx = np.array(chan_idx)
    chan_freqs = cubewcs.array_index_to_world_values(chan_idx)
    chan_freqs = np.array(chan_freqs)[:, 2]

    chan_vels = (chan_freqs * u.Hz).to(u.km / u.s, equivalencies=freq_to_vel)
    meanvel = np.median(chan_vels)
    if center_mean == True:
        p0[1] = meanvel.value

    # bandwidth
    deltanu = -np.diff(chan_vels)[0]

    # multiply by clean beam area (in arcseconds squared ***)
    # if the image cube beam is different from the moment one, that's accounted for here automatically because
    # the header is called individually
    abeam = (np.pi * hdul[0].header['BMAJ'] * hdul[0].header['BMIN'] * u.deg ** 2).to(u.arcsec ** 2).value / 4

    # also need to find the area of the window region and divide by that area, because we're integrating over it
    # currently assuming the ellipse has no rotation, so the sigma values are just in x and y in native pixels
    centerpix = cube.shape[2] // 2  # middle pixel coordinate
    skycoordx1, skycoordx2 = cubecelwcs.pixel_to_world([[centerpix, centerpix], [centerpix + nstds*sigx, centerpix]],
                                                       [centerpix, centerpix])
    skycoordy1, skycoordy2 = cubecelwcs.pixel_to_world([[centerpix, centerpix], [centerpix, centerpix + nstds*sigy]],
                                                       [centerpix, centerpix])

    # semi-major and -minor axes in arcseconds
    rega = skycoordx2.separation(skycoordx1).to(u.arcsec)[0]
    regb = skycoordy2.separation(skycoordy1).to(u.arcsec)[1]

    # area of the 4sigma window region in arcseconds squared
    areg = np.pi * rega * regb

    # (Jy/beam)*beam = Jy integrated over the window region divided by the region's area
    total_i = total_i / areg.value  # * abeam

    # fit to a gaussian
    opt1, cov1 = curve_fit(gaussian, chan_vels.value, total_i.value, p0=p0)
    # fit to two gaussians
    if not p02:
        # if no fit parameters are passed, guess based off the fit parameters for one gaussian
        p02 = [p0[0], p0[0], p0[1] - 50, p0[1] + 50, p0[2], p0[2], 0]
    opt2, cov2 = curve_fit(twogaussian, chan_vels.value, total_i.value, p0=p02)

    vel = chan_vels.value
    flux = total_i.value * 1e3

    # find the reduced chi-squared value for each fit
    rms1, chi1 = rms_chi(flux, gaussian(vel, *opt1) * 1e3, 4)
    rms2, chi2 = rms_chi(flux, twogaussian(vel, *opt2) * 1e3, 8)

    # print out the chi-squared values for each fit
    print("chi-squared for one Gaussian: {:.3f}".format(chi1))
    print("chi-squared for two Gaussians: {:.3f}".format(chi2))

    if chi1 < chi2:
        use_twog = False
    else:
        use_twog = True

    print("using a two-gaussian fit: {}".format(use_twog))

    # NOTE: for now that there are so many channels that the three extra free parameters in 
    #the two-gaussian fit dont affect the p-value significantly. otherwise, need to compare actual probabilities
    #instead of just the chi-squared numbers

    # find the center value of the signal in Hz
    #     mean = (opt[1] * u.km / u.s).to(u.Hz, equivalencies=freq_to_vel)

    # plot the intensity data to make sure the fit worked
    x = np.linspace(np.min(chan_vels.value), np.max(chan_vels.value), num=500)
    plt.bar(chan_vels.value, total_i.value, width=deltanu.value)
    plt.ylabel('Integrated Intensity (Jy)')
    plt.xlabel('Velocity (km/s, LSRK)')
    if use_twog == False:
        plt.plot(x, gaussian(x, *opt1), color='orange')
    else:
        plt.plot(x, twogaussian(x, *opt2), color='orange')

    # find area under gaussian fit - integration means -> Jy*km/s
    if use_twog == False:
        specmax = quad(gaussian, np.min(chan_vels.value), np.max(chan_vels.value),
                       args=(opt1[0], opt1[1] - opt1[3], opt1[2], 0))[0]
        opt = opt1
    else:
        specmax = quad(twogaussian, np.min(chan_vels.value), np.max(chan_vels.value),
                       args=(opt2[0], opt2[1], opt2[2], opt2[3], opt2[4], opt2[5], opt2[6]))[0]
        opt = opt2
    if not use_twog:
        rms = rms1
    else:
        rms = rms2

    return mommax * u.Jy * u.km / u.s, specmax * u.Jy * u.km / u.s, opt, rms
    
    '''


def co_line_luminosity(vel_int_flux, nu_obs, z, cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):
    '''equation to find the line luminosity of CO. Taken from Solomon and Vanden Bout (2005). 
       INPUTS: vel_int_flux: velocity integrated flux (Jy.km/s)
               nu_obs: The center observed frequency of the CO line (GHz)
               cosmo: an astropy.cosmology object describing the desired cosmology (needed to calculate the
                      luminosity distance). Default is definition in Webb (2015): flat with H0=70 km/s.Mpc and
                      Omega_dark matter = 0.7
               z: the redshift of the source
        RETURNS the CO line luminosity in solar luminosities
    '''
    DL = cosmo.luminosity_distance(z)  # luminosity distance in Mpc
    print(DL)
    return (3.25e7 * vel_int_flux * np.power(nu_obs, -2) * DL ** 2 * np.power(1 + z, -3)).value


def M_gas(L_line, r_21=0.85, alpha_CO=1):
    '''equation to find total gas mass from a CO line luminosity. taken from Noble et al. (2017). Defaults
       are from the ALMA proposal
       Returns total gas mass in solar masses
    '''
    return alpha_CO * (L_line / r_21)

def mgas_uncert(dlco, r_21=0.85, alpha_CO=1):
    return alpha_CO*dlco/r_21


def get_M_gas_txt(textfile, z, p0, p02=None, cosmo=FlatLambdaCDM(H0=70, Om0=0.3), uncertz=0.0001,
                  r_21=0.85, alpha_CO=1, center_mean=True, linefree=(0, 50), n_gauss=None, zvals=(90, 160),
                  centline=False, plot=True, bounds=(np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
                                                     np.array([np.inf, np.inf, np.inf, np.inf]))):
    specflux, specrms, specopt, speccov, centfreq, uncertA = get_vel_from_txt(textfile, p0=p0, p02=p02,
                                                                              centmean=center_mean, linefree=linefree,
                                                                              n_gauss=n_gauss, zvals=zvals, z=z,
                                                                              centline=centline, plot=plot,
                                                                              bounds=bounds)

    print("RMS value: {}".format(specrms))
    print(centfreq)
    # get the center frequency of the detection from the gaussian fit (change to GHz)
    centfreq = centfreq.to(u.GHz)

    if n_gauss == 1:
        dcentfreq = np.abs(speccov[1, 1])
        print("uncertainty on nuobs: {:.3e}".format(dcentfreq))
    else:
        dmean1, dmean2 = speccov[2, 2], speccov[3, 3]
        dcentfreq = np.sqrt(dmean1 ** 2 + dmean2 ** 2)
        print("uncertainty on nuobs: {:.3e}".format(dcentfreq))

    speccov = np.abs(speccov)
    specopt = np.abs(specopt)

    # line luminosity in solar luminosities
    Lco = co_line_luminosity(specflux, centfreq, z, cosmo)
    print("line luminosity: {:.3e}".format(Lco))
    # uncertainty on line luminosity
    dlco = lco_uncert(specflux, z, (centfreq).to(u.Hz).value, Lco, uncertA, uncertz, dcentfreq, cosmo)
    print("uncertainty on line luminosity: {:.3e}".format(dlco))

    # gas mass in solar masses
    Mgas = M_gas(Lco, r_21, alpha_CO)
    # uncertainty on gas mass
    dmgas = mgas_uncert(dlco, r_21, alpha_CO)
    print("uncertainty on gas mass: {:.3e}".format(dmgas))

    return Mgas, dmgas

'''
def get_M_gas(moment_file, cube_file, z, cutout=(90, 120), p0=[0.00004, 131700, 100, 0],
              cosmo=FlatLambdaCDM(H0=70, Om0=0.3), r_21=0.85, alpha_CO=1, use_momflux=False, center_mean=True):
    '''''' Wrapper function encompassing all of the others - passing an image to this function should return the total gas mass
        in solar masses corresponding to the CO detection in the image.
        INPUTS: moment_file: file containing the moment 0 plot of the line detection
                cube_file: image cube
                z: the redshift of the target galaxy
                cutout: (xmin,xmax): the region (in native image pixels) of the moment0 map the ellipse fitter should look
                                     for signal
                p0: [amplitude, mean, standard deviation, y offset]: the best guess parameters for the gaussian fit to the 
                                                                     spectral profile
                cosmo: FlatLambdaCDM object describing the desired cosmology (used to calculate the luminosity distance)
                use_momflux: if True, the calculation will use the peak intensity value from the moment 0 map, instead of the 
                             value from fitting to the spectral profile of the detection
        RETURNS: Total gas mass of the source in the moment 0 plot in solar masses
    
    ''''''

    # first, find the total velocity integrated flux of the co signal
    momflux, specflux, centfreq = get_velocity_int_flux(moment_file, cube_file, cutout, p0, center_mean=center_mean)

    # values returned are in mJy km/s - change them to Jy km/s
    momflux = momflux.to(u.Jy * u.km / u.s)
    specflux = specflux.to(u.Jy * u.km / u.s)

    # use the flux found by getting the area under a gaussian fit to the spectral profile of the detection
    if use_momflux == True:
        totflux = momflux
    else:
        totflux = specflux

    # get the center frequency of the detection from the gaussian fit (change to GHz)
    centfreq = centfreq.to(u.GHz)

    # line luminosity in solar luminosities
    Lco = co_line_luminosity(totflux, centfreq, z, cosmo)

    # gas mass in solar masses
    Mgas = M_gas(Lco, r_21, alpha_CO)

    return Mgas
'''


def plot_spectral_profile(file, nFWHMa, sigma=None, a=None, b=None, theta=None):
    ''' Function to plot a 1D spectral profile of a given region about the center of a pointing. a, b
        are the beam semimajor and semiminor axes, respectively, and are in arcminutes. 
    '''

    # get the data from the file
    hdu = fits.open(file)[0]
    data = hdu.data
    # remove the stokes axis
    data = data[0, :, :, :]
    # map NaNs to zero
    data[np.where(np.isnan(data))] = 0.

    if a is None:
        # get the semimajor and semiminor axis values from the header
        a = (hdu.header['BMAJ'] * u.deg)
    else:
        a = (a * u.deg)

    if b is None:
        b = (hdu.header['BMIN'] * u.deg)
    else:
        b = (b * u.deg)

    if theta is None:
        theta = (hdu.header['BPA'] * u.deg)
    else:
        theta = theta * u.deg

    # get wcs
    datwcs = wcs.WCS(hdu.header).sub(['celestial'])
    alldatwcs = wcs.WCS(hdu.header)

    # define skycoord objects for the extent of the axes to change into pix
    center = SkyCoord(hdu.header['CRVAL1'] * u.deg, hdu.header['CRVAL2'] * u.deg, frame='fk5')
    ext = SkyCoord(hdu.header['CRVAL1'] * u.deg - nFWHMa * a, hdu.header['CRVAL2'] * u.deg + nFWHMa * b, frame='fk5')

    # change the wcs into native pixels
    centerpix = datwcs.world_to_pixel(center)
    extpix = datwcs.world_to_pixel(ext)

    # center values are the center of the ellipse, the lengths are the differences between center and ext
    arad, brad = int(extpix[0] - centerpix[0]), int(extpix[1] - centerpix[1])

    # define the Ellipse2D object IGNORING ANY ANGLE IN THE BEAM FOR NOW***
    window = Ellipse2D(1, int(centerpix[0]), int(centerpix[1]), arad, brad, theta)

    # to get an actual array, need to pass it a meshgrid of x and y coordinates
    datx = np.arange(np.shape(data)[1])
    x, y = np.meshgrid(datx, datx)

    # apply the mask
    data = data * window(x, y)

    # sum over the spatial axes
    total_i = np.sum(data, axis=(1, 2)) * u.Jy

    # divide by the area of the window in arcsecs to normalize
    abeam = (np.pi * a * b).to(u.arcsec ** 2)
    total_i = total_i / abeam

    # change the bandwidth hertz units in the header to km/s
    # ref: https://keflavich-astropy.readthedocs.io/en/latest/units/equivalencies.html#a-slightly-more-complica
    # ted-example-spectral-doppler-equivalencies

    restfreq = (hdu.header['RESTFRQ'] * u.Hz).to(u.GHz)  # rest frequency of 12 CO 2-1 in GHz

    # *** THIS DOESN'T WORK IN RELATIVE UNITS - need to get the vel value of each channel and subtract them 
    # from each other

    freq_to_vel = u.doppler_radio(restfreq)
    nchans = data.shape[0]
    chan_idx = np.stack((np.zeros(nchans), np.zeros(nchans), np.arange(nchans), np.zeros(nchans)), axis=1)
    chan_idx = np.array(chan_idx)
    chan_freqs = alldatwcs.array_index_to_world_values(chan_idx)
    chan_freqs = np.array(chan_freqs)[:, 2]

    chan_vels = (chan_freqs * u.Hz).to(u.km / u.s, equivalencies=freq_to_vel)

    # add smoothing as an option
    if sigma is not None:
        total_i_sm = gaussian_filter1d(total_i, sigma=sigma, truncate=5)

    # plot
    plt.plot(chan_vels, total_i, zorder=10, lw=1, label='Raw')
    if sigma is not None:
        plt.plot(chan_vels, total_i_sm, zorder=11, lw=3, label='Smoothed (std = {})'.format(sigma))
        plt.legend()
    plt.ylabel('Integrated Intensity (Jy)')
    plt.xlabel('Velocity (km/s, LSRK)')
    plt.axhline(0, color='k')

    return total_i, chan_vels
