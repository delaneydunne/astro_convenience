import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from photutils import data_properties, EllipticalAperture, deblend_sources
from photutils.segmentation import detect_sources, source_properties
from photutils.aperture import SkyCircularAperture as sca
from photutils.aperture import aperture_photometry
from astropy.table import Table, join
from astropy.io import fits
from astropy import wcs
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from matplotlib.patches import Rectangle, Ellipse




# function to get an array kron fluxes by passing known wcs centroids and the image/error image
def get_kron_fluxes(wcscents, dat, errfile=False, nstds=1.5, npix=10, connectivity=4, apsize=20, plot=False,
                    columns=['id', 'xcentroid', 'ycentroid', 'kron_flux', 'kron_fluxerr', 'kron_radius'],
                    kron_params=('mask', 2.5, 0.0, 'exact', 5)):

    with fits.open(dat) as datfile:
        datahdr = datfile[0].header
        data = datfile[0].data

    if errfile:
        with fits.open(err) as errfile:
            errhdr = errfile[0].header
            error = errfile[0].data

    datwcs = wcs.WCS(datahdr)

    data[np.where(datwcs == np.inf)] = 0.
    data[np.where(datwcs == -np.inf)] = 0.
    data[np.where(np.isnan(data))] = 0.

    apers = sca(wcscents, r=apsize*u.arcsec)
    pixapers = apers.to_pixel(datwcs)
    masks = pixapers.to_mask(method='center')

    # list of data images, error images, and threshold images
    datcutlist = []
    errcutlist = []
    threshlist = []
    sourcelist = []
    catlist = []
    goods = []
    goodcatlist = []
    ulttable = []

    for mask in masks:
        datcutlist.append(mask.multiply(data))
        if errfile:
            errcutlist.append(mask.multiply(error))

        # make a threshold image
        tval = np.nanmean(data) + nstds * np.nanstd(data)
        threshold = np.ones(mask.shape) * tval
        threshlist.append(threshold)

    for i in range(len(datcutlist)):
        # get a segmentation image containing the sources
        sources = detect_sources(datcutlist[i], threshlist[i], npix, connectivity=connectivity)
        sources = deblend_sources(datcutlist[i], sources, npixels=npix, )
        sourcelist.append(sources)
        if errfile:
            cat = source_properties(datcutlist[i], sources, kron_params=kron_params, error=errcutlist[i])
        else:
            cat = source_properties(datcutlist[i], sources, kron_params=kron_params)
        catlist.append(cat)

        dist = np.mean((np.abs(cat.xcentroid.value - pixapers[i].r), np.abs(cat.ycentroid.value - pixapers[i].r)),
                       axis=0)
        good = np.where(dist == np.min(dist))[0][0]
        goods.append(good)
        goodcatlist.append(cat[good])

        if i == 0:
            tbl = cat[good].to_table(columns=columns)
        else:
            tbl = join(tbl, cat[good].to_table(columns=columns), join_type='outer')


    if plot:

        r = 3.  # approximate isophotal extent
        apertures = []
        kronapertures = []
        cents = []
        for obj in goodcatlist:
            position = np.transpose((obj.xcentroid.value, obj.ycentroid.value))
            a = obj.semimajor_axis_sigma.value * r
            b = obj.semiminor_axis_sigma.value * r
            theta = obj.orientation.to(u.rad).value
            kronrad = obj.kron_radius.value
            apertures.append(EllipticalAperture(position, a, b, theta=theta))
            kronapertures.append(obj.kron_aperture)
            cents.append(position)

        fig, ax = plt.subplots(len(datcutlist), 1, figsize=(5, 5*len(datcutlist)))
        for i in range(len(datcutlist)):
            ax[i].pcolormesh(datcutlist[i], cmap='plasma')
            apertures[i].plot(color='white', axes=ax[i])
            kronapertures[i].plot(color='green', axes=ax[i])
            ax[i].set_aspect(aspect=1)
            ax[i].set_title(str(i))

    tbl['kron_flux'] = tbl['kron_flux']*datahdr['FLUXZERO'] / 1e3
    tbl['kron_fluxerr'] = tbl['kron_fluxerr']*datahdr['FLUXZERO'] / 1e3

    return tbl


def get_kron_flux(coord, im, errfile=None, rad=15, nstds=1, npix=10, connectivity=8,
                  kron_params=('mask', 2, 8, 'exact', 1), plot=False, forcegood=None):

    ''' Function to get a single kron flux given a single image (instead of an array of them)
    '''

    with fits.open(im) as file:
        mipshdr = file[0].header
        mipsdat = file[0].data

    if errfile:
        with fits.open(errfile) as file:
            errhdr = file[0].header
            errdat = file[0].data

    mipswcs = wcs.WCS(mipshdr)

    # first clip out the region surrounding the source you want (gotta be pretty tight, because you're going to fit it
    # right away)
    pixcent = wcs.utils.skycoord_to_pixel(coord, mipswcs) # input source center in pixel coordinates

    # take a circular aperture around that centroid and extract the pixels inside that aperture as a data array
    roughaper = sca(coord, r=rad*u.arcsec)
    pixroughaper = roughaper.to_pixel(mipswcs)
    roughmask = pixroughaper.to_mask(method='center')

    roughdatcut = roughmask.multiply(mipsdat)

    if errfile:
        rougherrcut = roughmask.multiply(errdat)

    # make a threshold image
    tval = np.nanmean(mipsdat) + nstds * np.nanstd(mipsdat)
    threshold = np.ones(np.shape(roughdatcut)) * tval


    # get a segmentation image containing the sources
    sources = detect_sources(roughdatcut, threshold, npix, connectivity=connectivity)
    sources = deblend_sources(roughdatcut, sources, npixels=npix)
    cat = source_properties(roughdatcut, sources, kron_params=kron_params)

    if len(cat) > 1:
        dist = np.mean((np.abs(cat.xcentroid.value - rad), np.abs(cat.ycentroid.value - rad)),
                       axis=0)
        good = np.where(dist == np.min(dist))[0][0]
    else:
        good = 0

    if forcegood:
        good = forcegood


    if plot:
        plt.pcolormesh(roughdatcut, cmap='plasma')
        cat[good].kron_aperture.plot(color='white')

    columns = ['id', 'xcentroid', 'ycentroid', 'kron_flux', 'kron_fluxerr', 'kron_radius']
    tbl = cat.to_table(columns=columns)

    flux = (tbl['kron_flux'][good]*u.MJy/u.sr*(np.abs(mipshdr['CDELT1']*mipshdr['CDELT2'])*u.deg**2)
            .to(u.sr)).to(u.Jy)
    if errfile:
        err = (tbl['kron_fluxerr'][good]*u.MJy/u.sr*(np.abs(mipshdr['CDELT1']*mipshdr['CDELT2'])*
                                               u.deg**2).to(u.sr)).to(u.Jy)
        print(' {:.4f}, {:.3e}'.format(flux.value, err.value))
    else:
        print(' {:.4f},,'.format(flux.value))

    return tbl




def get_ap_flux(coord, im, rad=(3.8, 5.8), plot=False):

    with fits.open(im) as file:
        irachdr = file[0].header
        iracdat = file[0].data

    iracrms = np.sqrt(np.mean(iracdat**2))*np.ones(np.shape(iracdat))

    iracwcs = wcs.WCS(irachdr)

    # first clip out the region surrounding the source you want (gotta be pretty tight, because you're going to fit it
    # right away)
    pixcent = wcs.utils.skycoord_to_pixel(coord, iracwcs) # input source center in pixel coordinates

    # take a circular aperture around that centroid and extract the pixels inside that aperture as a data array
    aplist = []
    for radval in rad:
        roughaper = sca(coord, r=radval*u.arcsec)
        pixroughaper = roughaper.to_pixel(iracwcs)
        aplist.append(pixroughaper)

    photlist = aperture_photometry(iracdat, aplist)
    errlist = aperture_photometry(iracrms, aplist)

    if plot:
        plt.pcolormesh(iracdat, cmap='plasma')
        for ap in aplist:
            ap.plot(color='white')


    flux38 = (photlist['aperture_sum_0'][0]*u.MJy/u.sr*(np.abs(irachdr['CDELT1']*irachdr['CDELT2'])*u.deg**2)
            .to(u.sr)).to(u.Jy)
    err38 = (errlist['aperture_sum_0'][0]*u.MJy/u.sr*(np.abs(irachdr['CDELT1']*irachdr['CDELT2'])*u.deg**2)
            .to(u.sr)).to(u.Jy)
    flux58 = (photlist['aperture_sum_1'][0]*u.MJy/u.sr*(np.abs(irachdr['CDELT1']*irachdr['CDELT2'])*u.deg**2)
             .to(u.sr)).to(u.Jy)
    err58 = (errlist['aperture_sum_1'][0]*u.MJy/u.sr*(np.abs(irachdr['CDELT1']*irachdr['CDELT2'])*u.deg**2)
             .to(u.sr)).to(u.Jy)

    print(' {:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(flux38.value, err38.value, flux58.value, err58.value))


    return photlist

    
