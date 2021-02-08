import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Ellipse
from photutils.centroids import fit_2dgaussian
from astropy.modeling.functional_models import Ellipse2D
from scipy.optimize import curve_fit
import co_masses as co



def gaussian(x, amp, mu, sig, C):
    '''simple 1d gaussian function with included offset'''
    return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))  # + C


def twogaussian(x, amp1, amp2, mu1, mu2, sig1, sig2, C):
    gauss1 = amp1 * np.exp(-np.power(x - mu1, 2.) / (2 * np.power(sig1, 2.)))  # + C
    gauss2 = amp2 * np.exp(-np.power(x - mu2, 2.) / (2 * np.power(sig2, 2.)))  # + C
    return gauss1 + gauss2


def find_SN_sigma(file, cutout, mips=False, plot=False, rad=None, ycutout=None):
    # define an elliptical mask to block out the source
    data = fits.open(file)[0].data
    if not mips:
        data = data[0,0,:,:]

    data[np.where(np.isnan(data))] = 0.
    xmin,xmax = cutout[0],cutout[1]
    if ycutout:
        ymin,ymax = ycutout[0],ycutout[1]
    cent = len(data)//2
    if ycutout:
        cent = ((xmax-xmin)/2, (ymax-ymin)/2)
    if not rad:
        rad = len(data)//4
    datacut = data[xmin:xmax, xmin:xmax]
    if ycutout:
        datacut = data[xmin:xmax, ymin:ymax]
    datafit = fit_2dgaussian(datacut)

    maxval = np.nanmax(datacut)

    # ellipse2d object blocking out the signal
    ell = Ellipse2D(1, datafit.x_mean.value + xmin, datafit.y_mean.value + xmin, datafit.x_stddev.value*2.3,
                    datafit.y_stddev.value*2.3, datafit.theta.value)
    if ycutout:
        ell = Ellipse2D(1, datafit.x_mean.value+xmin, datafit.y_mean.value+ymin, datafit.x_stddev.value*2.3,
                        datafit.y_stddev.value*2.3, datafit.theta.value)

    datx = np.arange(np.shape(data)[0])
    x, y = np.meshgrid(datx, datx)

    # larger circle around the center - go with 1/6 of the imsize as the radius
    if ycutout:
        circ = Ellipse2D(1, cent[0], cent[1], rad, rad, 0)
    else:
        circ = Ellipse2D(1, cent, cent, rad, rad, 0)

    # mask is the larger circle minus the covering ellipse
    mask = circ(x,y) - ell(x,y)
    masked_data = data*mask

    vals = masked_data[np.where(masked_data != 0)].flatten()

    if mips:
        vals=data[0:10, 0:10].flatten()

    if plot:
        plt.pcolormesh(masked_data)

    sigma = np.sqrt(np.sum(vals**2) / len(vals))

    SN = maxval / sigma
    print(sigma, SN)

    fitxcent = datafit.x_mean.value+xmin
    if ycutout:
        fitycent = datafit.y_mean.value+ymin
    else:
        fitycent = datafit.y_mean.value+xmin

    xpeak = np.where(datacut == maxval)[1] + xmin
    if ycutout:
        ypeak = np.where(datacut == maxval)[0] + ymin
    else:
        ypeak = np.where(datacut == maxval)[0] + xmin

    return sigma, SN, fitxcent, fitycent, xpeak, ypeak


from matplotlib.patches import Rectangle, Ellipse


def plot_contour_overlay(files, optmin, optmax, cutout, pixsize, step=1, inputfreq=None, freqlabel=False,
                         whole_image=False, use_Hz=False, max_stds=10, max_scale=None, beam_center=None, mips=False,
                         irac=False, mipscutout=None, startsig=2, deltap=None):
    ''' function to plot the contours of a radio spectral line integrated flux over an optical image.
        files should be structured ('optical file path', 'moment 0 file path') to fits files. Optmin and optmax are
        the vmin, vmax of the colourmap for the optical image, and lims are the (min,max) limits of the
        coordinates that should be plotted, in native pixels to the radio contours.
    '''

    # open the images
    opthdu = fits.open(files[0])[0]
    momhdu = fits.open(files[1])[0]

    if mips == True:
        mipshdu = fits.open(files[2])[0]
        mipswcs = wcs.WCS(mipshdu.header)

    # WCS info for both images
    optwcs = wcs.WCS(opthdu.header)
    momwcs = wcs.WCS(momhdu.header)

    momwcs = momwcs.sub(['celestial'])

    # define contour levels starting at 2*RMS and then increasing in intervals of step*RMS
    # pass infinities to zero to avoid them messing up statistics
    momdata = momhdu.data[0, 0, :, :]
    momdata[(np.where(momdata == -np.inf))] = 0.
    mommax = np.nanmax(momdata)
    momstd, momSN, xcent, ycent, xpeak, ypeak = find_SN_sigma(files[1], cutout)
    levels = np.ones((max_stds - startsig) // step) * momstd
    stds = np.arange(startsig, max_stds, step=step) * momstd
    #     levels = np.flip(levels - stds)
    #     print(stds, levels)
    print(stds)
    levels = stds

    if mips == True:
        # define MIPS contour levels
        mipsdata = mipshdu.data
        if mipscutout:
            mipsdata = mipsdata[mipscutout[0]:mipscutout[1], mipscutout[2]:mipscutout[3]]
        # pass infinities and nans to zero
        mipsdata[(np.where(mipsdata == -np.inf))] = 0.
        mipsdata[(np.where(np.isnan(mipsdata)))] = 0.
        mipsmax = np.nanmax(mipsdata)
        mipsstd, mipsSN, xmcent, ymcent = find_SN_sigma(files[2], (0, len(mipsdata)), mips=True)
        mipslevels = np.ones((max_stds - 2) // step) * mipsmax
        mipsstds = np.arange(max_stds, 2, step=-step) * mipsstd
        mipslevels = mipslevels - mipsstds

    # plot the contours first
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=momwcs)
    ax.contour(momdata, colors='red', levels=levels, zorder=5)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # WCS transform to line the optical image up
    opt_transform = ax.get_transform(optwcs)

    if mips == True:
        # WCS transform to line the mips image up
        mips_transform = ax.get_transform(mipswcs)

        # plot the mips flux contour levels
        ax.contour(mipsdata, transform=mips_transform, colors='pink', levels=mipslevels, zorder=6)

    # plot the optical
    im = ax.imshow(opthdu.data, cmap=plt.cm.gray, transform=opt_transform, vmin=optmin, vmax=optmax, zorder=1)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.07)

    if inputfreq is not None:
        # add the contour spacing to the string
        inputfreq = inputfreq + ', spacing=' + str(step) + r'$\sigma$'
        # add a legend showing the central wavelength of the moment contours (using the inputted frequency)
        custom_lines = [Line2D([0], [0], color='red', lw=2)]
        ax.legend(custom_lines, ['{}'.format(inputfreq)])

    if freqlabel == True:
        # add a legend showing the central wavelength of the moment contours
        custom_lines = [Line2D([0], [0], color='red', lw=2)]

        # get velocity of the moment frequency
        nuobs = momhdu.header['cfreq']
        if use_Hz == False:
            # coordinate transformation from hz to km/s
            freq_to_vel = u.doppler_radio(momhdu.header['RESTFRQ'] * u.Hz)
            vel = (nuobs * u.Hz).to(u.km / u.s, equivalencies=freq_to_vel)

            ax.legend(custom_lines, ['{:.1f}'.format(vel)])

        else:
            vel = (nuobs * u.Hz).to(u.GHz)
            ax.legend(custom_lines, ['{:.3f}'.format(vel)])

    # get the semimajor and semiminor axis values from the header
    a = (momhdu.header['BMAJ'] * u.deg)
    b = (momhdu.header['BMIN'] * u.deg)
    theta = (momhdu.header['BPA'] * u.deg).to(u.rad).value + np.pi / 2

    # define skycoord objects for the extent of the axes to change into pix
    center = SkyCoord(momhdu.header['CRVAL1'] * u.deg, momhdu.header['CRVAL2'] * u.deg, frame='fk5')
    ext = SkyCoord(momhdu.header['CRVAL1'] * u.deg - a, momhdu.header['CRVAL2'] * u.deg + b, frame='fk5')

    # change the wcs into native pixels
    centerpix = momwcs.world_to_pixel(center)
    extpix = momwcs.world_to_pixel(ext)

    # center values are the center of the ellipse, the lengths are the differences between center and ext
    arad, brad = int(extpix[0] - centerpix[0]), int(extpix[1] - centerpix[1])

    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')

    if whole_image == True:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        beambkg = Rectangle((xlim[0], ylim[0]), width=arad * 1.5, height=brad * 1.5, facecolor='w',
                            edgecolor='k', zorder=10, alpha=0.9)
        beam = Ellipse((xlim[0] + arad * 0.75, ylim[0] + brad * 0.75),
                       width=arad, height=brad, angle=theta + np.pi / 2, facecolor='w', edgecolor='k', zorder=11)

    else:
        if irac == True:
            xcent = np.shape(momdata)[0] // 2
            ycent = np.shape(momdata)[0] // 2
        xlims = ax.set_xlim([int(xcent) - pixsize, int(xcent) + pixsize])
        ylims = ax.set_ylim([int(ycent) - pixsize, int(ycent) + pixsize])

        beambkg = Rectangle((xlims[0], ylims[0]), width=arad * 1.5, height=brad * 1.5, facecolor='w',
                            edgecolor='k', zorder=10, alpha=0.9)
        beam = Ellipse((xlims[0] + arad * 0.75, ylims[0] + brad * 0.75),
                       width=arad, height=brad, angle=theta + np.pi / 2, facecolor='w', edgecolor='k', zorder=11)

    if beam_center is not None:
        beam_center = SkyCoord(beam_center[0] * u.deg, beam_center[1] * u.deg, frame='fk5')
        centval = momwcs.world_to_pixel(beam_center)
        ax.scatter(centval[0], centval[1], marker='+', s=300, zorder=20)

    if max_scale is not None:
        arcsecperpix = np.abs((momhdu.header['CDELT1'] * u.deg).to(u.arcsec)).value
        max_scale = max_scale / arcsecperpix
        print(max_scale)
        mcirc = Ellipse([xpeak, ypeak], width=max_scale, height=max_scale,
                        facecolor='green', edgecolor='k', zorder=3, alpha=0.3)
        ax.add_patch(mcirc)

    if deltap is not None:
        arcsecperpix = np.abs((momhdu.header['CDELT1'] * u.deg).to(u.arcsec)).value
        deltap = deltap / arcsecperpix
        ax.scatter(xpeak, ypeak, color='orange', zorder=20)
        horline = Line2D([xpeak - deltap, xpeak + deltap], [ypeak, ypeak], color='orange', zorder=10)
        vertline = Line2D([xpeak, xpeak], [ypeak - deltap, ypeak + deltap], color='orange', zorder=10)
        ax.add_line(horline)
        ax.add_line(vertline)

    ax.add_patch(beambkg)
    ax.add_patch(beam)

    return



def spw_contours(files, mom_cutout, npix, redshift, optext=(1,200), startstds=2, step=1, max_stds=50,
                 whole_image=False, bcgcent=None):
    # files should be organized like (optfile, spw1, spw2, spw3, spw4)
    # open up the files and get their data and WCS
    # all the different SPWs should have the same wcs
    WCSlist = []
    datalist = []
    hdulist = []
    for file in files:
        hdu = fits.open(file)[0]
        hdulist.append(hdu)
        WCSlist.append(wcs.WCS(hdu.header).sub(['celestial']))
        data = hdu.data
        data[np.where(np.isnan(data))] = 0.
        data[np.where(data == -np.inf)] = 0.
        data[np.where(data == np.inf)] = 0.
        if len(data) == 4:
            data = data[0, 0, :, :]
        datalist.append(data)



    # define a figure --- only need 1 set of axes here (maybe do one with individual spws?)
    fig = plt.figure(figsize=(7,7))

    # get SN statistics for each spw
    momstdlist = []
    momSNlist = []
    levelslist = []
    for file in files[1:]:
        print(file)
        mommax = np.nanmax(datalist[0])
        momstd, momSN, xcent, ycent, xpeak, ypeak = find_SN_sigma(file, mom_cutout)

        if file == files[1]:
            xcentkeep = xcent
            ycentkeep = ycent

        momstdlist.append(momstd)
        momSNlist.append(momSN)

        stds = np.arange(startstds, max_stds, step=step) * momstd
        levelslist.append(stds)

    colors=['red', 'yellow', 'green', 'blue']

    # plot the contours first
    ax = fig.add_subplot(111, projection=WCSlist[1])
    ax.contour(datalist[1][0][0], colors=colors[0], levels=levelslist[0], linewidths=1, zorder=5)
    for i in np.arange(2,5):
        j = i - 1
        transform = ax.get_transform(WCSlist[i])
        ax.contour(datalist[i][0][0], colors=colors[j], levels=levelslist[j], linewidths=1, zorder=5+j,
                   transform=transform)

    # WCS transform to line the optical image up
    opt_transform = ax.get_transform(WCSlist[0])

    # plot the optical
    im = ax.imshow(datalist[0], cmap=plt.cm.gray, transform=opt_transform, vmin=optext[0], vmax=optext[1], zorder=1)

    # axis labels
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # plot the WCS center of the BCG if given
    if bcgcent:
        bcgpixcent = bcgcent.to_pixel(WCSlist[1])
        ax.scatter(*bcgpixcent, color='magenta', zorder=10, s=100, marker='+', label='BCG WCS Center')

    # clip the axes to xlim, ylim
    if whole_image == True:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    else:
        xlims = ax.set_xlim([int(xcentkeep) - npix, int(xcentkeep) + npix])
        ylims = ax.set_ylim([int(ycentkeep) - npix, int(ycentkeep) + npix])

        if bcgcent:
            xlims = ax.set_xlim([int(bcgpixcent[0]) - npix, int(bcgpixcent[0]) + npix])
            ylims = ax.set_ylim([int(bcgpixcent[1]) - npix, int(bcgpixcent[1]) + npix])

    ''' Show the beam --- assume it will be different in each spw'''
    nextstart = 0
    for i in np.arange(1, 5):
        # get the semimajor and semiminor axis values from the header
        a = (hdulist[i].header['BMAJ'] * u.deg)
        b = (hdulist[i].header['BMIN'] * u.deg)
        theta = (hdulist[i].header['BPA'] * u.deg).value + 90

        # define skycoord objects for the extent of the axes to change into pix
        center = SkyCoord(hdulist[i].header['CRVAL1'] * u.deg, hdulist[i].header['CRVAL2'] * u.deg, frame='fk5')
        ext = SkyCoord(hdulist[i].header['CRVAL1'] * u.deg - a, hdulist[i].header['CRVAL2'] * u.deg + b, frame='fk5')

        # change the wcs into native pixels
        centerpix = WCSlist[i].world_to_pixel(center)
        extpix = WCSlist[i].world_to_pixel(ext)

        # center values are the center of the ellipse, the lengths are the differences between center and ext
        arad, brad = int(extpix[0] - centerpix[0]), int(extpix[1] - centerpix[1])
        if i == 1:
            maxarad = arad
            maxbrad = brad

        # store a list of beam objects --- one per spw
        if whole_image == True:
            beambkg = Rectangle((xlim[0], ylim[0]), width=maxarad * 6, height=maxbrad * 1.5, facecolor='w',
                                zorder=10, alpha=0.9)
            beam = Ellipse((xlim[0] + arad * 0.75 + nextstart, ylim[0] + brad * 0.75), alpha=0.7,
                           width=arad, height=brad, angle=theta, facecolor=colors[i-1], edgecolor='k', zorder=11)
            ax.add_patch(beambkg)
            ax.add_patch(beam)

            nextstart += arad*1.5



        else:
            beambkg = Rectangle((xlims[0], ylims[0]), width=maxarad * 6, height=maxbrad * 1.5, facecolor='w',
                                zorder=10, alpha=0.9)
            beam = Ellipse((xlims[0] + arad * 0.75 + nextstart, ylims[0] + brad * 0.75), alpha=0.7,
                           width=arad, height=brad, angle=theta, facecolor=colors[i-1], edgecolor='k', zorder=11)
            ax.add_patch(beambkg)
            ax.add_patch(beam)

            nextstart += arad*1.5


        '''get a legend on'''
        ax.legend()
        custom_lines = [Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='yellow', lw=2),
                       Line2D([0], [0], color='green', lw=2), Line2D([0], [0], color='blue', lw=2)]
        obsfreqs = np.array([224,226,240,242])
        restfreqs = obsfreqs / (1+redshift)
        freqlabels = str(restfreqs)+'GHz'
        ax.legend(custom_lines, ['{:.3f}GHz'.format(restfreqs[0]), '{:.3f}GHz'.format(restfreqs[1]),
                                 '{:.3f}GHz'.format(restfreqs[2]), '{:.3f}GHz'.format(restfreqs[3])]).set_zorder(20)






def make_stamp_array(files, mom_cutout, npix, z, optext=(0, 80), mipsext=(0, 0.5), iracext=(0, 1), delp=None,
                     maxscale=None, max_stds=50, step=2, whole_image=False, startstds=2, p0=[0.3, 67200, 50, 0],
                     centmean=True, p02=None, linefree=(0, 50), n_gauss=None, zvals=(90, 160), SN=None,
                     centline=True, spaxlims=None, max_scale=None, deltap=None, fit=True, crtf=None, ycutout=None,
                     xpeakpass=None, ypeakpass=None, xcentpass=None, ycentpass=None, std=None,
                     bounds=(np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
                             np.array([np.inf, np.inf, np.inf, np.inf]))):
    ''' make an array of postage stamp images for easy analysis of a single souce.
        INPUTS:
            - files: (momfile, optfile, mipsfile, iracfile, specfile) list of file pointers
            - mom_cutout: (xmin, xmax) coordinates of a region around the source in the moment 0 file small
                enough that the source can be fit to a 2D gaussian to do statistics with
            - npix: number of moment 0 native pixels on each side of the center of the source to include in stamps
            - z: redshift of the source (to put a reference line down in the spectral profile)
            - optmin, optmax: vmin, vmax for the DES optical colorscale
            - delp: positional uncertainty for ALMA moment 0 (if not none, will plot a cross centered around the
                middle of the source to show by how much the position is uncertain)
            - maxscale: maximum recoverable angular scale for ALMA observations (if not none, will plot a circle
                showing how much of the structure falls into this scale)
            - max_stds: maximum number of S/N sigma to reach with the contours
            - step: spacing between moment 0 contours, in S/N sigma
            - whole_image: plot the entire extent of the moment 0 image, instead of just a cutout
            - startstds: S/N level above RMS at which contours should start
    '''

    # open all of the files and get their data and wcs
    # files are organized like files[0]: momfile, files[1]: optfile, files[2]: mipsfile, files[3]: iracfile
    # files[4]: specfile
    WCSlist = []
    datalist = []
    hdulist = []
    for file in files[0:4]:
        hdu = fits.open(file)[0]
        hdulist.append(hdu)
        WCSlist.append(wcs.WCS(hdu.header).sub(['celestial']))
        data = hdu.data
        data[np.where(np.isnan(data))] = 0.
        data[np.where(data == -np.inf)] = 0.
        data[np.where(data == np.inf)] = 0.
        if len(data) == 4:
            data = data[0, 0, :, :]
        datalist.append(data)

    # define the figure with 4 sets of axes
    fig = plt.figure(figsize=(10, 7))

    ''' first set of axes: this is the moment 0 overlaid on DES '''
    # get moment 0 statistics to plot S/N contours
    mommax = np.nanmax(datalist[0])
    if not SN:
        momstd, momSN, xcent, ycent, xpeak, ypeak = find_SN_sigma(files[0], mom_cutout)
    else:
        momstd = std
        momSN = SN
        xcent = xcentpass
        ycent = ycentpass
        xpeak = xpeakpass
        ypeak = ypeakpass

    levels = np.ones((max_stds - startstds) // step) * momstd
    stds = np.arange(startstds, max_stds, step=step) * momstd
    levels = stds

    # plot the contours first
    alax = fig.add_subplot(231, projection=WCSlist[0])
    alax.contour(datalist[0][0][0], colors='red', levels=levels, zorder=5)

    #     custom_lines = [Line2D([0],[0], color='red', lw=2)]
    #     alax.legend(custom_lines, ['{} sigma steps'.format(step)])

    xlim = alax.get_xlim()
    ylim = alax.get_ylim()

    # WCS transform to line the optical image up
    opt_transform = alax.get_transform(WCSlist[1])

    # plot the optical
    im = alax.imshow(datalist[1], cmap=plt.cm.gray, transform=opt_transform, vmin=optext[0], vmax=optext[1], zorder=1)

    ''' Show the beam'''
    # get the semimajor and semiminor axis values from the header
    a = (hdulist[0].header['BMAJ'] * u.deg)
    b = (hdulist[0].header['BMIN'] * u.deg)
    theta = (hdulist[0].header['BPA'] * u.deg).value + 90

    # define skycoord objects for the extent of the axes to change into pix
    center = SkyCoord(hdulist[0].header['CRVAL1'] * u.deg, hdulist[0].header['CRVAL2'] * u.deg, frame='fk5')
    ext = SkyCoord(hdulist[0].header['CRVAL1'] * u.deg - a, hdulist[0].header['CRVAL2'] * u.deg + b, frame='fk5')

    # change the wcs into native pixels
    centerpix = WCSlist[0].world_to_pixel(center)
    extpix = WCSlist[0].world_to_pixel(ext)

    # center values are the center of the ellipse, the lengths are the differences between center and ext
    arad, brad = int(extpix[0] - centerpix[0]), int(extpix[1] - centerpix[1])

    alax.set_xlabel('RA')
    alax.set_ylabel('DEC')

    if whole_image == True:
        alax.set_xlim(xlim)
        alax.set_ylim(ylim)

        beambkg = Rectangle((xlim[0], ylim[0]), width=arad * 1.5, height=brad * 1.5, facecolor='w',
                            edgecolor='k', zorder=10, alpha=0.9)
        beam = Ellipse((xlim[0] + arad * 0.75, ylim[0] + brad * 0.75), alpha=0.9,
                       width=arad, height=brad, angle=theta, facecolor='yellow', edgecolor='k', zorder=11)

    else:
        xlims = alax.set_xlim([int(xcent) - npix, int(xcent) + npix])
        ylims = alax.set_ylim([int(ycent) - npix, int(ycent) + npix])

        beambkg = Rectangle((xlims[0], ylims[0]), width=arad * 1.5, height=brad * 1.5, facecolor='w',
                            edgecolor='k', zorder=10, alpha=0.9)
        beam = Ellipse((xlims[0] + arad * 0.75, ylims[0] + brad * 0.75), alpha=0.9,
                       width=arad, height=brad, angle=theta, facecolor='yellow', edgecolor='k', zorder=11)

    #     alax.add_patch(beambkg)
    alax.add_patch(beam)
    alax.set_title('DES z-band')
    alax.tick_params(
        axis='y',
        which='both',
        left='on',
        right='off',
        direction='out',
        labelleft=True,
        length=5,
        labelsize='small'
    )
    alax.tick_params(
        axis='x',
        labelsize='small'
    )

    if max_scale is not None:
        arcsecperpix = np.abs((hdulist[0].header['CDELT1'] * u.deg).to(u.arcsec)).value
        max_scale = max_scale / arcsecperpix
        mcirc = Ellipse((xcent, ycent), width=max_scale, height=max_scale,
                        facecolor='green', edgecolor='green', zorder=10, alpha=0.3)
        print("maxscale")
        print(mcirc)
        alax.add_patch(mcirc)

    if deltap is not None:
        arcsecperpix = np.abs((hdulist[0].header['CDELT1'] * u.deg).to(u.arcsec)).value
        deltap = deltap / arcsecperpix
        horline = Line2D([xpeak - deltap, xpeak + deltap], [ypeak, ypeak], color='orange', zorder=10)
        vertline = Line2D([xpeak, xpeak], [ypeak - deltap, ypeak + deltap], color='orange', zorder=10)
        alax.add_line(horline)
        alax.add_line(vertline)

    if crtf:
        ''' show the region over which the spectral profile was taken'''
        with open(crtf, 'r') as file:
            txt = file.read().split()

            xcent = '0'
            for s in txt[1]:
                if (s.isdigit() or (s == '.')):
                    xcent += s
            xcent = float(xcent[1:])

            ycent = '0'
            for s in txt[2]:
                if (s.isdigit() or (s == '.')):
                    ycent += s
            ycent = float(ycent[1:])

            lena = '0'
            for s in txt[3]:
                if (s.isdigit() or (s == '.')):
                    lena += s
            lena = float(lena[1:]) * 2

            lenb = '0'
            for s in txt[4]:
                if (s.isdigit() or (s == '.')):
                    lenb += s
            lenb = float(lenb[1:]) * 2

            regtheta = '0'
            for s in txt[5]:
                if (s.isdigit() or (s == '.')):
                    regtheta += s
            regtheta = float(regtheta[1:]) + 90
        # end with open(crtf) as file

        beamreg = Ellipse((xcent, ycent), width=lena, height=lenb, angle=regtheta, alpha=0.3,
                          facecolor='orange', zorder=4, edgecolor='orange')
        print("beamreg")
        print(beamreg)
        alax.add_patch(beamreg)
    # end if crtf

    ''' second set of axes: MIPS imaging '''
    miax = fig.add_subplot(232, projection=WCSlist[0])

    # tranformation to line the MIPS axes up
    mips_transform = miax.get_transform(WCSlist[2])

    miax.imshow(datalist[2], cmap=plt.cm.gray, vmin=mipsext[0], vmax=mipsext[1], zorder=1, transform=mips_transform)
    miax.contour(datalist[0][0][0], colors='red', levels=levels, zorder=5)

    if whole_image == True:
        miax.set_xlim(xlim)
        miax.set_ylim(ylim)

        beambkg = Rectangle((xlim[0], ylim[0]), width=arad * 1.5, height=brad * 1.5, facecolor='w',
                            edgecolor='k', zorder=10, alpha=0.9)
        beam = Ellipse((xlim[0] + arad * 0.75, ylim[0] + brad * 0.75), alpha=0.9,
                       width=arad, height=brad, angle=theta, facecolor='yellow', edgecolor='k', zorder=11)

    else:
        xlims = miax.set_xlim([int(xcent) - npix, int(xcent) + npix])
        ylims = miax.set_ylim([int(ycent) - npix, int(ycent) + npix])

        beambkg = Rectangle((xlims[0], ylims[0]), width=arad * 1.5, height=brad * 1.5, facecolor='w',
                            edgecolor='k', zorder=10, alpha=0.9)
        beam = Ellipse((xlims[0] + arad * 0.75, ylims[0] + brad * 0.75), alpha=0.9,
                       width=arad, height=brad, angle=theta, facecolor='yellow', edgecolor='k', zorder=11)

    #     alax.add_patch(beambkg)
    miax.add_patch(beam)
    miax.set_xlabel('RA')
    miax.tick_params(
        axis='y',
        which='both',
        left='on',
        right='off',
        direction='out',
        labelleft=False,
        labelright=False,
        length=5
    )
    miax.tick_params(
        axis='x',
        labelsize='small'
    )
    miax.set_title(r'MIPS $24\ \mu\mathrm{m}$')

    if max_scale is not None:
        mcirc = Ellipse([xpeak, ypeak], width=max_scale, height=max_scale,
                        facecolor='green', edgecolor='k', zorder=3, alpha=0.3)
        miax.add_patch(mcirc)

    if deltap is not None:
        #         miax.scatter(xpeak, ypeak, color='orange', zorder=20)
        horline = Line2D([xpeak - deltap, xpeak + deltap], [ypeak, ypeak], color='orange', zorder=10)
        vertline = Line2D([xpeak, xpeak], [ypeak - deltap, ypeak + deltap], color='orange', zorder=10)
        miax.add_line(horline)
        miax.add_line(vertline)

    ''' third set of axes: IRAC imaging '''
    irax = fig.add_subplot(233, projection=WCSlist[0])

    # tranformation to line the IRAC axes up
    irac_transform = irax.get_transform(WCSlist[3])

    irax.imshow(datalist[3], cmap=plt.cm.gray, vmin=iracext[0], vmax=iracext[1], zorder=1, transform=irac_transform)
    irax.contour(datalist[0][0][0], colors='red', levels=levels, zorder=5)

    if whole_image == True:
        irax.set_xlim(xlim)
        irax.set_ylim(ylim)

        beambkg = Rectangle((xlim[0], ylim[0]), width=arad * 1.5, height=brad * 1.5, facecolor='w',
                            edgecolor='k', zorder=10, alpha=0.9)
        beam = Ellipse((xlim[0] + arad * 0.75, ylim[0] + brad * 0.75), alpha=0.9,
                       width=arad, height=brad, angle=theta, facecolor='yellow', edgecolor='k', zorder=11)

    else:
        xlims = irax.set_xlim([int(xcent) - npix, int(xcent) + npix])
        ylims = irax.set_ylim([int(ycent) - npix, int(ycent) + npix])

        beambkg = Rectangle((xlims[0], ylims[0]), width=arad * 1.5, height=brad * 1.5, facecolor='w',
                            edgecolor='k', zorder=10, alpha=0.9)
        beam = Ellipse((xlims[0] + arad * 0.75, ylims[0] + brad * 0.75), alpha=0.9,
                       width=arad, height=brad, angle=theta, facecolor='yellow', edgecolor='k', zorder=11)

    #     alax.add_patch(beambkg)
    irax.add_patch(beam)
    irax.set_xlabel('RA')
    irax.tick_params(
        axis='y',
        which='both',
        left='on',
        right='off',
        direction='out',
        labelleft=False,
        labelright=False,
        length=5
    )
    irax.tick_params(
        axis='x',
        labelsize='small'
    )
    irax.set_title(r'IRAC $3.6\ \mu\mathrm{m}$')

    if max_scale is not None:
        mcirc = Ellipse([xpeak, ypeak], width=max_scale, height=max_scale,
                        facecolor='green', edgecolor='k', zorder=3, alpha=0.3)
        irax.add_patch(mcirc)

    if deltap is not None:
        #         irax.scatter(xpeak, ypeak, color='orange', zorder=20)
        horline = Line2D([xpeak - deltap, xpeak + deltap], [ypeak, ypeak], color='orange', zorder=10)
        vertline = Line2D([xpeak, xpeak], [ypeak - deltap, ypeak + deltap], color='orange', zorder=10)
        irax.add_line(horline)
        irax.add_line(vertline)

    ''' Spectral profile axes '''
    # load in the spectral profile from the file
    specprof = np.genfromtxt(files[4])
    vel, flux = specprof[:, 0], specprof[:, 1]

    zmin, zmax = zvals[0], zvals[1]

    # xarray to fit
    x = np.linspace(np.min(vel), np.max(vel), num=500)

    spax = fig.add_subplot(212)

    # plot the intensity data to make sure the fit worked
    spax.bar(vel, flux * 1e3, width=np.abs(vel[1] - vel[0]))
    spax.set_xlabel('Velocity (km/s), LSRK)')
    spax.set_ylabel('Integrated Intensity (mJy)')
    spax.set_title('CO (2-1) Spectral Profile')

    if fit == True:
        # adjust the guess fit parameters to match the data
        if centmean:
            meanval = np.mean(x)
            p0[1] = meanval
        if not p02:
            # if no fit parameters are passed, guess based off the fit parameters for one gaussian
            p02 = [p0[0], p0[0], p0[1] - 50, p0[1] + 50, p0[2], p0[2], 0]

        # fit to a gaussian
        opt1, cov1 = curve_fit(gaussian, vel, flux, p0=p0, bounds=bounds)
        # fit to two gaussians
        opt2, cov2 = curve_fit(twogaussian, vel, flux, p0=p02)

        # find the reduced chi-squared value for each fit
        rms1 = co.rmsfunc(flux[zmin:zmax], linefree=linefree)
        rms2 = co.rmsfunc(flux[zmin:zmax], linefree=linefree)

        chi1 = co.chi(flux[zmin:zmax], gaussian(vel[zmin:zmax], *opt1), 4, linefree=linefree)
        chi2 = co.chi(flux[zmin:zmax], twogaussian(vel[zmin:zmax], *opt2), 8, linefree=linefree)

        if not n_gauss:
            if chi1 < chi2:
                use_twog = False
                rms = rms1
                opt = opt1
            elif chi1 > chi2:
                use_twog = True
                rms = rms2
                opt = opt2
        else:
            if n_gauss == 1:
                use_twog = False
                rms = rms1
                opt = opt1
            elif n_gauss == 2:
                use_twog = True
                rms = rms2
                opt = opt2
            else:
                print("Warning: Calling for a number other than 1 or 2 Gaussians")
                return

        if use_twog == False:
            spax.plot(x, gaussian(x, *opt1) * 1e3, color='orange')
        else:
            spax.plot(x, twogaussian(x, *opt2) * 1e3, color='orange')

    freq_to_vel = u.doppler_radio(230.528 * u.GHz)
    if centline:
        centfreq = 230.538 * u.GHz / (1 + z)
        centvel = centfreq.to(u.km / u.s, equivalencies=freq_to_vel)

        spax.axvline(x=0, zorder=20, color='k')

    if spaxlims:
        spax.set_xlim(spaxlims)

    spax.tick_params(
        axis='x',
        labelsize='small'
    )
    spax.yaxis.set_label_position('left')
    spax.tick_params(
        axis='y',
        left='on',
        right='off',
        labelright=False,
        labelleft=True
    )

    spax.set_title('Contour steps: {}sig, Redshift: z = {}'.format(step, z))

    return datalist


def cont_stamp_array(files, mom_cutout, npix, z, optext=(0, 80), mipsext=(0, 0.5), iracext=(0, 1), delp=None,
                     maxscale=None, max_stds=50, step=2, whole_image=False, startstds=2, SN=None,
                     max_scale=None, deltap=None, fit=True, crtf=None, ycutout=None,
                     xpeakpass=None, ypeakpass=None, xcentpass=None, ycentpass=None, std=None, bcgcent=None,
                     ozdes=None):
    ''' make an array of postage stamp images for easy analysis of a single souce.
        INPUTS:
            - files: (momfile, optfile, mipsfile, iracfile) list of file pointers
            - mom_cutout: (xmin, xmax) coordinates of a region around the source in the moment 0 file small
                enough that the source can be fit to a 2D gaussian to do statistics with
            - npix: number of moment 0 native pixels on each side of the center of the source to include in stamps
            - z: redshift of the source (to put a reference line down in the spectral profile)
            - optmin, optmax: vmin, vmax for the DES optical colorscale
            - delp: positional uncertainty for ALMA moment 0 (if not none, will plot a cross centered around the
                middle of the source to show by how much the position is uncertain)
            - maxscale: maximum recoverable angular scale for ALMA observations (if not none, will plot a circle
                showing how much of the structure falls into this scale)
            - max_stds: maximum number of S/N sigma to reach with the contours
            - step: spacing between moment 0 contours, in S/N sigma
            - whole_image: plot the entire extent of the moment 0 image, instead of just a cutout
            - startstds: S/N level above RMS at which contours should start
            - ozdes: data from the ozdes fits file with all the redshifts
    '''

    # open all of the files and get their data and wcs
    # files are organized like files[0]: momfile, files[1]: optfile, files[2]: mipsfile, files[3]: iracfile
    # files[4]: specfile
    WCSlist = []
    datalist = []
    hdulist = []
    for file in files[0:4]:
        hdu = fits.open(file)[0]
        hdulist.append(hdu)
        WCSlist.append(wcs.WCS(hdu.header).sub(['celestial']))
        data = hdu.data
        data[np.where(np.isnan(data))] = 0.
        data[np.where(data == -np.inf)] = 0.
        data[np.where(data == np.inf)] = 0.
        if len(data) == 4:
            data = data[0, 0, :, :]
        datalist.append(data)

    if ozdes is not None:
        # split apart the ozdes data into only the useful stuff:
        ozdesra = ozdes['RA']
        ozdesdec = ozdes['DEC']
        ozdesz = ozdes['z']

        # make a skycoord array of all the ozdes coordinates
        ozdescoords = SkyCoord(ra=ozdesra*u.deg, dec=ozdesdec*u.deg, frame='icrs')

        haszidx = np.where(WCSlist[0].footprint_contains(ozdescoords))
        haszcoords = ozdescoords[haszidx]
        haszpixcoords = haszcoords.to_pixel(WCSlist[0])
        haszredshift = ozdesz[haszidx]

    # define the figure with 3 sets of axes
    fig = plt.figure(figsize=(15, 5))

    ''' first set of axes: this is the moment 0 overlaid on DES '''
    # get moment 0 statistics to plot S/N contours
    mommax = np.nanmax(datalist[0])
    if not SN:
        momstd, momSN, xcent, ycent, xpeak, ypeak = find_SN_sigma(files[0], mom_cutout)
    else:
        momstd = std
        momSN = SN
        xcent = xcentpass
        ycent = ycentpass
        xpeak = xpeakpass
        ypeak = ypeakpass

    levels = np.ones((max_stds - startstds) // step) * momstd
    stds = np.arange(startstds, max_stds, step=step) * momstd
    levels = stds

    # plot the contours first
    alax = fig.add_subplot(131, projection=WCSlist[0])
    alax.contour(datalist[0][0][0], colors='red', levels=levels, zorder=5)

    #     custom_lines = [Line2D([0],[0], color='red', lw=2)]
    #     alax.legend(custom_lines, ['{} sigma steps'.format(step)])

    xlim = alax.get_xlim()
    ylim = alax.get_ylim()

    # WCS transform to line the optical image up
    opt_transform = alax.get_transform(WCSlist[1])

    # plot the optical
    im = alax.imshow(datalist[1], cmap=plt.cm.gray, transform=opt_transform, vmin=optext[0], vmax=optext[1], zorder=1)

    ''' Show the beam'''
    # get the semimajor and semiminor axis values from the header
    a = (hdulist[0].header['BMAJ'] * u.deg)
    b = (hdulist[0].header['BMIN'] * u.deg)
    theta = (hdulist[0].header['BPA'] * u.deg).value + 90

    # define skycoord objects for the extent of the axes to change into pix
    center = SkyCoord(hdulist[0].header['CRVAL1'] * u.deg, hdulist[0].header['CRVAL2'] * u.deg, frame='fk5')
    ext = SkyCoord(hdulist[0].header['CRVAL1'] * u.deg - a, hdulist[0].header['CRVAL2'] * u.deg + b, frame='fk5')

    # change the wcs into native pixels
    centerpix = WCSlist[0].world_to_pixel(center)
    extpix = WCSlist[0].world_to_pixel(ext)

    # center values are the center of the ellipse, the lengths are the differences between center and ext
    arad, brad = int(extpix[0] - centerpix[0]), int(extpix[1] - centerpix[1])

    alax.set_xlabel('RA')
    alax.set_ylabel('DEC')

    if whole_image == True:
        alax.set_xlim(xlim)
        alax.set_ylim(ylim)

        beambkg = Rectangle((xlim[0], ylim[0]), width=arad * 1.5, height=brad * 1.5, facecolor='w',
                            edgecolor='k', zorder=10, alpha=0.9)
        beam = Ellipse((xlim[0] + arad * 0.75, ylim[0] + brad * 0.75), alpha=0.9,
                       width=arad, height=brad, angle=theta, facecolor='yellow', edgecolor='k', zorder=11)

    else:
        xlims = alax.set_xlim([int(xcent) - npix, int(xcent) + npix])
        ylims = alax.set_ylim([int(ycent) - npix, int(ycent) + npix])

        if bcgcent:
            bcgpixcent = bcgcent.to_pixel(WCSlist[0])
            xlims = alax.set_xlim([int(bcgpixcent[0]) - npix, int(bcgpixcent[0]) + npix])
            ylims = alax.set_ylim([int(bcgpixcent[1]) - npix, int(bcgpixcent[1]) + npix])

        beambkg = Rectangle((xlims[0], ylims[0]), width=arad * 1.5, height=brad * 1.5, facecolor='w',
                            edgecolor='k', zorder=10, alpha=0.9)
        beam = Ellipse((xlims[0] + arad * 0.75, ylims[0] + brad * 0.75), alpha=0.9,
                       width=arad, height=brad, angle=theta, facecolor='yellow', edgecolor='k', zorder=11)

    #     alax.add_patch(beambkg)
    alax.add_patch(beam)
    alax.set_title('DES z-band')
    alax.tick_params(
        axis='y',
        which='both',
        left='on',
        right='off',
        direction='out',
        labelleft=True,
        length=5,
        labelsize='small'
    )
    alax.tick_params(
        axis='x',
        labelsize='small'
    )

    if max_scale is not None:
        arcsecperpix = np.abs((hdulist[0].header['CDELT1'] * u.deg).to(u.arcsec)).value
        max_scale = max_scale / arcsecperpix
        mcirc = Ellipse((xcent, ycent), width=max_scale, height=max_scale,
                        facecolor='green', edgecolor='green', zorder=10, alpha=0.3)
        print("maxscale")
        print(mcirc)
        alax.add_patch(mcirc)

    if deltap is not None:
        arcsecperpix = np.abs((hdulist[0].header['CDELT1'] * u.deg).to(u.arcsec)).value
        deltap = deltap / arcsecperpix
        horline = Line2D([xpeak - deltap, xpeak + deltap], [ypeak, ypeak], color='orange', zorder=10)
        vertline = Line2D([xpeak, xpeak], [ypeak - deltap, ypeak + deltap], color='orange', zorder=10)
        alax.add_line(horline)
        alax.add_line(vertline)


    if bcgcent:
        bcgpixcent = bcgcent.to_pixel(WCSlist[0])
        alax.scatter(*bcgpixcent, color='green', zorder=10, s=100, marker='+')
        xlims = alax.set_xlim([int(bcgpixcent[0]) - npix, int(bcgpixcent[0]) + npix])
        ylims = alax.set_ylim([int(bcgpixcent[1]) - npix, int(bcgpixcent[1]) + npix])

    if ozdes is not None:
        alax.scatter(*haszpixcoords, zorder=20, color='tab:orange', marker='x')
        for i in range(len(haszpixcoords[0])):
            alax.text(haszpixcoords[0][i]+8, haszpixcoords[1][i]+8, str(haszredshift[i]),
                     bbox=dict(facecolor='tab:orange', edgecolor='tab:orange', alpha=0.7, boxstyle='round'))


    if crtf:
        ''' show the region over which the spectral profile was taken'''
        with open(crtf, 'r') as file:
            txt = file.read().split()

            xcent = '0'
            for s in txt[1]:
                if (s.isdigit() or (s == '.')):
                    xcent += s
            xcent = float(xcent[1:])

            ycent = '0'
            for s in txt[2]:
                if (s.isdigit() or (s == '.')):
                    ycent += s
            ycent = float(ycent[1:])

            lena = '0'
            for s in txt[3]:
                if (s.isdigit() or (s == '.')):
                    lena += s
            lena = float(lena[1:]) * 2

            lenb = '0'
            for s in txt[4]:
                if (s.isdigit() or (s == '.')):
                    lenb += s
            lenb = float(lenb[1:]) * 2

            regtheta = '0'
            for s in txt[5]:
                if (s.isdigit() or (s == '.')):
                    regtheta += s
            regtheta = float(regtheta[1:]) + 90
        # end with open(crtf) as file

        beamreg = Ellipse((xcent, ycent), width=lena, height=lenb, angle=regtheta, alpha=0.3,
                          facecolor='orange', zorder=4, edgecolor='orange')
        print("beamreg")
        print(beamreg)
        alax.add_patch(beamreg)
    # end if crtf



    ''' second set of axes: MIPS imaging '''
    miax = fig.add_subplot(132, projection=WCSlist[0])

    # tranformation to line the MIPS axes up
    mips_transform = miax.get_transform(WCSlist[2])

    miax.imshow(datalist[2], cmap=plt.cm.gray, vmin=mipsext[0], vmax=mipsext[1], zorder=1, transform=mips_transform)
    miax.contour(datalist[0][0][0], colors='red', levels=levels, zorder=5)

    if whole_image == True:
        miax.set_xlim(xlim)
        miax.set_ylim(ylim)

        beambkg = Rectangle((xlim[0], ylim[0]), width=arad * 1.5, height=brad * 1.5, facecolor='w',
                            edgecolor='k', zorder=10, alpha=0.9)
        beam = Ellipse((xlim[0] + arad * 0.75, ylim[0] + brad * 0.75), alpha=0.9,
                       width=arad, height=brad, angle=theta, facecolor='yellow', edgecolor='k', zorder=11)

    else:
        xlims = miax.set_xlim([int(xcent) - npix, int(xcent) + npix])
        ylims = miax.set_ylim([int(ycent) - npix, int(ycent) + npix])

        if bcgcent:
            bcgpixcent = bcgcent.to_pixel(WCSlist[0])
            xlims = miax.set_xlim([int(bcgpixcent[0]) - npix, int(bcgpixcent[0]) + npix])
            ylims = miax.set_ylim([int(bcgpixcent[1]) - npix, int(bcgpixcent[1]) + npix])

        beambkg = Rectangle((xlims[0], ylims[0]), width=arad * 1.5, height=brad * 1.5, facecolor='w',
                            edgecolor='k', zorder=10, alpha=0.9)
        beam = Ellipse((xlims[0] + arad * 0.75, ylims[0] + brad * 0.75), alpha=0.9,
                       width=arad, height=brad, angle=theta, facecolor='yellow', edgecolor='k', zorder=11)

    #     alax.add_patch(beambkg)
    miax.add_patch(beam)
    miax.set_xlabel('RA')
    miax.tick_params(
        axis='y',
        which='both',
        left='on',
        right='off',
        direction='out',
        labelleft=False,
        labelright=False,
        length=5
    )
    miax.tick_params(
        axis='x',
        labelsize='small'
    )
    miax.set_title(r'MIPS $24\ \mu\mathrm{m}$')

    if max_scale is not None:
        mcirc = Ellipse([xcent, ycent], width=max_scale, height=max_scale,
                        facecolor='green', edgecolor='k', zorder=3, alpha=0.3)
        miax.add_patch(mcirc)

    if deltap is not None:
        #         miax.scatter(xpeak, ypeak, color='orange', zorder=20)
        horline = Line2D([xpeak - deltap, xpeak + deltap], [ypeak, ypeak], color='orange', zorder=10)
        vertline = Line2D([xpeak, xpeak], [ypeak - deltap, ypeak + deltap], color='orange', zorder=10)
        miax.add_line(horline)
        miax.add_line(vertline)

    if bcgcent:
        bcgpixcent = bcgcent.to_pixel(WCSlist[0])
        miax.scatter(*bcgpixcent, color='green', zorder=10, s=100, marker='+')
        xlims = miax.set_xlim([int(bcgpixcent[0]) - npix, int(bcgpixcent[0]) + npix])
        ylims = miax.set_ylim([int(bcgpixcent[1]) - npix, int(bcgpixcent[1]) + npix])

    if ozdes is not None:
        miax.scatter(*haszpixcoords, zorder=20, color='tab:orange', marker='x')
        for i in range(len(haszpixcoords[0])):
            miax.text(haszpixcoords[0][i]+8, haszpixcoords[1][i]+8, str(haszredshift[i]),
                     bbox=dict(facecolor='tab:orange', edgecolor='tab:orange', alpha=0.7, boxstyle='round'))

    ''' third set of axes: IRAC imaging '''
    irax = fig.add_subplot(133, projection=WCSlist[0])

    # tranformation to line the IRAC axes up
    irac_transform = irax.get_transform(WCSlist[3])

    irax.imshow(datalist[3], cmap=plt.cm.gray, vmin=iracext[0], vmax=iracext[1], zorder=1, transform=irac_transform)
    irax.contour(datalist[0][0][0], colors='red', levels=levels, zorder=5)

    if whole_image == True:
        irax.set_xlim(xlim)
        irax.set_ylim(ylim)

        beambkg = Rectangle((xlim[0], ylim[0]), width=arad * 1.5, height=brad * 1.5, facecolor='w',
                            edgecolor='k', zorder=10, alpha=0.9)
        beam = Ellipse((xlim[0] + arad * 0.75, ylim[0] + brad * 0.75), alpha=0.9,
                       width=arad, height=brad, angle=theta, facecolor='yellow', edgecolor='k', zorder=11)

    else:
        xlims = irax.set_xlim([int(xcent) - npix, int(xcent) + npix])
        ylims = irax.set_ylim([int(ycent) - npix, int(ycent) + npix])

        if bcgcent:
            bcgpixcent = bcgcent.to_pixel(WCSlist[0])
            xlims = irax.set_xlim([int(bcgpixcent[0]) - npix, int(bcgpixcent[0]) + npix])
            ylims = irax.set_ylim([int(bcgpixcent[1]) - npix, int(bcgpixcent[1]) + npix])

        beambkg = Rectangle((xlims[0], ylims[0]), width=arad * 1.5, height=brad * 1.5, facecolor='w',
                            edgecolor='k', zorder=10, alpha=0.9)
        beam = Ellipse((xlims[0] + arad * 0.75, ylims[0] + brad * 0.75), alpha=0.9,
                       width=arad, height=brad, angle=theta, facecolor='yellow', edgecolor='k', zorder=11)

    #     alax.add_patch(beambkg)
    irax.add_patch(beam)
    irax.set_xlabel('RA')
    irax.tick_params(
        axis='y',
        which='both',
        left='on',
        right='off',
        direction='out',
        labelleft=False,
        labelright=False,
        length=5
    )
    irax.tick_params(
        axis='x',
        labelsize='small'
    )
    irax.set_title(r'IRAC $3.6\ \mu\mathrm{m}$')

    if max_scale is not None:
        mcirc = Ellipse([xcent, ycent], width=max_scale, height=max_scale,
                        facecolor='green', edgecolor='k', zorder=3, alpha=0.3)
        irax.add_patch(mcirc)

    if deltap is not None:
        #         irax.scatter(xpeak, ypeak, color='orange', zorder=20)
        horline = Line2D([xpeak - deltap, xpeak + deltap], [ypeak, ypeak], color='orange', zorder=10)
        vertline = Line2D([xpeak, xpeak], [ypeak - deltap, ypeak + deltap], color='orange', zorder=10)
        irax.add_line(horline)
        irax.add_line(vertline)

    if bcgcent:
        bcgpixcent = bcgcent.to_pixel(WCSlist[0])
        irax.scatter(*bcgpixcent, color='green', zorder=10, s=100, marker='+')
        xlims = irax.set_xlim([int(bcgpixcent[0]) - npix, int(bcgpixcent[0]) + npix])
        ylims = irax.set_ylim([int(bcgpixcent[1]) - npix, int(bcgpixcent[1]) + npix])

    if ozdes is not None:
        irax.scatter(*haszpixcoords, zorder=20, color='tab:orange', marker='x')
        for i in range(len(haszpixcoords[0])):
            irax.text(haszpixcoords[0][i]+8, haszpixcoords[1][i]+8, str(haszredshift[i]),
                     bbox=dict(facecolor='tab:orange', edgecolor='tab:orange', alpha=0.7, boxstyle='round'))

    return datalist



def plot_velocity_overlay(files, optmin, optmax, xlims, ylims, alpha=0.05, vmin=None, vmax=None, step=None):
    ''' function to plot the filled contours of a radio velocity field over an optical image.
        files should be structured ('optical file path', 'moment 1 file path') to fits files. Optmin and optmax are
        the vmin, vmax of the colourmap for the optical image.
    '''

    opthdu = fits.open(files[0])[0]
    momhdu = fits.open(files[1])[0]
    if len(files) > 2:
        plotmom0 = True
        mom0hdu = fits.open(files[2])[0]
        mom0wcs = wcs.WCS(mom0hdu.header)
        mom0wcs = mom0wcs.sub(['celestial'])
        mom0data = mom0hdu.data[0, 0, :, :]
        mom0data[np.where(mom0data == -np.inf)] = 0.
        mom0data[np.where(mom0data == np.inf)] = 0.
        mom0data[np.where(np.isnan(mom0data))] = 0.
    else:
        plotmom0 = False

    # read in WCS information for both images
    optwcs = wcs.WCS(opthdu.header)
    momwcs = wcs.WCS(momhdu.header)

    momwcs = momwcs.sub(['celestial'])

    # define contour levels for the velocity field. Contours start at 2 sigma and increase by 1sigma intervals
    momdata = momhdu.data[0, 0, :, :]
    # pass any infinities to zero to avoid them messing up statistics
    momdata[np.where(momdata == -np.inf)] = 0.
    momdata[np.where(momdata == np.inf)] = 0.
    momstd = np.nanstd(momdata)
    momdata[np.where(np.isnan(momdata))] = -9e10
    mommax = np.nanmax(momdata)

    levels = np.ones(8) * mommax
    stds = np.arange(10, 2, step=-1) * momstd
    levels = (levels - stds)

    if not vmin:
        vmin = np.min(momdata[np.where(momdata > -9e9)])
    if not vmax:
        vmax = np.max(momdata)


    if not step:
        levels = np.arange(vmin, vmax, step=momstd / 4)
    else:
        levels = np.arange(vmin, vmax, step=step)

    momdata[np.where(momdata < vmin)] = -9e10
    momdata[np.where(momdata > vmax)] = -9e10
    momdata[np.where(momdata == -9e10)] = np.nan

    # plot the velocity field
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection=momwcs)
    cont = ax.pcolor(momdata, cmap=plt.cm.Spectral, alpha=alpha, vmin=vmin, vmax=vmax, ec='none')  # , levels=levels)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # WCS transform to line the images up
    opt_transform = ax.get_transform(optwcs)

    # plot the optical
    im = ax.imshow(opthdu.data, cmap=plt.cm.gray, transform=opt_transform, vmin=optmin, vmax=optmax)

    if plotmom0:
        # get moment 0 statistics to plot S/N contours
        mom0max = np.nanmax(mom0data)
        momstd, momSN, xcent, ycent, xpeak, ypeak = find_SN_sigma(files[2], xlims)
        levels = np.ones((50 - 2) // step) * momstd
        stds = np.arange(50, 2, step=2) * momstd
        levels = stds

        # plot the moment 0 contours over the moment 1
        ax.contour(mom0data, colors='red', levels=levels, zorder=5)

    # colorbar will correspond to the velocity field, in units of km/s
    cbar = plt.colorbar(cont, fraction=0.046, pad=0.07)

    cbar.ax.set_ylabel("Velocity (km/s)")

    # get the semimajor and semiminor axis values from the header
    a = (momhdu.header['BMAJ'] * u.deg)
    b = (momhdu.header['BMIN'] * u.deg)
    theta = (momhdu.header['BPA'] * u.deg).to(u.rad).value

    # define skycoord objects for the extent of the axes to change into pix
    center = SkyCoord(momhdu.header['CRVAL1'] * u.deg, momhdu.header['CRVAL2'] * u.deg, frame='fk5')
    ext = SkyCoord(momhdu.header['CRVAL1'] * u.deg - a, momhdu.header['CRVAL2'] * u.deg + b, frame='fk5')

    # change the wcs into native pixels
    centerpix = momwcs.world_to_pixel(center)
    extpix = momwcs.world_to_pixel(ext)

    # center values are the center of the ellipse, the lengths are the differences between center and ext
    arad, brad = int(extpix[0] - centerpix[0]), int(extpix[1] - centerpix[1])

    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')

    ax.set_xlim([*xlims])
    ax.set_ylim([*ylims])

    beambkg = Rectangle((xlims[0], ylims[0]), width=arad * 1.5, height=brad * 1.5, facecolor='w',
                        edgecolor='k', zorder=10, alpha=0.9)
    beam = Ellipse((xlims[0] + arad * 0.75, ylims[0] + brad * 0.75),
                   width=arad, height=brad, angle=theta, facecolor='w', edgecolor='k', zorder=11)

    ax.add_patch(beambkg)
    ax.add_patch(beam)

    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    ax.set_xlim([*xlims])
    ax.set_ylim([*ylims])
    return




'''
SPECIAL 2-LINE MOMENT PLOT FOR XMM-8
opthdu = fits.open(xmm5_cubefiles[1])[0]
momhdu1 = fits.open(xmm5_cubefiles[5])[0]
momhdu2 = fits.open(xmm5_cubefiles[6])[0]

optwcs = wcs.WCS(opthdu.header)
momwcs = wcs.WCS(momhdu1.header)

momwcs = momwcs.sub(['celestial'])

%matplotlib notebook

momdata1 = momhdu1.data[0,0,:,:]
mommax1 = np.nanmax(momdata1)
momstd1 = np.nanstd(momdata1)
levels1 = np.ones(8)*mommax1
stds1 = np.arange(10, 2, step=-1)*momstd1
levels1 = (levels1 - stds1)

momdata2 = momhdu2.data[0,0,:,:]
mommax2 = np.nanmax(momdata2)
momstd2 = np.nanstd(momdata2)
levels2 = np.ones(8)*mommax2
stds2 = np.arange(10, 2, step=-1)*momstd2
levels2 = (levels2 - stds2)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection=momwcs)
cs1 = ax.contour(momdata1, colors='red', levels=levels1)
cs2 = ax.contour(momdata2, colors='lime', levels=levels2)

custom_lines = [Line2D([0], [0], color='red', lw=2),
                Line2D([0], [0], color='lime', lw=2)]

ax.legend(custom_lines, ['52574.1 km/s', '52208.3 km/s'])

xlim = ax.get_xlim()
ylim = ax.get_ylim()

opt_transform = ax.get_transform(optwcs)

im = ax.imshow(opthdu.data, cmap=plt.cm.gray, transform=opt_transform, vmin=1500, vmax=1700)

cbar = plt.colorbar(im, fraction=0.046, pad=0.07)

# get the semimajor and semiminor axis values from the header
a = (momhdu1.header['BMAJ']*u.deg)
b = (momhdu1.header['BMIN']*u.deg)
theta = momhdu1.header['BPA']


# define skycoord objects for the extent of the axes to change into pix
center = SkyCoord(momhdu1.header['CRVAL1']*u.deg, momhdu1.header['CRVAL2']*u.deg, frame='fk5')
ext = SkyCoord(momhdu1.header['CRVAL1']*u.deg - a, momhdu1.header['CRVAL2']*u.deg + b, frame='fk5')

# change the wcs into native pixels
centerpix = momwcs.world_to_pixel(center)
extpix = momwcs.world_to_pixel(ext)

# center values are the center of the ellipse, the lengths are the differences between center and ext
arad, brad = int(extpix[0] - centerpix[0]), int(extpix[1] - centerpix[1])



ax.set_xlabel('RA')
ax.set_ylabel('DEC')


beambkg = Rectangle((83, 83), width=arad*1.5, height=brad*1.5, facecolor='w',
                    edgecolor='k',zorder=10, alpha=0.9)
beam = Ellipse((83 + arad*0.75, 83 + brad*0.75),
               width=arad, height=brad, angle=theta, facecolor='w', edgecolor='k', zorder=11)

ax.add_patch(beambkg)
ax.add_patch(beam)


ax.set_xlim([83, 133])
ax.set_ylim([83, 133])
'''
