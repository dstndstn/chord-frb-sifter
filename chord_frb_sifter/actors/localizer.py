"""
This a CHORD/FRB prototype for the localizer.

It determines the best-fit sky position of the event based on the S/N per beam.
"""

import numpy as np
from scipy.optimize import least_squares
import cfbm
from datetime import datetime

from chord_frb_sifter.actors.actor import Actor

class Localizer(Actor):
    """
    Determines the best-fit sky position of an astrophysical event
    based on the signal-to-noise ratio (S/N) measured in multiple telescope beams.

    Fits a 2D Gaussian function to the detected S/Ns with only the position in
    unit-sphere coordinates as free parameters. It then converts to equatorial
    coordinates (RA, Dec) using the beam model. The uncertainties in RA and Dec 
    are also estimated.
    """

    def __init__(self, **kwargs):

        self.bm = cfbm.current_model_class()


    def _perform_action(self, event):
        """
        Perform the localization action on the given event.
        """


        # fix this after making L1Events recarray?
        beams = []
        snrs = []
        for e in event["l1_events"]:
            beams.append(e["beam"])
            snrs.append(e["snr"])
        beams = np.array(beams)
        snrs = np.array(snrs)

        # get beam positions from the beam model. If calced earlier and attached
        # to L2 header can use that instead.
        step1 = self.bm.get_beam_positions(beams,freqs=self.bm.clamp_freq)[:,0,:].T
        x,y = self.bm.get_cartesian_from_position(*step1)

        central_freq = 600.0 # 600.0 MHz for CHIME, should be 900.0 for CHORD

        x_out, y_out, x_err, y_err = fit_2dgauss_simplified(x,y,snrs,central_freq)

        # For CHORD will translate directly from unit sphere coords to equat.
        pos = self.bm.get_position_from_cartesian(x_out, y_out)
        #t = np.datetime64(int(event["timestamp_utc"]),"us")

        # ephem needs time to be a datetime object (and not a np.datetime64)
        t = datetime.utcfromtimestamp(event["timestamp_utc"] / 1e6)
        ra, dec = self.bm.get_equatorial_from_position(pos[0],pos[1],t)

        # Setting these to what they are called in the DB schema.
        event["ra"] = ra
        event["dec"] = dec

        # Estimate errors in RA/Dec from errors in x/y, assuming small angle approx.
        # Should be reasonable if near meridian and errors are small.
        event["ra_err"] = np.rad2deg(x_err)
        event["dec_err"] = np.rad2deg(y_err)
        print(ra,dec,x_err,y_err)

        return [event]
        

def gauss2d(xy,A,x0,y0,sigma_x,sigma_y):
    return A * np.exp(-(((xy[0] - x0)**2) / (2 * sigma_x**2) + ((xy[1] - y0)**2) / (2 * sigma_y**2)))

def residuals_gauss2d_analytical_width(p0,xy,sigma_x,sigma_y,snr):

    x0 = p0[0]
    y0 = p0[1]
    
    gauss_xy = gauss2d(xy,1.0,x0,y0,sigma_x,sigma_y)
    A = np.dot(gauss_xy,snr) / np.dot(gauss_xy,gauss_xy)

    return snr - A*gauss_xy

def fit_2dgauss_simplified(x,y,snr,central_freq):
    """
    Simplified = fixed sigmas (based on freq of detection) and amplitude calcualted analytically for each trial position.
    """
    max_i = np.argmax(snr)
    p0 = [
        x[max_i], # max S/N beam x
        y[max_i], # max S/N beam y
    ]

    # this is FWHM at 900 MHz, TODO: calculate at central_freq
    # Eventually we will want to use the center of the subband for the event.
    sigma_x, sigma_y = 0.0046 / 2.355, 0.0068 / 2.355 
    
    bounds = (
        [-1.0, -1.0],
        [1.0, 1.0]
    )
    
    # TODO: provide analytic jacobian function to make faster and more robust?
    result = least_squares(
        residuals_gauss2d_analytical_width,
        p0,
        args = ([x,y],sigma_x,sigma_y,snr),
        bounds=bounds,
    )

    popt = result.x
    x_out, y_out = popt[0], popt[1]

    # estimate output covariance matrix from Jacobian    
    J = result.jac
    #if np.any(J==0): # singular
    #    x_err, y_err = np.nan, np.nan # use a fiducial error (e.g. N*beam width) instead?
    #else:
    try:
        pcov = np.linalg.inv(J.T.dot(J))
        x_err, y_err = np.sqrt(np.diag(pcov))
    except np.linalg.LinAlgError:
        x_err, y_err = np.nan, np.nan

    return x_out, y_out, x_err, y_err
