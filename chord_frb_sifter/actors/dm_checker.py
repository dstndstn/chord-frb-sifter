"""
Compares the DM from an incoming packet to those predicted by the NE2001
and YMW2016 electron-density models.

Based on CHIME's dm_checker module, but simpilfied.
"""

from scipy.interpolate import LinearNDInterpolator
import numpy as np

from chord_frb_sifter.actors.actor import Actor
from frb_L2_L3 import config_dir # will want to replace with CHORD data dir later.


class DMChecker(Actor):
    """
    A subclass of ``ActorBaseClass`` for computing maximum Galactic DMs given
    an L2-estimated line of site, and using the predicted and L2-estimated DMs to
    determine if an unknown astrophysical source is extragalactic (i.e. an FRB) or not
    (i.e. ambiguous or Galactic).

    Parameters
    ----------

    systematic_uncertainty_limit : float
        A fraction of predicted-DM values to use as a lower limit on the systematic 
        uncertainty in calculations for source classification. This is useful for 
        in/near-Plane candidates where the difference in the NE2001 and YMW16 is 
        considerably small, though systematic uncertainty in either model is high.

    ambiguous_threshold : float
        The number of standard deviations used as a threshold for determining
        if the astrophysical signal is an ambiguous source, i.e. if its DM is marginally
        larger than the predicted Galactic component. Default is 2.

    frb_threshold : float
        The number of standard deviations used as a threshold for determining
        if the astrophysical signal is extragalactic, i.e. if its an FRB. Default is 5.

    use_measured_uncertainty : bool
        If True, add measured and systematic uncertainties in quadrature to obtain 
        a "full" measure of uncertainty for use in classification. If False, only 
        use systematic uncertainty in calculations.

    Notes
    -----
    The thresholds are used by comparing them with the difference in measured DM and the
    predicted Galactic DM in units of estimated DM uncertainty. (See Sphinx documentation for
    the equation used here.) If this difference exceeds the threshold, then it is
    considered to be extragalactic.
    """

    def __init__(
        self,
        systematic_uncertainty_limit,
        ambiguous_threshold,
        frb_threshold,
        use_measured_uncertainty,
        **kwargs
    ):

        super(DMChecker, self).__init__(**kwargs)

        # store configuration parameters.
        self.systematic_uncertainty_limit = systematic_uncertainty_limit
        self.ambiguous_threshold = ambiguous_threshold
        self.frb_threshold = frb_threshold
        self.use_measured_uncertainty = use_measured_uncertainty

        # load maps and set up interpolators.
        map_YMW16 = np.load(config_dir + "/data/dm_checker/YMW16_map.npy").T
        map_NE2001 = np.load(config_dir + "/data/dm_checker/NE2001_map.npy").T

        self.interp_map_ymw16 = LinearNDInterpolator(map_YMW16[:2].T, map_YMW16[2].T)
        self.interp_map_ne2001 = LinearNDInterpolator(map_NE2001[:2].T, map_NE2001[2].T)

    def _perform_action(self, event):
        """
        Runs main action, to determine if source is extragalactic, Galactic or 
        statistically ambiguous.
        """

        # RFI or known source -- don't perform DM check.
        if (
            (hasattr(event,"is_rfi") and event.is_rfi)
            or (hasattr(event,"is_known_source") and event.is_known_source)
        ):
            return [event]

        # convert copy of input RA/DEC to Galactic coordinates.
        right_ascension = event.ra
        declination = event.dec
        dm_measured = event.dm
        dm_uncertainty = event.dm_error

        print("Performing action: obtain predicted DMs from maps...")

        dm_ymw16 = self.interp_map_ymw16(right_ascension, declination)[0]
        dm_ne2001 = self.interp_map_ne2001(right_ascension, declination)[0]

        dm_pred = np.array([dm_ne2001, dm_ymw16])
        dm_systematic_error = np.fabs(dm_pred[1] - dm_pred[0])

        # set to uncertainty floor if raw systematic uncertainty is too small.
        if dm_systematic_error / np.max(dm_pred) < self.systematic_uncertainty_limit:
            dm_systematic_error = self.systematic_uncertainty_limit * np.max(dm_pred)

        if not self.use_measured_uncertainty:
            dm_uncertainty = 0.0

        # finally, compare with threshold and return boolean.
        dm_diff = (dm_measured - dm_pred) / np.sqrt(
            dm_uncertainty ** 2 + dm_systematic_error ** 2
        )
        
        # update 'unknown_event_type' attribute in L2/L3 header.
        if all(dm_diff > self.frb_threshold): # extragalactic
            event.unknown_event_type = 1  # FRB

        else:
            if any(dm_diff < self.frb_threshold) and all(
                dm_diff >= self.ambiguous_threshold
            ):
                event.unknown_event_type = 2  # ambiguous

            else:
                event.unknown_event_type = 0  # Galactic

        # update 'max_dm' attribute in accordance with configured DM model.
        event.dm_gal_ymw_2016_max = dm_ymw16
        event.dm_gal_ne_2001_max = dm_ne2001

        return [event]
