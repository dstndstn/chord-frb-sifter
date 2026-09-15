"""
Compares the DM from an incoming packet to those predicted by the NE2025
and YMW2016 electron-density models.

Based on CHIME's dm_checker module, but simpilfied.
"""
import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from chord_frb_sifter.actors import Actor

class DMChecker(Actor):
    """
    A subclass of ``ActorBaseClass`` for computing maximum Galactic DMs given
    an L2-estimated line of site, and using the predicted and L2-estimated DMs to
    determine if an unknown astrophysical source is extragalactic (i.e. an FRB) or not.

    Parameters
    ----------

    systematic_uncertainty_limit : float
        A fraction of predicted-DM values to use as a lower limit on the systematic
        uncertainty in calculations for source classification. This is useful for
        in/near-Plane candidates where the difference in the NE2001 and YMW16 is
        considerably small, though systematic uncertainty in either model is high.

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
        frb_threshold,
        use_measured_uncertainty,
        **kwargs
    ):

        super().__init__(**kwargs)

        # store configuration parameters.
        self.systematic_uncertainty_limit = systematic_uncertainty_limit
        self.frb_threshold = frb_threshold
        self.use_measured_uncertainty = use_measured_uncertainty

        # load maps and set up interpolators.
        import chord_frb_sifter
        config_dir = os.path.join(os.path.dirname(chord_frb_sifter.__file__), 'data', 'dm_checker')
        map_YMW16  = np.load(os.path.join(config_dir, 'YMW16_map.npy' )).T
        map_NE2025 = np.load(os.path.join(config_dir, 'NE2025_map.npy')).T
        # These files are N x 3 arrays, columns = [RA_deg, Dec_deg, DM_max_pc_cm3].
        self.interp_map_ymw16  = LinearNDInterpolator(map_YMW16 [:2].T, map_YMW16 [2].T)
        self.interp_map_ne2025 = LinearNDInterpolator(map_NE2025[:2].T, map_NE2025[2].T)

    def __str__(self):
        return 'DMChecker'

    def _perform_action(self, event_group):
        """
        Runs main action, to determine if source is extragalactic, Galactic or 
        statistically ambiguous.
        """

        for event in event_group.events:
            # RFI or known source -- don't perform DM check.
            if event.is_rfi() or event.is_known_source():
                continue

            dm_measured = event.dm
            dm_uncertainty = event.dm_error

            #print('Looking up predicted DMs from maps...')

            dm_ymw16  = float(self.interp_map_ymw16 (event.ra, event.dec))
            dm_ne2025 = float(self.interp_map_ne2025(event.ra, event.dec))

            dm_pred = np.array([dm_ne2025, dm_ymw16])
            dm_systematic_error = np.fabs(dm_pred[1] - dm_pred[0])

            # set to uncertainty floor if raw systematic uncertainty is too small.
            if dm_systematic_error / np.max(dm_pred) < self.systematic_uncertainty_limit:
                dm_systematic_error = self.systematic_uncertainty_limit * np.max(dm_pred)

            if not self.use_measured_uncertainty:
                dm_uncertainty = 0.0

            # finally, compare with threshold and return boolean.
            dm_diff = (dm_measured - dm_pred) / np.hypot(dm_uncertainty, dm_systematic_error)
        
            # classify into galactic / ambiguous / extragalactic = FRB
            if all(dm_diff > self.frb_threshold):
                event.set_frb()

            event.dm_gal_ymw_2016_max = dm_ymw16
            event.dm_gal_ne_2025_max = dm_ne2025

        return [event_group]
