'''

A simplifying rewrite of the CHIME/FRB Known Source Sifter.


Notes:

-- incoherence beams - for CHIME - should get their Dec uncertainties
   increased / maxed out (not in this class)

'''

import numpy as np
import concurrent.futures as cf
from chord_frb_sifter.actors import Actor
from chord_frb_sifter.common import ra_difference
from chord_frb_db.models import KnownSource

def _load_known_sources(database_engine):
    from sqlalchemy.orm import Session
    from sqlalchemy import select
    with Session(database_engine) as session:
        known = list(session.scalars(select(KnownSource)))
    print('KnownSourceSifter: loaded %d known sources from DB' % len(known))
    return known

class KnownSourceTagger(Actor):
    def __init__(self, threshold=None, #sky_region, dm_region,
                 database_engine=None, **kwargs):
        assert(threshold is not None)
        super().__init__(**kwargs)
        # probability (Bayes factor) to call an event associated with a known source
        self.threshold = threshold
        #self.sky_region = sky_region
        #self.dm_region = dm_region
        self.database_engine = database_engine
        self.ks_database = _load_known_sources(database_engine)
        # For async updating the known sources
        self._executor = cf.ThreadPoolExecutor(max_workers=1)
        self._update_future = None

    def shutdown(self):
        self._executor.shutdown(wait=False)

    def update(self):
        # Trigger a background refresh of the known source catalog.
        if self._update_future is None:
            self._update_future = self._executor.submit(
                _load_known_sources, self.database_engine)

        # if self._update_future is not None and self._update_future.done():
        #     try:
        #         self.ks_database = self._update_future.result()
        #     except Exception:
        #         import traceback; traceback.print_exc()
        #     self._update_future = None
        # FILTER known sources by Dec based on CHORD pointing??
        
    def _perform_action(self, event_group):

        # FIXME - cache these arrays / use a table / recarray?
        known_ra = np.array([s.ra for s in self.ks_database], dtype=np.float32)
        known_dec = np.array([s.dec for s in self.ks_database], dtype=np.float32)
        known_dm = np.array([s.dm for s in self.ks_database], dtype=np.float32)
        known_ra_err = np.array([s.ra_error for s in self.ks_database], dtype=np.float32)
        known_dec_err = np.array([s.dec_error for s in self.ks_database], dtype=np.float32)
        known_dm_err = np.array([s.dm_error for s in self.ks_database], dtype=np.float32)

        # some of the catalog entries have None values, which become NaNs in the above np.array call
        known_ra_err [~np.isfinite(known_ra_err) ] = 0.1
        known_dec_err[~np.isfinite(known_dec_err)] = 0.1
        known_dm_err[~np.isfinite(known_ra_err)] = 1.0

        for event in event_group.events:
            if event.is_rfi():
                continue
        
            # print('hi, np is', np)
            # print('np.hypot is', np.hypot)
            # print('event.dm', event.dm)
            # print('known_dm', known_dm)
            # print('event.dm_error', event.dm_error)
            # print('known_dm_err', known_dm_err)
            
            dm_sigmas = (np.abs(event.dm - known_dm) /
                         np.hypot(event.dm_error, known_dm_err))

            dec_sigmas = (np.abs(event.dec - known_dec) /
                          np.hypot(event.dec_err, known_dec_err))

            # NOTE - both the event and known-source RA errors should be
            # isotropic (ie, include cos(Dec) terms)
        
            ra_sigmas = (np.abs(ra_difference(event.ra, known_ra)) *
                         np.cos(np.deg2rad(known_dec)) /
                         np.hypot(event.ra_err, known_ra_err))

            sigmas = np.sqrt(dm_sigmas**2 + ra_sigmas**2 + dec_sigmas**2)
            i = np.argmin(sigmas)
            if sigmas[i] < self.threshold:
                print('Matched known source with RA,Dec sigmas %.1f, %.1f and DM sigma %.1f: %s' %
                      (ra_sigmas[i], dec_sigmas[i], dm_sigmas[i], self.ks_database[i]))
                event.known_source_id = self.ks_database[i].id
            else:
                print('Nearest known source: sigmas %.1f - RA,Dec sigmas %.1f, %.1f and DM sigma %.1f' %
                      (sigmas[i], ra_sigmas[i], dec_sigmas[i], dm_sigmas[i]))

        return [event_group]

#KSS: event: {'chunk_utc': 1794951406.4418616,
#'events': [{'is_incoherent': False, 'is_fake': False, 'beam_id': 6, 'fpga_timestamp': 7189494,  'rfi_prob': 0.0, 'width_ms': 23.961599999999997, 'subband_freq_lo_MHz': 505.96442562694074, 'subband_freq_hi_MHz': 604.7431568147636, 'tree_index': 0, 'rfi_grade_level1': 10.0, 'chunk_fpga_start': 7188480, 'chunk_utc': 1794951406.4418616, 'timestamp_utc': 1794951406.4470534, 'pipeline_timestamp': 305367.083312891, 'pipeline_id': 50, 'best_beam_grid_x': 0.033333333333333326, 'best_beam_grid_y': -0.03333333333333334, 'best_snr': 18.432565689086914, 'beam_activity': 1, 'dm_activity': 1, 'beam_activity_lookback': [1, 1, 1, 1, 1, 1, 2, 1, 1, 1], 'dm_activity_lookback': [1, 1, 1, 1, 1, 1, 2, 1, 1, 1], 'avg_l1_grade': np.float64(10.0), 'n_live_beams': 16, 'l1_events': [{'is_incoherent': False, 'is_fake': False, 'beam_id': 6, 'fpga_timestamp': 7189494, 'dm': 437.4172366352279, 'snr': 18.432565689086914, 'rfi_prob': 0.0, 'width_ms': 23.961599999999997, 'subband_freq_lo_MHz': 505.96442562694074, 'subband_freq_hi_MHz': 604.7431568147636, 'tree_index': 0, 'dm_error': 0.1, 'rfi_grade_level1': 10.0, 'beam_grid_x': 0.033333333333333326, 'beam_grid_y': -0.03333333333333334, 'chunk_fpga_start': 7188480, 'chunk_utc': 1794951406.4418616, 'timestamp_utc': 1794951406.4470534, 'pipeline_timestamp': 305367.083312891, 'pipeline_id': 50, 'id': 267365}],
#
# 'event_id': 267366, 'rfi_grade_level2': np.float64(9.992892845271587), 'rfi_grade_metrics_level2': {'ML_Classifier_grade': np.float64(0.9992892845271587)},
# 'is_rfi': False, 'is_bright_pulsar': False,
#'ra': np.float64(351.29439193609556), 'dec': np.float64(47.5360653267289),
#'ra_err': np.float64(0.11191532304041558), 'dec_err': np.float64(0.16544004275539695)}]}
# 'dm': 437.4172366352279, 'dm_error': 0.1,



if __name__ == '__main__':
    from chord_frb_db.utils import get_db_engine
    engine = get_db_engine()
    known = _load_known_sources(engine)
    print('Known sources:', len(known), known[0])

