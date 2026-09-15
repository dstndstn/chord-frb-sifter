"""
Definition of L1 and L2 event classes.

Both L1Event and L2Event are a dictionary with the ability to
manipulate dictionary items as class attributes.

We also define an EventGroup, which is basically just a wrapper for a
list of events plus properties that belong to the group.

"""
import numpy as np
from chord_frb_sifter import config

def simulate_l2_event():
    """Returns a minimal L2Event for smoke-testing pipeline actors."""

    snrs = [10.0, 8.0, 6.0]
    dm = 100.0
    tree_index = 2
    beam_ids = [0, 1, 2]

    # snr_vs_dm needs nonzero values so RFI feature extraction doesn't get an empty array
    #fake_l1['snr_vs_dm'] = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 9.5, 10.0, 9.5, 9.0,
    #                                  8.0, 7.0, 6.0, 5.0, 4.5, 4.0, 3.5, 3.0])

    fake_l1 = []
    for snr,beam_id in zip(snrs, beam_ids):
        fake_l1.append(L1Event(snr=snr, beam_id=beam_id, dm=dm, tree_index=tree_index,
                               is_incoherent=False,
                               rfi_grade_level1=0))

    return L2Event(dm=100.,
                   timestamp_utc=0,
                   beam_activite=10,
                   dead_beams=[],
                   l1_events=fake_l1,
                   n_live_beams=10,
                   beam_activity=3)

class AttribDict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    _reserved = set(dir(dict))
    
    def __getattr__(self, name):
        if name in self._reserved:
            return super().__getattribute__(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError('no key %s in class %s' % (name, type(self)))

    def __setattr__(self, name, value):
        if name.startswith("_") or name in self._reserved:
            raise AttributeError(f"'{name}' is reserved")
        self[name] = value

class EventGroup(AttribDict):
    pass

class L1Event(AttribDict):

    def get_dm_error(self):
    
        ## should load from config...
        #nds = [1, 2, 4, 8, 16]
        #nups = 1
        #dm_coarse_graining_factor = [16, 8, 8, 8, 8]
        #time_coarse_graining_factor = [16, 8, 8, 8, 8]
        #tree_size = [ 32768, 32768, 32768, 32768, 16384 ]
        #dt_sample = 0.00098304
        #freq_lo_MHz = 400.0
        #freq_hi_MHz = 800.0
    
        itree = self["tree_index"][0]
        tree_dt = config.l1_config.dt_sample * config.l1_config.nds[itree] / config.l1_config.nups
    
        tree_max_dm = (
                (config.l1_config.tree_size[itree] - 1)
                * tree_dt
                / (4.148806e3 * (1.0 / config.l1_config.freq_lo_MHz**2 - 1.0 / config.l1_config.freq_hi_MHz**2))
                )
        
        return tree_max_dm * config.l1_config.dm_coarse_graining_factor[itree] / config.l1_config.tree_size[itree] / 2.0

    def get_time_error(self):

        itree = self["tree_index"][0]
        tree_dt = config.l1_config.dt_sample * config.l1_config.nds[itree] / config.l1_config.nups

        return tree_dt * config.l1_config.dm_coarse_graining_factor[itree] / 2.0

    
    '''
    Creates a dict to initialize a database object,
    chord_frb_db.models.EventBeam

    ... which is basically just this L1Event dict with same keys
    renamed and some values normalized
    '''
    def database_payload(self):
        # my name -> db name (or True if the name is the same)
        l1_name_map = {
            'id': True,
            'beam_id': True,
            'snr': True,
            'fpga_timestamp': 'timestamp_fpga',
            'timestamp_utc': True,
            #'time_error': True,
            'tree_index': True,
            'rfi_grade_level1': 'rfi_grade',
            #'rfi_mask_fraction': True,
            #'rfi_clip_fraction': True,
            'dm': True,
            'dm_error': True,
            #'ra': True,
            #'ra_error': True,
            #'dec': True,
            #'dec_error': True,
        }

        db_args = {}
        for key,val in self.items():
            val = to_db_type(val)
            k2 = l1_name_map.get(key, None)
            if k2 is not None:
                # same key name
                if k2 is True:
                    k2 = key
                db_args[k2] = val

        ## FIXME -- fake up some required fields!
        for key in ['time_error', 'rfi_mask_fraction', 'rfi_clip_fraction',
                    'ra','dec', 'ra_error', 'dec_error']:
            if not key in db_args:
                db_args[key] = 0.

        return db_args

class L2Event(AttribDict):
    def get_l1_events_array(self, key, dtype=None):
        import numpy as np
        return np.array([e[key] for e in self.l1_events], dtype=dtype)

    def is_rfi(self):
        return getattr(self, 'flag_rfi', False)
    def is_frb(self):
        return getattr(self, 'flag_frb', False)
    def is_known_pulsar(self):
        return getattr(self, 'flag_known_pulsar', False)
    def is_repeating_frb(self):
        return getattr(self, 'flag_repeating_frb', False)
    def is_new_burst(self):
        return getattr(self, 'flag_new_burst', False)
    def is_known_source(self):
        return self.is_known_pulsar() or self.is_repeating_frb()

    def set_rfi(self):
        self.flag_rfi = True
    def set_frb(self):
        self.flag_frb = True
    def set_known_pulsar(self):
        self.flag_known_pulsar = True
    def set_repeating_frb(self):
        self.flag_repeating_frb = True
    def set_new_burst(self):
        self.flag_new_burst = True

    '''
    Creates a dict to initialize a database object,
        chord_frb_db.models.Event
    '''
    def database_payload(self):
        # dict shallow copy
        l2_db_args = { 'is_rfi': self.is_rfi(),
                       'is_known_pulsar': self.is_known_pulsar(),
                       'is_new_burst': self.is_new_burst(),
                       'is_frb': self.is_frb(),
                       'is_repeating_frb': self.is_repeating_frb(),
                       'scattering': 0.,
                       'fluence': 0.,
        }
        l2_name_map = {
            'event_id': True,
            'timestamp_utc': 'timestamp',
            'combined_snr': 'total_snr',
            'best_snr': True,
            'dm': True,
            'dm_error': True,
            'ra': True,
            'dec': True,
            'is_frb': True,
            'pos_ra_deg': 'ra',
            'pos_error_semimajor_deg_68': 'ra_error',
            'pos_dec_deg': 'dec',
            'pos_error_semiminor_deg_68': 'dec_error',
            'dm_gal_ne_2025_max': 'dm_ne2025',
            'dm_gal_ymw_2016_max': 'dm_ymw2016',
            'spectral_index': True,
            'pulse_width_ms': 'pulse_width',
            'rfi_grade_level2': 'rfi_grade',
            'beam_activity': True,
            'flux_mjy': 'flux',
        }

        n_l1 = 0
        for k,v in self.items():
            # skip...
            if k in ['dead_beam_nos']:
                continue
            if k == 'l1_events':
                n_l1 = len(v)
                continue

            # FIXME
            if k == 'known_source_name':
                if v != "":
                    print('Known source!')
                    print('val: "%s"' % v)

            v = to_db_type(v)
            # if k == 'timestamp_utc':
            #     # microseconds -> seconds
            #     v *= 1e-6
            if k == 'flux_mjy':
                # milli -> Jansky
                v *= 0.001

            k2 = l2_name_map.get(k, None)
            if k2 is not None:
                # same key name
                if k2 is True:
                    k2 = k
                l2_db_args[k2] = v
            else:
                pass
        l2_db_args['nbeams'] = n_l1
        return l2_db_args

def to_db_type(v):
    # convert to normal python types for database interaction
    if isinstance(v, (np.float32, np.float64)):
        v = float(v)
    if isinstance(v, (np.uint64, np.uint16, np.uint8)):
        v = int(v)
    return v
