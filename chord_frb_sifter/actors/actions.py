from chord_frb_sifter.actors import Actor
from chord_frb_sifter.event import L1Event, L2Event

from chord_frb_db.utils import get_db_engine
from sqlalchemy.orm import Session

from queue import Queue
from threading import Thread

import numpy as np

#import concurrent.futures as cf

# Should the database stuff be in a separate actor?

class ActionPicker(Actor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.db_queue = Queue()
        self.db_thread = Thread(target=ActionPicker.run_db, args=(self,))
        self.db_thread.start()
        #self.db_executor = cf.ThreadPoolExecutor(max_workers=3)

    def shutdown(self):
        #self.db_executor.shutdown(wait=True)
        print('Shutting down ActionPicker... sending None on db queue')
        self.db_queue.put(None)
        print('Joining db thread')
        self.db_thread.join()
        print('done shutdown of ActionPicker')

    # There's no real reason this needs to be a class method...
    def run_db(self):
        print('Starting database interaction thread.')
        database_engine = get_db_engine()
        with Session(database_engine) as session:
            while True:
                print('Waiting for event from db queue.  Approx size: %i' % self.db_queue.qsize())
                event = self.db_queue.get()
                if event is None:
                    break
                print('Sending event to db:', event)
                self.save_event_to_db(session, event)
                session.flush()
                session.commit()
                print('Committing db')
        print('Database thread ending')

    def save_event_to_db(self, session, event):
        from chord_frb_db.models import EventBeam, Event
        l1_name_map = {
            'beam': True,
            'beam_no': 'beam',
            'snr': True,
            'timestamp_fpga': True,
            'timestamp_utc': True,
            'time_error': True,
            'tree_index': True,
            'rfi_grade_level1': 'rfi_grade',
            'rfi_mask_fraction': True,
            'rfi_clip_fraction': True,
            'dm': True,
            'dm_error': True,
            'pos_ra_deg': 'ra',
            'pos_ra_error_deg': 'ra_error',
            'pos_dec_deg': 'dec',
            'pos_dec_error_deg': 'dec_error',
        }
        l2_name_map = {
            'timestamp_utc': 'timestamp',
            'combined_snr': 'total_snr',
            'dm': True,
            'dm_error': True,
            'ra': True,
            'dec': True,
            'is_rfi': True,
            'is_frb': True,
            'pos_ra_deg': 'ra',
            'pos_error_semimajor_deg_68': 'ra_error',
            'pos_dec_deg': 'dec',
            'pos_error_semiminor_deg_68': 'dec_error',
            'dm_gal_ne_2001_max': 'dm_ne2001',
            'dm_gal_ymw_2016_max': 'dm_ymw2016',
            'spectral_index': True,
            'pulse_width_ms': 'pulse_width',
            'rfi_grade_level2': 'rfi_grade',
            'beam_activity': True,
        }
        l2_db_args = { 'is_rfi': event.is_rfi(),
                       'is_known_pulsar': False,
                       'is_new_burst': False,
                       'is_frb': event.is_frb(),
                       'is_repeating_frb': False,
                       'scattering': 0.,
                       'fluence': 0.,
        }
        l1_objs = []
        payload = event.database_payload()
        for k,v in payload.items():
            # skip...
            if k in ['dead_beam_nos']:
                continue
            if k == 'l1_events':
                # if we convert the L1 events to a numpy recarray (dustin doesn't want to do this)
                # we need to pull values back out into list/dict for db interface.
                n = v.size
                l1list = [{} for i in range(n)]
                for col in v.dtype.names:
                    vals = v[col]
                    for i,val in enumerate(vals):
                        l1list[i][col] = val
                v = l1list

                for l1 in v:
                    # Add L1 event to database, save its key
                    l1_db_args = {}

                    for l1key,l1val in l1.items():
                        if l1key == 'timestamp_utc':
                            #l1val = l1val.timestamp()
                            # microsec -> sec
                            l1val *= 1e-6
                            pass
                        l1val = to_db_type(l1val)

                        k2 = l1_name_map.get(l1key, None)
                        if k2 is not None:
                            # same key name
                            if k2 is True:
                                k2 = l1key
                            l1_db_args[k2] = l1val

                    ## FIXME -- fake up some required fields!
                    for key in ['time_error', 'dm_error', 'ra', 'dec', 'ra_error', 'dec_error']:
                        if not key in l1_db_args:
                            l1_db_args[key] = 0.

                    l1_db_obj = EventBeam(**l1_db_args)
                    #print('Created L1 db object:', l1_db_obj)
                    session.add(l1_db_obj)
                    session.flush()
                    # Now we know the L1 event's unique id
                    assert(l1_db_obj.id is not None)
                    l1_objs.append(l1_db_obj)
                continue
            v = to_db_type(v)

            if k == 'timestamp_utc':
                #print('timestamp_utc:', v)
                # microseconds -> seconds
                v *= 1e-6
                #v = v.timestamp()
            k2 = l2_name_map.get(k, None)
            if k2 is not None:
                # same key name
                if k2 is True:
                    k2 = k
                l2_db_args[k2] = v
            else:
                print('Ignoring L2 key:', k, '=', v)

            if k == 'known_source_name':
                # FIXME .... connect to known source db id / name -- but we should probably just do
                # that in the known source actor!
                if v != "":
                    print('Known source!')
                    print('val: "%s"' % v)
            if k == 'flux_mjy':
                # milli -> Jansky
                l2_db_args['flux'] = 0.001 * v

        l2_db_args['nbeams'] = len(l1_objs)
        # print('L2 DB args:')
        # for k,v in l2_db_args.items():
        #     print('  ', k, ':', v)
        l2_db_obj = Event(**l2_db_args)
        # Add L2 event to db
        session.add(l2_db_obj)
        # Now we can associate the L1 events with the L2 event.
        for e in l1_objs:
            l2_db_obj.beams.append(e)

    def _perform_action(self, event):
        # Log everything in db?
        self.save_to_db(event)

        # if event.is_rfi():
        #     return event
        # 
        # if event.is_frb():
        #     pass
        return event

    def save_to_db(self, event):
        # FIXME -- put_nowait ? queue size? timeout?
        print('Saving event to db:', event)
        self.db_queue.put(event)
    
def to_db_type(v):
    # convert to normal python types for database interaction
    if isinstance(v, (np.float32, np.float64)):
        v = float(v)
    if isinstance(v, (np.uint64, np.uint16, np.uint8)):
        v = int(v)
    return v
