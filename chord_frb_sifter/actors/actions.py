from chord_frb_sifter.actors import Actor
from chord_frb_sifter.event import L1Event, L2Event

from chord_frb_db.utils import get_db_engine
from chord_frb_db.models import EventBeam, Event, KnownSource
from sqlalchemy.orm import Session
from sqlalchemy import select

from queue import Queue
from threading import Thread

import numpy as np

import concurrent.futures as cf

# Should the database stuff be in a separate actor?

# Returns dispersion delay in seconds
# The 'DM' parameter is the dispersion measure in its standard units (pc cm^{-3})
# freq is in MHz.
# This was lifted from https://github.com/kmsmith137/simpulse/blob/master/simpulse.hpp#L32
def dispersion_delay(dm, freq_MHz):
    return 4.148806e3 * dm / (freq_MHz * freq_MHz)

class ActionPicker(Actor):

    '''The ActionPicker decides which actions to take based on
    information that other actors have provided about events.

    We run actions that can block in their own threads:

    * database actions
    * intensity callbacks to pirate

    Database calls are sent to the db thread via a thread-safe db_queue.

    Intensity calls are done via concurrent.futures.  These could also
    happen via a queue + thread, but futures allows several calls to
    run in parallel.

    '''
    def __init__(self, database_engine=None, sifter=None, **kwargs):
        super().__init__(**kwargs)
        self.sifter = sifter
        self.db_queue = Queue()
        self.db_thread = Thread(target=ActionPicker.run_db, args=(self, database_engine))
        self.db_thread.start()
        self.intensity_executor = cf.ThreadPoolExecutor(max_workers=10)
        self.intensity_futures = []

    def __str__(self):
        return 'ActionPicker'

    def shutdown(self):
        print('Shutting down ActionPicker... sending None on db queue')
        self.db_queue.put(None)
        print('Joining db thread')
        self.db_thread.join()
        print('Shutting down intensity callback thread pool...')
        self.intensity_executor.shutdown(wait=True)
        print('Shutdown of intensity callback thread pool finished')
        print('done shutdown of ActionPicker')

    def _perform_action(self, event_group):
        for event in event_group.events:
            # Log everything in db?  Maybe we should omit is_rfi events!
            self.save_to_db(event)

            # Intensity callback
            if event.is_frb() or event.is_new_burst():
                print('FRB or Ambiguous event -- sending intensity callback!')
                self.send_intensity_callback(event)
        return [event_group]

    def save_to_db(self, event):
        self.db_queue.put(('event', event))

    def send_intensity_callback(self, event):
        future = self.intensity_executor.submit(ActionPicker.intensity_callback, self, event)
        try:
            e_immediate = future.exception(timeout=0)
            if e_immediate is not None:
                raise e_immediate
        except TimeoutError:
            pass
        self.intensity_futures.append(future)

    # There's no real reason this needs to be a class method...
    # This is the database interaction thread.
    def run_db(self, database_engine):
        from chord_frb_db.models import IntensityFile
        import sqlalchemy as sa
        print('Starting database interaction thread.')
        with Session(database_engine) as session:
            while True:
                try:
                    print('Waiting for event from db queue.  Approx size: %i' % self.db_queue.qsize())
                    req = self.db_queue.get()
                    if req is None:
                        break
                    (req_type, arg) = req
                    if req_type == 'event':
                        event = arg
                        self.save_event_to_db(session, event)
                        session.flush()
                        session.commit()

                except Exception as e:
                    print('Database thread hit exception:', e)
                    import traceback
                    traceback.print_exc()
                    raise e
        print('Database thread ending')

    def save_event_to_db(self, session, event):
        # Save L1 events
        l1_events = event.get('l1_events', [])
        l1_db_objs = []
        l1_payload = [e.database_payload() for e in l1_events]
        for args in l1_payload:
            #print('L1 event db from:', args)
            db_obj = EventBeam(**args)
            session.add(db_obj)
            session.flush()
            # Now we know the L1 event's unique id
            # (actually, didn't we set the event_id in the event_id_stamper??)
            assert(db_obj.id is not None)
            l1_db_objs.append(db_obj)

        # Save L2 event
        l2_payload = event.database_payload()
        l2_db_obj = Event(**l2_payload)

        # Resolve known_source_name → known_id
        known_name = event.get('known_source_name', '')
        if known_name:
            ks = session.scalars(
                select(KnownSource).where(KnownSource.name == known_name).limit(1)
            ).first()
            if ks is not None:
                l2_db_obj.known_id = ks.id

        session.add(l2_db_obj)

        # Now we can associate the L1 events with the L2 event.
        for e in l1_db_objs:
            l2_db_obj.beams.append(e)
        # DEBUG
        session.flush()
        #print('Saved L2 event id', l2_db_obj.event_id)

    def intensity_callback(self, event):
        print('Intensity callback: event id %i' % event.event_id)
        # gather all beams that triggered for this grouped event.
        # FIXME -- add adjacent beams?
        beams = set([event.beam_id])
        for e in event.l1_events:
            beams.add(e['beam_id'])
        beams = list(beams)
        #print('All triggered beams:', beams)

        from chord_frb_grpc.frb_search_pb2 import WriteFilesRequest, SubscribeFilesRequest

        # Sort the beams into their beamsets -- since each pirate node handles a beamset,
        # we need to know which beams to ask for each pirate to dump.
        beamset_beams = {}
        for beam in beams:
            beamset = self.sifter.beam_id_to_beamset[beam]
            if not beamset in beamset_beams:
                beamset_beams[beamset] = []
            beamset_beams[beamset].append(beam)

        # Common parameters for the intensity callbacks...
        ns_per_fpga = self.sifter.nanosec_per_fpga_seq()
        fpga_buffer = int(1e9 / ns_per_fpga)
        # the event's FPGA_TIMESTAMP is the arrival time at the bottom of the frequency band.
        fpga_end = event.fpga_timestamp + fpga_buffer
        freq_lo, freq_hi = self.sifter.get_freq_range()
        dt_lo = dispersion_delay(event.dm, freq_lo)
        dt_hi = dispersion_delay(event.dm, freq_hi)
        #print('DM %.1f: delay at %f MHz is %.3f sec' % (event.dm, freq_lo, dt_lo))
        #print('DM %.1f: delay at %f MHz is %.3f sec' % (event.dm, freq_hi, dt_hi))
        dt = dt_lo - dt_hi
        assert(dt > 0)
        dfpga = int(1e9 * dt / ns_per_fpga)
        fpga_start = event.fpga_timestamp - dfpga - fpga_buffer
        #print('FPGA range requested: %i, %i' % (fpga_start, fpga_end))

        for beamset,beams in beamset_beams.items():
            (addr, stub) = self.sifter.beamset_pirate_rpc[beamset]
            #print('Sending intensity callback to pirate RPC address:', addr, 'using stub', stub)
            acqdir = 'event-%08i' % event.event_id
            req = WriteFilesRequest(protocol_version = 2,
                                    beams = beams,
                                    fpga_seq_start = fpga_start,
                                    fpga_seq_end = fpga_end,
                                    acqdir = acqdir)
            #print('Sending WriteFilesRequest:', req)
            resp = stub.WriteFiles(req)
            #print('Got WriteFiles response:', resp)

            # The Sifter has a queue for handling pirate write request database updates
            for fn in resp.filename_list:
                #print('Pirate will write file %s' % fn)
                self.sifter.file_update_queue.put((event.event_id, fn, None))
