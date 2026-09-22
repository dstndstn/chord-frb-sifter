import time
from chord_frb_grpc import frb_sifter_pb2_grpc
from chord_frb_grpc.frb_sifter_pb2 import ConfigReply, FrbEventsReply
import queue
from chord_frb_sifter.event import L1Event, EventGroup
from chord_frb_sifter.pipeline import setup, simple_create_pipeline
from chord_frb_sifter.pipeline import simple_process_events

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from datetime import datetime

from sqlalchemy.orm import Session
from chord_frb_db.models import PirateConfig, BeamSNR

'''
This script opens a gRPC server socket listening for connections from Pirate.
It loads the pipeline and sends any events it receives through the pipeline.
It interacts with a database and sends intensity callback requests to Pirate.
'''

class FrbSifter(frb_sifter_pb2_grpc.FrbSifterServicer):
    def __init__(self, injections):
        # SimpleQueue is thread-safe
        self.event_queue = queue.SimpleQueue()
        self.beam_snr_queue = queue.SimpleQueue()
        self.injections = injections
        self.xengine_config_yaml = None
        self.xengine_config = None
        self.pirate_config_yaml = None
        #self.dedisp_config = None
        #self.grouper_config = None

        self.subscribe_files_threads = []
        self.file_update_queue = queue.SimpleQueue()
        
        # beamset (int) -> arrays of (id, x, y)
        self.beamset_meta = {}

        # beamset (int) to (Pirate callback address, Pirate gRPC stub)
        self.beamset_pirate_rpc = {}

        # beam_id to (beam_x, beam_y)
        self.beam_id_to_xy = {}

        # beam_id to beamset
        self.beam_id_to_beamset = {}
        
        # is a gRPC peer address the fake x-engine??
        self.peer_is_fake = {}

    def fpga_to_utc_seconds(self, fpga):
        t0 = self.xengine_config['unix_ns_at_seq_0']
        dt = self.xengine_config['dt_ns_per_seq']
        utc = (t0 + dt * fpga) * 1e-9
        return utc

    def nanosec_per_fpga_seq(self):
        dt = self.xengine_config['dt_ns_per_seq']
        return dt

    # in MHz
    def get_freq_range(self):
        edges = self.xengine_config['zone_freq_edges']
        return (edges[0], edges[-1])

    # Called by beam_buffer
    def beamset_to_beam_ids(self, beamset):
        beam_ids,_,_ = self.beamset_meta[beamset]
        return beam_ids

    def print_yaml(self, y):
        # just indent...
        lines = y.split('\n')
        for line in lines:
            if len(line) > 100:
                line = line[:96] + ' ...'
            print('    ' + line)

    def CheckConfiguration(self, request, context):
        print('CheckConfiguration: context', context)
        print('  peer:', context.peer())
        xengine = request.xengine_yaml
        print('Received Xengine YAML config: len %i' % len(xengine))
        self.print_yaml(xengine)
        pirate = request.pirate_yaml
        print('Received Pirate YAML config: len %i' % len(pirate))
        self.print_yaml(pirate)
        dedisp = request.dedispersion_plan_yaml
        print('Received Pirate dedisperser YAML config: len %i' % len(dedisp))
        self.print_yaml(dedisp)
        grouper = request.grouper_yaml
        print('Received Pirate Grouper YAML config: len %i' % len(grouper))
        self.print_yaml(grouper)
        pirate_rpc = request.search_ip_addr
        print('Received Pirate RPC address: "%s"' % pirate_rpc)

        ok = True
        if not self.check_configs(context.peer(), xengine, pirate, dedisp, grouper, pirate_rpc):
            print('Failed YAML config check')
            ok = False
        r = ConfigReply(ok=ok)
        return r

    def check_configs(self, peer, xengine_yaml, pirate_yaml, dedisp_yaml, grouper_yaml, pirate_rpc):
        xengine = yaml.load(xengine_yaml, Loader=Loader)
        beamset = xengine['beamset']
        new_meta = (xengine['beam_ids'],
                    xengine['beam_positions_x'],
                    xengine['beam_positions_y'])
        if beamset in self.beamset_meta:
            print('Already received beamset %i - checking consistency' % beamset)
            if new_meta != self.beamset_meta[beamset]:
                print('Metadata for beamset %i not equal:\n%s\nvs\n%s' %
                      (beamset, new_meta, self.beamset_meta[beamset]))
                return False
        else:
            self.beamset_meta[beamset] = new_meta

        # Fake x-engine sending us injected events?  Don't validate.
        is_fake = ((len(xengine_yaml) > 0) and
                   (len(pirate_yaml) == 0) and
                   (len(dedisp_yaml) == 0) and
                   (len(grouper_yaml) == 0) and
                   (len(pirate_rpc) == 0))
        if is_fake:
            print('Only x-engine yaml is non-empty -- assuming fake x-engine message')

        assert(peer not in self.peer_is_fake)
        self.peer_is_fake[peer] = is_fake

        if not is_fake:
            # Populate beam info: beam x,y
            for beam_id, beam_x, beam_y in zip(xengine['beam_ids'],
                                               xengine['beam_positions_x'],
                                               xengine['beam_positions_y']):
                new_val = (beam_x, beam_y)
                if beam_id in self.beam_id_to_xy:
                    old_val = self.beam_id_to_xy[beam_id]
                    if new_val != old_val:
                        print('Already received beam_id %i with conflicting value: old %s, new %s' %
                              (beam_id, old_val, new_val))
                        return False
                self.beam_id_to_xy[beam_id] = new_val

                self.beam_id_to_beamset[beam_id] = beamset

            # delta-RA,delta-Dec of beam positions relative to boresight.

                
        if not is_fake:
            import grpc
            from chord_frb_grpc.frb_search_pb2_grpc import FrbSearchStub
            from chord_frb_grpc.frb_search_pb2 import GetStatusRequest, SubscribeFilesRequest
            # Check that we can call back to Pirate...
            print('Pirate RPC address:', pirate_rpc)
            # testing...
            if len(pirate_rpc):
                # Open connection...
                ch1 = grpc.insecure_channel(pirate_rpc)
                pirate = FrbSearchStub(ch1)
                # Make a pirate Status call
                req = GetStatusRequest(protocol_version=2)
                print('Pirate Status RPC request:', req)
                resp = pirate.GetStatus(req)
                print('Got pirate RPC response:', resp)
                self.beamset_pirate_rpc[beamset] = (pirate_rpc, pirate)
                # Subscribe to file updates!
                req = SubscribeFilesRequest(protocol_version=2,
                                            subscribe_streams=False)
                print('Subscribing to file updates:', req)
                resp = pirate.SubscribeFiles(req)
                print('Subscribed to file updates:', resp)
                for val in resp:
                    print('First subscribe-files response:', val)
                    # check that it's a ReadySentinal
                    ready = val.ready
                    print('Ready:', ready)
                    break
                print('Response is a', type(resp))
                print('Response is', resp)
                # Create a thread to continue reading updates from this pirate?
                import threading
                t = threading.Thread(target=file_update_rpc_reader,
                                     args=(resp, self.file_update_queue, beamset),
                                     name='file-update-handler-beamset-%i' % beamset)
                self.subscribe_files_threads.append(t)
                t.start()
                
        # save parsed and yaml xengine config
        self.xengine_config = xengine
        self.xengine_config_yaml = xengine_yaml
        if not is_fake:
            self.pirate_config_yaml = pirate_yaml
            # dedisp? grouper?

        return True

    def FrbEvents(self, request, context):
        # if request.has_injections != self.injections:
        #     print('Received FRB Events %s injections, but this FRB Sifter is%s handling injections!' % ('with' if request.has_injections else 'without', '' if self.injections else ' not'))
        #     return FrbEventsReply(ok=False, message='Expected has_injections=%s, got %s - are you sending to the wrong FRB Sifter (injection vs prod)?' % (self.injections, request.has_injections))
        msg = ''
        ok = True

        # print(('Got FRB Events grpc: beam-set %i, chunk FPGA %i to %i, ' +
        #        'coarse-grain SNR array: %i beams; ' +
        #        'events: %i, ' +
        #        'peer: %s, is_fake? %s') %
        #       (request.beam_set_id, request.chunk_fpga_start, request.chunk_fpga_end,
        #        len(request.coarsegrain_snr),
        #        len(request.events), 
        #        context.peer(),
        #        self.peer_is_fake.get(context.peer(), 'unknown')))
        #for e in request.events:
        #    print('  event', type(e), e)

        is_fake = self.peer_is_fake.get(context.peer())

        # Convert grpc FrbEvent objects into simple dicts.
        event_list = []
        for e in request.events:
            # FIXME -- we should use introspection to get the keys defined in frb_sifter.proto....
            event = L1Event(is_incoherent=False,
                            is_fake=is_fake)
            print('GRPC event: beam %i, FPGA %i, DM %.1f, SNR %.1f' %
                  (e.beam_id, e.fpga_timestamp, e.dm, e.snr))
            for key in ['beam_id', 'fpga_timestamp', 'dm', 'snr', 'rfi_prob',
                        'width_ms', 'subband_freq_lo_MHz', 'subband_freq_hi_MHz',
                        'tree_index']:
                event[key] = getattr(e, key)
            # HACK
            event.dm_error = 0.1

            # yuck
            # CHIME/FRB's rfi_grade_level2
            # values are 0 to 10, with RFI:0 and Astrophysical:10.
            event['rfi_grade_level1'] = 10. * (1. - event['rfi_prob'])
            #print('Created L1Event:', event)
            event_list.append(event)

        # Send event list even if empty - BeamBuffer needs to know it has heard from all beamsets.
        event_group = EventGroup(is_fake=is_fake,
                                 chunk_fpga_start=request.chunk_fpga_start,
                                 beamset=request.beam_set_id,
                                 events=event_list)
        self.event_queue.put(event_group)

        if not is_fake and len(request.coarsegrain_snr):
            self.beam_snr_queue.put(dict(beamset=request.beam_set_id,
                                         fpga_start=request.chunk_fpga_start,
                                         fpga_end=request.chunk_fpga_end,
                                         beam_snr=request.coarsegrain_snr,
                                         peer=context.peer()))

        return FrbEventsReply(ok=ok, message=msg)

# One thread per pirate, just to receive the streaming file-update RPCs.
def file_update_rpc_reader(stream, file_update_queue, beamset):
    print('file_update_handler starting for beamset', beamset)
    try:
        #print('stream:', stream)
        for val in stream:
            #print('Got file update for beamset', beamset)
            #print('file update:', val)
            notif = val.notification
            #print('notification:', notif)
            err = notif.error_message
            if err:
                print('filename: %s failed: error message: %s' % (notif.filename, err))
            else:
                print('filename %s written sucessfully' % notif.filename)
            # eg, filename event-00003959/frame_b8_t116.asdf written sucessfully
            file_update_queue.put((None, notif.filename, notif.error_message))
    except Exception as e:
        print('Exception in thread receiving file-writing updated from pirate (beamset %i): %s' % (beamset, e))
        print(e)
        import traceback
        traceback.print_exc(e)
        # FIXME -- fail more loudly?
        raise

def file_update_db_handler(file_update_queue, database):
    import sqlalchemy as sa
    from sqlalchemy.orm import Session
    from chord_frb_db.models import IntensityFile, Event

    with Session(database) as session:
        while True:
            fup = file_update_queue.get()
            (event_id, filename, error_message) = fup
            #print('Sending file update to db: event %s, filename %s, err %s' %
            #      (event_id, filename, error_message))

            # We either get
            # (event_id, filename, None) --> pirate told us it's going to write this file
            # or
            # (None, filename, error_message) --> pirate updated us about this file
            #

            if event_id is not None:
                # Check that this event_id exists in the database... we can get a race condition
                # between this thread and the event-inserting database thread!
                event = session.execute(sa.select(Event).where(Event.event_id==event_id)).scalar_one_or_none()
                if event is None:
                    print('File update: event id %i does not exist in the database yet' % event_id)
                    # re-queue for later processing
                    # FIXME -- sleep for a bit of queue is empty?
                    file_update_queue.put(fup)
                    continue

            # create if it doesn't exist, update its status if it does
            ifile = session.execute(sa.select(IntensityFile).where(
                IntensityFile.filename==filename)).scalar_one_or_none()
            print('IntensityFile object with that filename:', ifile)
            if ifile is not None:
                print('Got IntensityFile:', ifile)
                if event_id is not None and event_id != ifile.event_id:
                    ifile.event_id = event_id
                if error_message is not None:
                    ok = (len(error_message) == 0)
                    ifile.succeeded = ok
                    ifile.failed = not(ok)
                    ifile.error_message = error_message
                print('Updated IntensityFile status')
            else:
                kwargs = {}
                if event_id is not None:
                    kwargs.update(event_id=event_id)
                if error_message is not None:
                    ok = (len(error_message) == 0)
                    kwargs.update(succeeded=ok,
                                  failed=not(ok),
                                  error_message=error_message)
                ifile = IntensityFile(filename=filename,
                                      **kwargs)
                print('Adding IntensityFile to db')
                session.add(ifile)
            session.flush()
            session.commit()
            
def serve(sifter, port=10000, max_threads=10):
    import grpc
    from concurrent import futures
    #from chord_frb_grpc import frb_sifter_pb2_grpc

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_threads))

    frb_sifter_pb2_grpc.add_FrbSifterServicer_to_server(sifter, server)

    server.add_insecure_port('[::]:' + str(port))
    print('Server started, listening on', port)
    server.start()
    return server

def event_handler(sifter, event_queue, pipeline):
    # beam id to xy position
    beam_id_to_xy = sifter.beam_id_to_xy
    
    while True:
        event_group = event_queue.get()
        if event_group.is_fake:
            if len(event_group.events):
                print('Ignoring', len(event_group.events), 'fake events')
            continue
        chunk_utc = sifter.fpga_to_utc_seconds(event_group.chunk_fpga_start)
        event_group.chunk_utc = chunk_utc

        later = False
        for e in event_group.events:
            # add beam metadata

            # ... if beam metadata is not yet known (the config
            # sending seems to actually get sent in a separate thread
            # from the event sending....), re-queue this event_group for
            # processing later.
            beam_id = e['beam_id']
            if not beam_id in beam_id_to_xy:
                print('Nothing is known about beam %i yet... saving this event group for reprocessing'
                      % beam_id)
                later = True
                break

            x,y = beam_id_to_xy[e['beam_id']]
            e['beam_grid_x'] = x
            e['beam_grid_y'] = y

            # Add values from the event-group
            for key in ['chunk_fpga_start']:
                e[key] = event_group[key]
            # convert FPGA to UTC seconds...
            e['chunk_utc'] = chunk_utc
            e['timestamp_utc'] = sifter.fpga_to_utc_seconds(e.fpga_timestamp)

        if later:
            if event_queue.empty():
                time.sleep(1.)
            print('Putting event group back on the queue for later processing')
            event_queue.put(event_group)
            continue

        simple_process_events(pipeline, event_group)

def beam_snr_handler(sifter, beam_snr_queue, database):
    known_beamsets = set()
    pirate_configs = {}

    with Session(database) as session:
        while True:
            beam_snr = beam_snr_queue.get()
            #print('beam_snr_handler: got', len(beam_snr))
            #beamset, fpga_start, fpga_end, snr_array = beam_snr
            beamset = beam_snr['beamset']
            fpga_start = beam_snr['fpga_start']
            fpga_end = beam_snr['fpga_end']
            peer = beam_snr['peer']
            snr_array = beam_snr['beam_snr']
            if beamset not in known_beamsets:
                # Add a database entry describing this beamset: beam ids, x,y locations,
                # peer?, unix_nano0, date of first message?
                print('Looking up beamset', beamset, 'in xengine YAML...')
                print(sifter.xengine_config)

                if not beamset in sifter.beamset_meta:
                    print('Unknown beamset', beamset)
                    continue

                # arrays
                beam_id, beam_x, beam_y = sifter.beamset_meta[beamset]

                # Add beamset, beam_ids, beam_x, beam_y to the db!
                pc = PirateConfig(beamset=beamset,
                                  start_time=datetime.now(),   # ???
                                  xengine_config=sifter.xengine_config_yaml,
                                  pirate_config=sifter.pirate_config_yaml,
                                  beam_x=beam_x,
                                  beam_y=beam_y,
                                  beam_id=beam_id)
                try:
                    #print('Saving Pirate Config to database:', pc)
                    session.add(pc)
                    session.flush()
                    print('Saved PirateConfig to database: id', pc.id)
                    session.commit()
                    pirate_configs[beamset] = pc.id
                except Exception as e:
                    import traceback
                    print('Failed to insert PirateConfig to database;', e)
                    traceback.print_exc()
                    raise
                known_beamsets.add(beamset)
    
                # timing ... this only needs to get initialized once!
                xengine = sifter.xengine_config
                seq_per_frb_time_sample = xengine['seq_per_frb_time_sample']
                fpga0_nano = xengine['unix_ns_at_seq_0']
                nano_per_fpga = xengine['dt_ns_per_seq']
                #print('FPGA seq per time sample:', seq_per_frb_time_sample)
                #print('nanoseconds per FPGA seq:', nano_per_fpga)
    
            unix_time_nano_start = fpga0_nano + fpga_start * nano_per_fpga
            unix_time_nano_end   = fpga0_nano + fpga_end   * nano_per_fpga
            nano = 1_000_000_000
            date_start = datetime.fromtimestamp(unix_time_nano_start // nano +
                                                1e-9 * (unix_time_nano_start % nano))
            date_end = datetime.fromtimestamp(unix_time_nano_end // nano +
                                              1e-9 * (unix_time_nano_end % nano))
    
            # Add beamset, date_start, date_end, snr_array to the db!
            bs = BeamSNR(pirate_config_id=pirate_configs[beamset],
                         timestamp=date_start,
                         beam_snr=snr_array)
            try:
                #print('Saving BeamSNR to db:', bs)
                session.add(bs)
                session.flush()
                #print('Saved BeamSNR to database: id', bs.id)
                session.commit()
            except Exception as e:
                import traceback
                print('Error saving BeamSNR to database:', e)
                raise

'''
Example xengine metadata:

beamset: 0
beam_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
beam_positions_x: [-0.10000000000000001, -0.03333333333333334, 0.033333333333333326, 0.10000000000000001, -0.10000000000000001, -0.03333333333333334, 0.033333333333333326, 0.10000000000000001, -0.10000000000000001, -0.03333333333333334, 0.033333333333333326, 0.10000000000000001, -0.10000000000000001, -0.03333333333333334, 0.033333333333333326, 0.10000000000000001]
beam_positions_y: [-0.10000000000000001, -0.10000000000000001, -0.10000000000000001, -0.10000000000000001, -0.03333333333333334, -0.03333333333333334, -0.03333333333333334, -0.03333333333333334, 0.033333333333333326, 0.033333333333333326, 0.033333333333333326, 0.033333333333333326, 0.10000000000000001, 0.10000000000000001, 0.10000000000000001, 0.10000000000000001]
unix_ns_at_seq_0: 1772483060000000000
dt_ns_per_seq: 5120
seq_per_frb_time_sample: 195
tel_origin_itrs_lat_deg: 49.320751444439999
tel_origin_itrs_lon_deg: -119.62081125
tel_grid_x_axis: [0.99997434239835936, -3.7539331442771999e-05, -0.0071633187676754936]
tel_grid_y_axis: [6.5403387739209999e-05, 0.99999243322034881, 0.0038896303735576139]
tel_dish_elev_axis: [0.99999999838132392, -5.6897733584326999e-05, 0]
tel_dish_vert_axis: [0, 0, 1]
tel_dish_coelev_deg: 0
tel_dish_separation_x_m: 6.300156854906823
tel_dish_separation_y_m: 8.5000578097963082
'''

def main():
    import threading
    from chord_frb_db.utils import get_db_engine
    #from chord_frb_sifter.pipeline import setup, simple_create_pipeline
    #from chord_frb_sifter.pipeline import simple_process_events

    is_injections = False
    sifter = FrbSifter(is_injections)

    database_engine = get_db_engine()

    # Load pipeline config file
    setup()

    pipeline = simple_create_pipeline(database_engine, sifter=sifter)

    event_thread = threading.Thread(target=event_handler,
                                    args=(sifter, sifter.event_queue, pipeline),
                                    name='event-handler')
    event_thread.start()

    beam_snr_thread = threading.Thread(target=beam_snr_handler,
                                       args=(sifter, sifter.beam_snr_queue, database_engine),
                                       name='beam-snr-handler')
    beam_snr_thread.start()

    file_update_db_thread = threading.Thread(target=file_update_db_handler,
                                       args=(sifter.file_update_queue, database_engine),
                                       name='file-update-db-handler')
    file_update_db_thread.start()

    if False:
        fake_xengine_conf = {'version': 2, 'zone_nfreq': [640], 'zone_freq_edges': [400, 800], 'beamset': 0, 'beam_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'beam_positions_x': [-0.1, -0.03333333333333334, 0.033333333333333326, 0.09999999999999999, -0.1, -0.03333333333333334, 0.033333333333333326, 0.09999999999999999, -0.1, -0.03333333333333334, 0.033333333333333326, 0.09999999999999999, -0.1, -0.03333333333333334, 0.033333333333333326, 0.09999999999999999], 'beam_positions_y': [-0.1, -0.1, -0.1, -0.1, -0.03333333333333334, -0.03333333333333334, -0.03333333333333334, -0.03333333333333334, 0.033333333333333326, 0.033333333333333326, 0.033333333333333326, 0.033333333333333326, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], 'unix_ns_at_seq_0': 1786482526471023104, 'dt_ns_per_seq': 5120, 'seq_per_frb_time_sample': 195, 'tel_origin_itrs_lat_deg': 49.32075144444, 'tel_origin_itrs_lon_deg': -119.62081125, 'tel_grid_x_axis': [0.9999743423983594, -3.7539331442772e-05, -0.007163318767675494], 'tel_grid_y_axis': [6.540338773921e-05, 0.9999924332203488, 0.003889630373557614], 'tel_dish_elev_axis': [0.9999999983813239, -5.6897733584327e-05, 0], 'tel_dish_vert_axis': [0, 0, 1], 'tel_dish_coelev_deg': 0, 'tel_dish_separation_x_m': 6.300156854906823, 'tel_dish_separation_y_m': 8.500057809796308, 'noise_variance': [1]}
        xe_yaml = yaml.dump(fake_xengine_conf)
        sifter.check_configs(None, xe_yaml, 'x: 4', '', '', '')
                         
        g = EventGroup(**{'is_fake': False, 'chunk_fpga_start': 5890560, 'beamset': 0,
                          'events': [], 'chunk_utc': 1786482556.6306903})
        sifter.event_queue.put(g)

        g = EventGroup(**{'is_fake': False, 'chunk_fpga_start': 5940480, 'beamset': 0,
                          'events': [
                              L1Event(**{'beam_id': 4, 'fpga_timestamp': 5953740, 'dm': 1.437467336654663, 'snr': 29.343143463134766, 'rfi_prob': 0.0, 'width_ms': 0.9983999729156494, 'subband_freq_lo_MHz': 400.0, 'subband_freq_hi_MHz': 800.0, 'is_fake': False, 'rfi_grade_level1': 10.0, 'beam_grid_x': -0.1, 'beam_grid_y': -0.03333333333333334, 'chunk_fpga_start': 5940480, 'chunk_utc': 1786482556.8862808, 'timestamp_utc': 1786482556.9541721, 'is_incoherent': False, 'tree_index': 0, 'dm_error': 0.1}),
                              L1Event(**{'beam_id': 8, 'fpga_timestamp': 5989620, 'dm': 25.25835418701172, 'snr': 25.013864517211914, 'rfi_prob': 0.0, 'width_ms': 0.9983999729156494, 'subband_freq_lo_MHz': 400.0, 'subband_freq_hi_MHz': 800.0, 'is_fake': False, 'rfi_grade_level1': 10.0, 'beam_grid_x': -0.1, 'beam_grid_y': 0.033333333333333326, 'chunk_fpga_start': 5940480, 'chunk_utc': 1786482556.8862808, 'timestamp_utc': 1786482557.1378777, 'is_incoherent': False, 'tree_index': 0, 'dm_error': 0.1}),
                          ], 'chunk_utc': 1786482556.8862808})
        sifter.event_queue.put(g)

        import time
        time.sleep(15)
        return

    server = serve(sifter)
    server.wait_for_termination()


if __name__ == '__main__':
    import logging
    logging.basicConfig()
    main()
