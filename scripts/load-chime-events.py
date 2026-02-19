import sys
import os
import time
import pickle
from collections import Counter
import numpy as np
import fitsio

from copy import deepcopy

from frb_common.events.l1_event.dtypes import L1_EVENT_DTYPE
import cfbm

from chord_frb_db.utils import get_db_engine

def to_db_type(v):
    # convert to normal python types for database interaction
    if isinstance(v, (np.float32, np.float64)):
        v = float(v)
    if isinstance(v, (np.uint64, np.uint16, np.uint8)):
        v = int(v)
    return v

# Save events into our CHORD database
def send_to_db(session, events):
    from chord_frb_db.models import EventBeam, Event

    # Not using from pipeline:
    # snr_scale (what is it)
    # spectral_index
    # scattering_measure
    # level1_nhits
    # snr_vs_dm
    # snr_vs_tree_index
    # snr_vs_spectral_index
    # is_incoherent
    # snr_vs_dm_x
    # snr_vs_tree_index_x
    # snr_vs_spectral_index_x
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

    # Not setting:
    # is_known
    # is_frb
    # best_beam
    # best_snr
    # pos_error_theta_deg_68 --> assert == 0?
    # known_source_name
    # known_source_rating -1
    # known_source_metrics {}
    # beam_sensitivity{,_min_95,_max_95}
    # spectral_index_error
    # flux_mjy_{min,max}
    # rfi_grade_level2
    # beam_activity
    # unknown_event_type
    # coh_dm_activity
    # avg_l1_grade

    # Ignoring L2 event fields:
    # scattering
    # fluence

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

    for event in events:
        #print('Send-to-db: event is', type(event), event)
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

                    # Find the known source by name

                    #Ignoring L2 key: known_source_rating = 0.9988938570022583
                    #Ignoring L2 key: known_source_metrics = {'position_weight': 1.0, 'position_bayes_factor_J0824+00': 7.4928274, 'dm_weight': 1.0, 'dm_bayes_factor_J0824+00': 120.52226426229063}

            # FIXME -- numerical event categories are silly; populate flags for RFI, known source,
            # pulsar, repeating FRB, new FRB, etc etc.
            # if k == 'event_category':
            #     if v == 3:
            #         l2_db_args['is_rfi'] = True
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
        session.flush()
        session.commit()

def setup():
    from frb_common import pipeline_tools
    from frb_common.events import L1Event
    import importlib.resources
    # all pipeline behaviour is encoded in config file
    configfn = 'drao_epsilon_pipeline_local.yaml'
    config = importlib.resources.files('chord_frb_sifter.config').joinpath(configfn)
    with importlib.resources.as_file(config) as config_path:
        pipeline_tools.load_configuration(config_path)
    bonsai_config = pipeline_tools.config["generics"]["bonsai_config"]
    L1Event.use_bonsai_config(bonsai_config)
                
# These are our simplified CHORD pipeline actors.
# A "pipeline" here is just a list of actors.
def simple_create_pipeline(database_engine):
    # Still reusing some of the config stuff... can probably simplify this too!!
    from frb_common import pipeline_tools

    from chord_frb_sifter.actors.beam_buffer import BeamBuffer
    from chord_frb_sifter.actors.beam_grouper import BeamGrouper
    from chord_frb_sifter.actors.localizer import Localizer
    from chord_frb_sifter.actors.simple_localizer import SimpleLocalizer
    from chord_frb_sifter.actors.bright_pulsar_sifter import BrightPulsarSifter
    from chord_frb_sifter.actors.rfi_sifter import RFISifter
    from chord_frb_sifter.actors.dm_checker import DMChecker
    #from chord_frb_sifter.actors.known_source import KnownSourceSifter
    from chord_frb_sifter.actors.actions import ActionPicker
    from chord_frb_sifter.actors.event_id_stamper import EventIdStamper

    pipeline = []
    for name,clz in [('BeamBuffer', BeamBuffer),
                     ('BeamGrouper', BeamGrouper),
                     ('EventIdStamper', EventIdStamper),
                     ('RFISifter', RFISifter),
                     ("BrightPulsarSifter", BrightPulsarSifter),
                     ('Localizer', Localizer), # Gauss2D localizer
                     #('Localizer', SimpleLocalizer), # S/N weighted
                     #('KnownSourceSifter', KnownSourceSifter),
                     ('DMChecker', DMChecker),
                     # ('FluxEstimator', FluxEstimator),
                     ('ActionPicker', ActionPicker),
                     ]:
        conf = pipeline_tools.get_worker_configuration(name)
        conf.pop('io')
        conf.pop('log')
        picl = conf.pop('use_pickle')
        conf.pop('timeout')
        conf.pop('periodic_update')
        conf.update(database_engine=database_engine)
        p = clz(**conf)
        pipeline.append(p)
    return pipeline

# Fires a set of saved events through the given pipeline.
def simple_process_events(pipeline, events):
    input_events = [events]
    output_events = []

    # This the famed "It's just a FOR loop" framework
    for actor in pipeline:
        output_events = []
        for in_item in input_events:
            items = actor.perform_action(in_item)
            if items is None:
                continue
            for item in items:
                if item is None:
                    continue
                output_events.append(item)
        if len(output_events) == 0:
            break
        input_events = output_events
    return output_events

# We should gather up all the CHIME and CHORD specific functionality we need into
# two subclasses.  For now...
def chime_beam_numbers_to_sky_grid(beams):
    # beam locations in units of "beam spacing" ... used by BeamGrouper, used related to thresholds
    # (in config file) ra_thr=3.1, dec_thr=2.1.
    beams = np.array(beams)
    ra_num = beams // 1000
    dec_num = beams % 1000
    return ra_num, dec_num

def chime_beam_numbers_to_dra_ddec(beams):
    beams = np.array(beams)
    ra_num = beams // 1000
    dec_num = beams % 1000

    northmost_beam = 60.0
    from scipy import constants as phys_const
    delta_y_feed_m = 0.3048
    freq_ref = (phys_const.speed_of_light * 128 /(np.sin(northmost_beam * np.pi / 180.0) * delta_y_feed_m * 256))
    clamp_freq = freq_ref

    Ny = 256
    feed_sep = delta_y_feed_m
    reference_indices = np.arange(Ny) + 1 - Ny / 2.0
    reference_angles = np.rad2deg(
        np.arcsin(phys_const.speed_of_light * reference_indices
                  / (clamp_freq * Ny * feed_sep)
                  )
    )
    assert(np.all(dec_num >= 0))
    assert(np.all(dec_num < len(reference_angles)))
    ddec = np.array(reference_angles)[dec_num]

    assert(np.all(ra_num >= 0))
    assert(np.all(ra_num < 4))
    ew_spacing = [-0.4,0,0.4,0.8]
    dra  = np.array(ew_spacing)[ra_num]
    return dra,ddec

# Reads a file full of events and fires it through the given pipeline, then saves results
# in the given database.
def simple_process_events_file(database_engine, pipeline, fn,
                               beam_to_dradec={}, beam_to_xygrid={}):
    from sqlalchemy.orm import Session
    fpgas,beams,events = simple_read_fits_events(fn)
    u_fpgas = np.unique(fpgas)
    for fpga in u_fpgas:
        I = np.flatnonzero(fpgas == fpga)
        b = beams[I]
        ubeams = np.unique(b)
        print(len(I), 'events for FPGA', fpga, 'in', len(ubeams), 'beams')
        for beam in ubeams:
            # Events for this FPGA chunk and beam number
            J = np.flatnonzero(b == beam)
            K = I[J]
            beam_events = [events[k] for k in K]

            gx,gy = beam_to_xygrid[beam]
            for e in beam_events:
                e['beam_grid_x'] = gx
                e['beam_grid_y'] = gy
            dr,dd = beam_to_dradec[beam]
            for e in beam_events:
                e['beam_dra'] = dr
                e['beam_ddec'] = dd
            for e in beam_events:
                e['is_incoherent'] = False

            #print('beam_events:', beam_events)

            outputs = simple_process_events(pipeline, beam_events)

            if outputs is None:
                print('Pipeline outputs:', outputs)
            else:
                print('Pipeline outputs:', len(outputs), 'events')

            # if len(outputs):
            #     # transaction block -- automatic commit on exit
            #     with Session(database_engine) as session:
            #         send_to_db(session, outputs)

# We dumped a bunch of CHIME/FRB events as FITS tables
def simple_read_fits_events(fn):
    events = fitsio.read(fn)
    print('Events file', fn, 'contains', len(events), 'events')
    beams = events['beam']
    fpgas = events['fpga']
    frame0nano = events['frame0_nano']
    eventlist = [{} for i in range(len(events))]
    for k in events.dtype.names:
        # output field name
        k_out = k
        # data vector for this column
        v = events[k]
        if k == 'beam_no':
            # duplicate of "beam", but in float for some reason
            continue
        if k == 'fpga':
            # rename!  This is the FPGAcount for the time chunk of data
            k_out = 'chunk_fpga'
        for i in range(len(events)):
            eventlist[i][k_out] = v[i]

    for e in eventlist:
        # compute timestamp_fpga to timestamp_utc (in micro-seconds)
        # ASSUME 2.56 microseconds per FPGA sample
        e['timestamp_utc'] = e['frame0_nano']/1000. + e['timestamp_fpga'] * 2.56
        # For completeness, also compute the chunk timestamp in UTC.
        e['chunk_utc'] = e['frame0_nano']/1000. + e['chunk_fpga'] * 2.56

        # HACK -- fake a DM error.  We should get this from, bonsai config?
        e['dm_error'] = 1.

    print('Event keys:', eventlist[0].keys())
    return fpgas,beams,eventlist

# This was old code that converted our FITS tables into the L1Event data type
# def read_fits_events(fn):
#     from frb_common.events import L1Event
#     events = fitsio.read(fn)
#     print('Events file', fn, 'contains', len(events), 'events')
# 
#     newevents = np.zeros(len(events), L1_EVENT_DTYPE)
#     for k in events.dtype.names:
#         if k in ['frame0_nano', 'beam', 'fpga']:
#             continue
#         newevents[k] = events[k]
#     beams = events['beam']
#     fpgas = events['fpga']
#     frame0nano = events['frame0_nano']
#     # compute timestamp_fpga to timestamp_utc (in micro-seconds)
#     # ASSUME 2.56 microseconds per FPGA sample
#     newevents['timestamp_utc'] = frame0nano/1000. + events['timestamp_fpga'] * 2.56
# 
#     events = newevents
#     events = L1Event(events)
#     events = events.demote()
#     return fpgas,beams,events

if __name__ == '__main__':
    '''
    export PYTHONPATH=${PYTHONPATH}:../frb_common/:../frb-l2l3/
    '''
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from chord_frb_db.models import Base, EventBeam, Event
    from sqlalchemy import delete

    database_engine = get_db_engine()

    # Drop all existing data!
    print('Dropping existing event data from database...')
    with Session(database_engine) as session:
        st = delete(EventBeam)
        session.execute(st)
        st = delete(Event)
        session.execute(st)
        session.commit()

    # Load pipeline config file
    setup()

    if False:
        # Mess around with known sources.
        # This loads an old CHIME known-sources table into the database,
        # but there is also chord_frb_sifter/scripts/load-known-sources.py, which does its own
        # parsing of the ATNF pulsar catalog and the RRATalog and puts them in the known-sources
        # database.
        import importlib.resources
        configfn = 'ks_database.npy'
        config = importlib.resources.files('chord_frb_sifter.data').joinpath(configfn)
        ks = None
        with importlib.resources.as_file(config) as config_path:
            ks = np.load(config_path)
        print('Got known sources:', ks)
        print('dtype:', ks.dtype)
    
        from chord_frb_db.models import KnownSource
        for i in range(len(ks)):
            name = ks['source_name'][i]
            ra = ks['pos_ra_deg'][i]
            dec = ks['pos_dec_deg'][i]
            dm = ks['dm'][i]
            typ = ks['source_type'][i]
    
            print('Type %i: DM %8.1f, name %s    RA,Dec %10.3f %10.3f' % (typ, dm, name, ra, dec))
    
            ks_args = dict(name=name, ra=ra, dec=dec, dm=dm)
            ks_obj = KnownSource(**ks_args)
            session.add(ks_obj)
        session.flush()
        session.commit()
        sys.exit(0)

    simple_pipeline = simple_create_pipeline(database_engine)

    # Set up CHIME beam stuff
    beams = np.hstack([np.arange(256) + i*1000 for i in range(4)])
    dra,ddec = chime_beam_numbers_to_dra_ddec(beams)
    beam_to_dradec = dict([(k,(v1,v2)) for k,v1,v2 in zip(beams, dra, ddec)])
    #xg,yg = chime_beam_numbers_to_sky_grid(beams)
    bm = cfbm.current_model_class()
    xg, yg = bm.get_cartesian_from_position(
        *bm.get_beam_positions(beams,freqs=bm.clamp_freq).squeeze().T
        )
    beam_to_xygrid = dict([(k,(v1,v2)) for k,v1,v2 in zip(beams, xg, yg)])

    # Fire saved CHIME/FRB events into pipeline
    for file_num in range(3):
        fn = 'events/events-%03i.fits' % file_num
        print('Processing events from file', fn)

        simple_process_events_file(database_engine, simple_pipeline, fn,
                                   beam_to_dradec=beam_to_dradec,
                                   beam_to_xygrid=beam_to_xygrid)

    print('Shutting down pipeline...')
    for actor in simple_pipeline:
        actor.shutdown()
    del simple_pipeline

    with Session(database_engine) as session:
        from sqlalchemy import select, func
        mx = session.query(func.max(Event.event_id)).scalar()
        print('Max event_id:', mx)
        stmt = select(Event).where(Event.event_id == mx)
        #print('Running statement:', stmt)
        result = session.execute(stmt)
        for r in result:
            e, = r
            print('Got:', e, 'with event_id', e.event_id)

    sys.exit(0)

'''
A reminder of what is in the events we have saved in the FITS files -- ie, what we get from
CHIME/FRB L1b:

L1 events:
     l1_timestamp 0.0
     pipeline_timestamp 679750.600012863
     pipeline_id 5
     beam_no 8
     timestamp_utc 1970-01-01 00:00:00
     timestamp_fpga 390856485888
     tree_index 0
     snr 10.849896430969238
     snr_scale 0.0
     dm 19.81433868408203
     spectral_index 0
     scattering_measure 0
     level1_nhits 0
     rfi_grade_level1 0
     rfi_mask_fraction 0.0
     rfi_clip_fraction 0.0
     snr_vs_dm [7.296021461486816, 6.753296375274658, 6.06824254989624, 5.921223163604736, 6.197881698608398, 7.471080780029297, 6.574099540710449, 8.159812927246094, 10.849896430969238, 7.941225051879883, 5.595312118530273, 7.371004104614258, 5.885530948638916, 6.015096187591553, 6.846845626831055, 5.988121032714844, 6.786647796630859]
     snr_vs_tree_index [10.849896430969238, 10.266348838806152, 0.0, 0.0, 0.0]
     snr_vs_spectral_index [10.849896430969238, 8.581052780151367]
     time_error 0.00786431971937418
     dm_error 0.40437427163124084
     pos_ra_deg 341.4575500488281
     pos_dec_deg -4.146359920501709
     pos_ra_error_deg 0.4806761145591736
     pos_dec_error_deg 0.8136351704597473
     is_incoherent False
     snr_vs_dm_x [16.579343795776367, 16.983718872070312, 17.388093948364258, 17.79246711730957, 18.196842193603516, 18.601215362548828, 19.005590438842773, 19.409963607788086, 19.81433868408203, 20.218713760375977, 20.62308692932129, 21.027462005615234, 21.431835174560547, 21.836210250854492, 22.240583419799805, 22.64495849609375, 23.049333572387695]
     snr_vs_tree_index_x [0.0, 0.0, 0.0, 0.0, 0.0]
     snr_vs_spectral_index_x [-3.0, 3.0]
     
'''
            
'''
L3 event:
list of L1 events

  timestamp_utc 1970-01-01 00:00:00+00:00
  combined_snr 13.975885
  beam_sensitivity 0.1264353532689731
  beam_sensitivity_min_95 0.0055235227409198935
  beam_sensitivity_max_95 0.12735711126662194
  flux_mjy 6635.414853517661
  flux_mjy_min_95 6543.348675854158
  flux_mjy_max_95 152902.43137003513
  pulse_width_ms 0.49151998246088624
  dm 19.814339
  dm_error 0.40437427
  spectral_index 0.0
  spectral_index_error 4.0
  rfi_grade_level2 10.0
  rfi_grade_metrics_level2 {}
  pos_ra_deg 341.4635913471214
  pos_dec_deg -2.231904521712195
  pos_error_semimajor_deg_68 0.4052587554112179
  pos_error_semimajor_deg_95 0.7943071606059872
  pos_error_semiminor_deg_68 0.2358026254396334
  pos_error_semiminor_deg_95 0.46217314586168146
  pos_error_theta_deg_68 0.0
  pos_error_theta_deg_95 0.0
  known_source_name
  known_source_rating -1
  known_source_metrics {}
  dm_gal_ne_2001_max 34.9819916452506
  dm_gal_ymw_2016_max 25.60786075834742
  beam_activity 16
  unknown_event_type 0
  event_category 1
  version XXX
  actions {'GET_INTENSITY': {'REQUEST': False}, 'GET_BASEBAND': {'REQUEST': False}, 'ALERT_PULSAR': {'REQUEST': False}, 'ALERT_COMMUNITY': {'REQUEST': False}, 'SEND_HEADER': {'REQUEST': True}}
  event_status {'BeamGrouper': 0, 'RFISifter': 0, 'Localizer': 0, 'KnownSourceSifter': 0, 'DMChecker': 0, 'FluxEstimator': 0, 'ActionPicker': 0}
  pipeline_mode {'BeamGrouper': -1, 'RFISifter': 2, 'Localizer': 2, 'KnownSourceSifter': 2, 'DMChecker': 2, 'FluxEstimator': 2, 'ActionPicker': 2}
  is_test False
  futures {'coh_dm_activity': 276, 'dm_activity_lookback': [0, 0, 0, 0, 0, 0, 0, 0, 0, 276], 'beam_activity_lookback': [0, 0, 0, 0, 0, 0, 0, 0, 0, 16], 'incoh_dm_activity': 0, 'dm_std': 41.658413, 'avg_l1_grade': 0.020357803824799507, 'trash_at_exit': False}
  event_processing_start_time 1724963214.8027909
'''
'''
  L1 Event:
     l1_timestamp 0.0
     pipeline_timestamp 327984.978451527
     pipeline_id 1093
     beam_no 11
     timestamp_utc 1970-01-01 00:00:00
     timestamp_fpga 390856731648
     tree_index 0
     snr 13.975885391235352
     snr_scale 0.0
     dm 19.81433868408203
     spectral_index 0
     scattering_measure 0
     level1_nhits 0
     rfi_grade_level1 0
     rfi_mask_fraction 0.0
     rfi_clip_fraction 0.0
     snr_vs_dm [12.128884315490723, 9.697505950927734, 7.5382080078125, 9.464571952819824, 8.81373119354248, 8.495182037353516, 8.880011558532715, 9.691327095031738, 13.975885391235352, 11.18685531616211, 8.045234680175781, 8.006065368652344, 7.89288330078125, 7.369781017303467, 9.25119686126709, 9.955826759338379, 8.459622383117676]
     snr_vs_tree_index [13.975885391235352, 12.649748802185059, 7.9205121994018555, 0.0, 0.0]
     snr_vs_spectral_index [13.975885391235352, 11.834529876708984]
     time_error 0.00786431971937418
     dm_error 0.40437427163124084
     pos_ra_deg 341.4533996582031
     pos_dec_deg -2.228651762008667
     pos_ra_error_deg 0.4806761145591736
     pos_dec_error_deg 0.7724871039390564
     is_incoherent False
     snr_vs_dm_x [16.579343795776367, 16.983718872070312, 17.388093948364258, 17.79246711730957, 18.196842193603516, 18.601215362548828, 19.005590438842773, 19.409963607788086, 19.81433868408203, 20.218713760375977, 20.62308692932129, 21.027462005615234, 21.431835174560547, 21.836210250854492, 22.240583419799805, 22.64495849609375, 23.049333572387695]
     snr_vs_tree_index_x [0.0, 0.0, 0.0, 0.0, 0.0]
     snr_vs_spectral_index_x [-3.0, 3.0]
'''
