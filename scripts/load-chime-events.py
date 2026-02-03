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

def read_fits_events(fn):
    from frb_common.events import L1Event
    events = fitsio.read(fn)
    print('Events file', fn, 'contains', len(events), 'events')

    newevents = np.zeros(len(events), L1_EVENT_DTYPE)
    for k in events.dtype.names:
        if k in ['frame0_nano', 'beam', 'fpga']:
            continue
        newevents[k] = events[k]
    beams = events['beam']
    fpgas = events['fpga']
    frame0nano = events['frame0_nano']
    # compute timestamp_fpga to timestamp_utc (in micro-seconds)
    # ASSUME 2.56 microseconds per FPGA sample
    newevents['timestamp_utc'] = frame0nano/1000. + events['timestamp_fpga'] * 2.56

    events = newevents
    events = L1Event(events)
    events = events.demote()
    #print('Final event type:', events.dtype)
    #print('Event timestamp_utc:', events['timestamp_utc'])
    return fpgas,beams,events

def send_to_db(session, outputs):
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

    payloads = []

    for olist in outputs:
        out_payloads = []
        payloads.append(out_payloads)
        for event in olist:

            l2_db_args = { 'is_rfi': False,
                           'is_known_pulsar': False,
                           'is_new_burst': False,
                           'is_frb': False,
                           'is_repeating_frb': False,
                           'scattering': 0.,
                           'fluence': 0.,
                           }
            l1_objs = []

            payload = event.database_payload()
            out_payloads.append(payload.copy())

            print('DB payload:')
            for k,v in payload.items():
                # skip...
                if k in ['dead_beam_nos']:
                    continue
                if k == 'l1_events':
                    #print(' ', k, ':', len(v))
                    for l1 in v:
                        l1_db_args = {}
                        #print('  L1 Event:')
                        for l1k,l1v in l1.items():
                            #print('    ', l1k, l1v)
                            if l1k == 'timestamp_utc':
                                l1v = l1v.timestamp()
                            k2 = l1_name_map.get(l1k, None)
                            if k2 is not None:
                                # same key name
                                if k2 is True:
                                    k2 = l1k
                                l1_db_args[k2] = l1v

                        #for k,v in l1_db_args.items():
                        #    print('  L1:', k, '=', type(v), v)
                        l1_db_obj = EventBeam(**l1_db_args)
                        #print('Created L1 db object:', l1_db_obj)
                        session.add(l1_db_obj)
                        session.flush()
                        #print('L1 db id:', l1_db_obj.id)
                        assert(l1_db_obj.id is not None)
                        #l1_event_ids.append(l1_db_obj.id)
                        l1_objs.append(l1_db_obj)
                        #session.commit()
                    continue
                if isinstance(v, (np.float32, np.float64)):
                    v = float(v)
                #print(' ', k, v)

                if k == 'timestamp_utc':
                    v = v.timestamp()
                k2 = l2_name_map.get(k, None)
                if k2 is not None:
                    # same key name
                    if k2 is True:
                        k2 = k
                    l2_db_args[k2] = v
                else:
                    print('Ignoring L2 key:', k, '=', v)

                if k == 'known_source_name':
                    if v != "":
                        print('Known source!')
                        print('val: "%s"' % v)

                        # Find the known source by name
                        

                        #Ignoring L2 key: known_source_rating = 0.9988938570022583
                        #Ignoring L2 key: known_source_metrics = {'position_weight': 1.0, 'position_bayes_factor_J0824+00': 7.4928274, 'dm_weight': 1.0, 'dm_bayes_factor_J0824+00': 120.52226426229063}
                        
                if k == 'event_category':
                    if v == 3:
                        l2_db_args['is_rfi'] = True
                if k == 'flux_mjy':
                    # milli -> Jansky
                    l2_db_args['flux'] = 0.001 * v

            #l2_db_args['nbeams'] = len(l1_event_ids)
            l2_db_args['nbeams'] = len(l1_objs)

            #for k,v in l2_db_args.items():
            #    print(' L2:', k, '=', type(v), v)

            l2_db_obj = Event(**l2_db_args)
            #print('Created L2 db object:', l2_db_obj)
            session.add(l2_db_obj)
            #print('L1 event ids:', l1_event_ids)
            #print('L2 beams:', l2_db_obj.beams)
            #for b in l1_event_ids:
            #    l2_db_obj.beams.append(b)
            for e in l1_objs:
                l2_db_obj.beams.append(e)
            session.flush()
            #print('L2 db id:', l2_db_obj.event_id)
            #print('L2 beam ids:', l2_db_obj.beams)
            #print('L1 events:', l1_objs)
            #print('  L1 back-pointers:', [e.event_id for e in l1_objs])
            session.commit()

        print('Output payloads:', len(out_payloads))
    print('Payloads:', len(payloads))

    return payloads
            
''' L1 events:
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
                
# Start creating a parallel version of the pipeline, porting stuff over while
# simplifying!

def simple_create_pipeline():
    from frb_common import pipeline_tools

    from chord_frb_sifter.actors.beam_buffer import BeamBuffer
    from chord_frb_sifter.actors.beam_grouper import BeamGrouper
    from chord_frb_sifter.actors.localizer import Localizer
    from chord_frb_sifter.actors.simple_localizer import SimpleLocalizer
    from chord_frb_sifter.actors.bright_pulsar_sifter import BrightPulsarSifter
    from chord_frb_sifter.actors.rfi_sifter import RFISifter
    from chord_frb_sifter.actors.dm_checker import DMChecker

    pipeline = []
    for name,clz in [('BeamBuffer', BeamBuffer),
                     ('BeamGrouper', BeamGrouper),
                     # ('EventMaker', EventMaker),
                     ('RFISifter', RFISifter),
                     ("BrightPulsarSifter", BrightPulsarSifter),
                     ('Localizer', Localizer), # Gauss2D localizer
                     #('Localizer', SimpleLocalizer), # S/N weighted
                     # ('KnownSourceSifter', KnownSourceSifter),
                     ('DMChecker', DMChecker),
                     # ('FluxEstimator', FluxEstimator),
                     # ('ActionPicker', ActionPicker),
                     ]:
        conf = pipeline_tools.get_worker_configuration(name)
        conf.pop('io')
        conf.pop('log')
        picl = conf.pop('use_pickle')
        conf.pop('timeout')
        conf.pop('periodic_update')
        p = clz(**conf)
        pipeline.append(p)
    return pipeline

#def simple_process_events(pipeline, fpga, beam, events):
def simple_process_events(pipeline, events):
    print('events:', type(events), events)

    input_events = [events]
    output_events = []
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
    # Event keys: dict_keys(['beam_no', 'timestamp_utc', 'timestamp_fpga',
    # 'tree_index', 'snr', 'snr_scale', 'dm', 'spectral_index',
    # 'scattering_measure', 'level1_nhits', 'rfi_grade_level1',
    # 'rfi_mask_fraction', 'rfi_clip_fraction', 'snr_vs_dm',
    # 'snr_vs_tree_index', 'snr_vs_spectral_index'])

    #events: <class 'list'> [{
    #  'beam_no': np.float64(10.0),
    #  'timestamp_utc': np.float64(1723661291587945.0),
    #  'timestamp_fpga': np.float64(390849057791.0),
    #  'tree_index': np.uint8(0),
    #  'snr': np.float32(7.524105),
    #  'snr_scale': np.float32(0.0),
    #  'dm': np.float32(21.431837),
    #  'spectral_index': np.uint8(1),
    #  'scattering_measure': np.uint8(0),
    #  'level1_nhits': np.float64(0.0),
    #  'rfi_grade_level1': np.uint8(9),
    #  'rfi_mask_fraction': np.float32(0.0),
    #  'rfi_clip_fraction': np.float32(0.0),
    #  'snr_vs_dm': array([4.6889935, 3.897043 , 4.323593 , 4.990237 ,
    #              5.1774907, 4.9759393, 5.056928 , 7.116945 , 7.524105 ,
    #              6.5427437, 5.561984 , 4.7888064, 4.408268 , 5.563717 ,
    #              6.746774 , 5.896961 , 4.91972  ], dtype='>f4'),
    #  'snr_vs_tree_index': array([7.524105, 0.      , 0.      ,
    #                              0.      , 0.      ], dtype='>f4'),
    #  'snr_vs_spectral_index': array([5.5828123, 7.524105 ], dtype='>f4')}]
    
    #outputs = process_events(pipeline, event_data)


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
    print('reference_angles', reference_angles)

    print('dec_num', dec_num.max(), 'refs', len(reference_angles))
    
    assert(np.all(dec_num >= 0))
    assert(np.all(dec_num < len(reference_angles)))
    ddec = np.array(reference_angles)[dec_num]

    assert(np.all(ra_num >= 0))
    assert(np.all(ra_num < 4))
    ew_spacing = [-0.4,0,0.4,0.8]
    dra  = np.array(ew_spacing)[ra_num]
    return dra,ddec

    
def simple_process_events_file(engine, pipeline, fn,
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
                print('Pipeline outputs:', len(outputs))
            # if len(outputs):
            #     # transaction block -- automatic commit on exit
            #     with Session(engine) as session:
            #         payloads = send_to_db(session, outputs)

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

    print('Event keys:', eventlist[0].keys())
    return fpgas,beams,eventlist

if __name__ == '__main__':
    '''
    export PYTHONPATH=${PYTHONPATH}:../frb_common/:../frb-l2l3/:../L4_pipeline/:../L4_databases/

    export PYTHONPATH=${PYTHONPATH}:../frb_common/:../frb-l2l3/
    '''

    
    from frb_common import pipeline_tools
    #from frb_L2_L3.actors.localizer import Localizer

    import importlib.resources
    # all pipeline behaviour is encoded in config file
    print('Loading config file...')
    configfn = 'drao_epsilon_pipeline_local.yaml'
    config = importlib.resources.files('chord_frb_sifter.config').joinpath(configfn)
    with importlib.resources.as_file(config) as config_path:
        pipeline_tools.load_configuration(config_path)
    bonsai_config = pipeline_tools.config["generics"]["bonsai_config"]

    if False:
        #from chord_frb_sifter.actors.beam_grouper import BeamGrouper
        from frb_L2_L3.actors.rfi_sifter import RFISifter
    
        #name,clz = ('Localizer', Localizer)
        name,clz = ('RFISifter', RFISifter)
        conf = pipeline_tools.get_worker_configuration(name)
        print('Got actor config:', conf)
        conf.pop('io')
        conf.pop('log')
        picl = conf.pop('use_pickle')
        conf.pop('timeout')
        conf.pop('periodic_update')
        p = clz(**conf)
        print('Got actor:', p)
    
        # from frb_L2_L3.actors.localizer import lookup
        
        sys.exit(0)
    
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from chord_frb_db.models import Base, EventBeam, Event
    from sqlalchemy import delete

    engine = get_db_engine()

    # Drop all existing data!!!

    print('Dropping existing event data...')
    with Session(engine) as session:
        st = delete(EventBeam)
        session.execute(st)
        st = delete(Event)
        session.execute(st)
        session.commit()

    setup()

    if False:
        import importlib.resources
        configfn = 'ks_database.npy'
        config = importlib.resources.files('chord_frb_sifter.data').joinpath(configfn)
        ks = None
        with importlib.resources.as_file(config) as config_path:
            ks = np.load(config_path)
        print('Got known sources:', ks)
        print('dtype:', ks.dtype)
    
        '''
         ID  SOURCE_TYPE      POS_RA_DEG       POS_ERROR_SEMIMAJOR_DEG    DM         DM_ERROR  SPIN_PERIOD_SEC  DM_GALACTIC_NE_2001_MAX    CHIME_FRB_PEAK_FLUX_DENSITY_JY
                  SOURCE_NAME           POS_DEC_DEG   POS_ERROR_SEMIMINOR_DEG                          SPIN_PERIOD_SEC_ERROR          SPECTRAL_INDEX
                                                               POS_ERROR_THETA_DEG                                        DM_GALACTIC_YMW_2016_MAX
        (2580, 1, 'J1336+34', 203.969  , 34.12    , 0.068, 0.068   , 0.,    8.     ,        nan, 1.506  , nan, 23.823574,  20.1787  , 0., 0.)
        (2581, 1, 'J1748+59', 267.599  , 59.8     , 0.068, 1.268432, 0.,   45.     ,        nan, 0.43604, nan, 46.943584,  40.937115, 0., 0.)
        (2582, 2, 'J0209+58',  32.273  , 58.178   , 0.068, 0.068   , 0.,   56.     ,        nan,     nan, nan, 90.727585, 106.394104, 0., 0.)
         ...
        (7045, 3, '63566525', 215.90079, 82.63237 , 0.068, 0.068   , 0.,  690.67126,  1.6174971,     nan, nan, 45.73806 ,  39.735348, 0., 0.)
        (7046, 3, '63563443', 186.47298, 65.13165 , 0.068, 0.068   , 0.,  375.2593 , 12.939977 ,     nan, nan, 34.318127,  25.975077, 0., 0.)
        (7047, 3, '63557484', 121.33715, 51.708534, 0.068, 0.068   , 0., 1095.0455 ,  1.6174971,     nan, nan, 52.56251 ,  48.663383, 0., 0.)]
        '''
    
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

    #pipeline = create_pipeline()

    simple_pipeline = simple_create_pipeline()

    beams = np.hstack([np.arange(256) + i*1000 for i in range(4)])
    dra,ddec = chime_beam_numbers_to_dra_ddec(beams)
    beam_to_dradec = dict([(k,(v1,v2)) for k,v1,v2 in zip(beams, dra, ddec)])
    #xg,yg = chime_beam_numbers_to_sky_grid(beams)

    bm = cfbm.current_model_class()
    xg, yg = bm.get_cartesian_from_position(
        *bm.get_beam_positions(beams,freqs=bm.clamp_freq).squeeze().T
        )
    
    beam_to_xygrid = dict([(k,(v1,v2)) for k,v1,v2 in zip(beams, xg, yg)])

    for file_num in range(3):
        fn = 'events/events-%03i.fits' % file_num
        #process_events_file(engine, pipeline, fn)

        # print('<<< simple >>>')
        simple_process_events_file(engine, simple_pipeline, fn,
                                   beam_to_dradec=beam_to_dradec,
                                   beam_to_xygrid=beam_to_xygrid)
        # print('<<< /simple >>>')
