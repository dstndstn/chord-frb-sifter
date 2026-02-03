"""
This a CHORD/FRB prototype, modified from the CHIME/FRB code.

It groups L1 events in DM, time, and position.
Grouped events form a candidate L2 event.
"""

import time
import json
import traceback
import pickle as pickle
from subprocess import check_output
from collections import deque

import numpy as np
from scipy.spatial import cKDTree
import msgpack

#from frb_common import ActorBaseClass
#from frb_common import configuration_manager as cm
#from frb_common.events import L1Event

from chord_frb_sifter.actors.actor import Actor
from chord_frb_sifter.event import L1Event, L2Event

__author__ = "CHIME FRB Group"
__developers__ = "Alex Josephy"
__email__ = "alexander.josephy@mail.mcgill.ca"

# This incorporates steps that used to be in the EventMaker actor.
def create_l2_event(l1_events, **kwargs):
    print('Creating L2 event from L1 events:')
    print("l1_events type:",type(l1_events),type(l1_events[0]))
    for e in l1_events:
        print('  ', e)
    from collections import Counter
    print('Beam counts:', Counter([e['beam'] for e in l1_events]))
    #
    # Keep only the max-SNR event for each beam.
    beam_maxsnr = {}
    for e in l1_events:
        beam = e['beam']
        snr = e['snr']
        if beam in beam_maxsnr:
            (oldsnr,_) = beam_maxsnr[beam]
            if snr > oldsnr:
                beam_maxsnr[beam] = (snr,e)
        else:
            beam_maxsnr[beam] = (snr, e)
    # Initialize L2 elements from the max-SNR event
    keep = []
    best_event = None
    best_snr = 0.
    for snr,e in beam_maxsnr.values():
        keep.append(e)
        if snr > best_snr:
            best_snr = snr
            best_event = e
    l1_events = keep
    # FIXME - this is silly
    l2_event = L2Event(best_event) # assuming best L1 event is a dictionary
    for k in ['beam_grid_x', 'beam_grid_y', 'beam_dra', 'beam_ddec', 'snr']:
        l2_event['max_' + k] = best_event[k]
        del l2_event[k]
    #
    l2_event.update(kwargs)
    
    # Now that things are grouped and number of L1 events is fixed, cast as L1Event recarray
    l2_event['l1_events'] = L1Event(l1_events) 
    return l2_event

class BeamGrouper(Actor):
    """
    The purpose of this class is to group together L1 events detected in
    different beams that were presumably caused by a common incident pulse.
    These multibeam detections may arise from very bright astrophysical bursts
    as well as near-field RFI.

    Parameters
    ----------
    t_thr, dm_thr, ra_thr, dec_thr : float
        Thresholds in ms, pc cm :sup:`-3`, and beam separation
    **kwargs : dict, optional
        Additional parameters are used to initialize superclass
        (``ActorBaseClass``).

    Extended Summary
    ----------------
    A group is defined as a number of L1 events where, for any event in
    the group, there exists another event whose differences in DM, time,
    RA, and Dec are all below the specified thresholds.

    The grouping is done with the DBSCAN algorithm :sup:`[1,2]`. To use this
    method, we need to first define what a *distance* between different events
    means.  To do this, we first scale the aforementioned axes by dividing the
    values by the relevant thresholds. We then apply the Chebyshev metric
    :sup:`[3]` to get a meaningful distance. The above group definition now
    ensures that, for every event in a group, there exists another event such
    that the *distance* between them is less than 1.

    See Also
    --------
    frb_L2_L3.BeamBuffer :
        The intended upstream actor :ref:`(link) <L1_L2_buffer_doc_page>`

    frb_L2_L3.RFISifter :
        The intended downstream actor :ref:`(link) <L2_rfi_sifter_doc_page>`

    frb_common.ActorBaseClass :
        The superclass :ref:`(link) <actor_base_class_doc_page>`

    frb_common.WorkerProcess :
        The usual wrapper class :ref:`(link) <pipeline_tools_doc_page>`

    References
    ----------
    [1] Ester, M., Kriegel, H.P., Sander, J., & Xu, X. A density-based
    algorithm for discovering clusters in large spatial databases with
    noise. 1996, Proc. 2nd Int. Conf. on Knowledge Discovery and Data
    Mining (Portland, OR: AAAI Press), 226

    [2] `DBSCAN <https://en.wikipedia.org/wiki/DBSCAN>`_

    [3] `Chebyshev Metric <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
    """
    def __init__(self, t_thr, dm_thr, ra_thr, dec_thr, **kwargs):
        super().__init__(**kwargs)
        self.thresholds = [t_thr, dm_thr, ra_thr, dec_thr]
        self.dm_activity_lookback = deque([0] * 10, maxlen=10)
        self.beam_activity_lookback = deque([0] * 10, maxlen=10)

    def _perform_action(self, events):
        """Pipeline function that groups L1 events.

        Parameters
        ----------

        Returns
        -------
        list of ``L2Event``
        """
        print('Beam grouper: %i events' % len(events))
        print('First event:', events[0])
        groups = self._cluster(events)

        dead_beams = [] # Needed for CHIME RFISifter, will probably want for CHORD, but likely won't come from L1 per event.
        beam_activity = len(set([e['beam'] for e in events]))
        dm_activity = len(set([e['dm'] for e in events]))
        avg_l1_grade = np.mean([e['rfi_grade_level1'] for e in events])
        self.dm_activity_lookback.append(dm_activity)
        self.beam_activity_lookback.append(beam_activity)

        # tnow = time.monotonic()
        # for g in groups:
        #     try:
        #         tmin = min(g['pipeline_timestamp'])
        #         pid = g['pipeline_id']
        #         self.print('BeamGrouper: Elapsed Time for %i: %.3f sec (pipeline timestamp: %s)' %
        #                    (pid, tnow - tmin, tmin))
        #     except:
        #         pass

        #"dm_std": coh_events.dm.std(),
        l2_events = []
        for group in groups:
            l2_event = create_l2_event(group,
                                       beam_activity=beam_activity,
                                       dm_activity=dm_activity,
                                       beam_activity_lookback=self.beam_activity_lookback,
                                       dm_activity_lookback=self.dm_activity_lookback,
                                       avg_l1_grade=avg_l1_grade,
                                       dead_beams=dead_beams,
                                       )
            l2_events.append(l2_event)
        return l2_events

    def _cluster(self, events):
        """ Performs event clustering via DBSCAN algorithm """
        # make a new (time, dm, x, y) array that will be scaled by thresholds
        tdmxy = np.empty((len(events), 4), np.float32)
        times = np.array([e['timestamp_utc'] for e in events])
        times = (times - min(times)) / 1e3
        tdmxy[:, 0] = times
        tdmxy[:, 1] = np.array([e['dm'] for e in events])
        tdmxy[:, 2] = np.array([e['beam_grid_x'] for e in events]) # x (RA--like) in beam grid
        tdmxy[:, 3] = np.array([e['beam_grid_y'] for e in events]) # y (Dec-like) in beam grid
        tdmxy /= self.thresholds

        tree = cKDTree(tdmxy)
        neighbors = tree.query_ball_tree(tree, r=1.0, p=np.inf)

        groups = {}
        visiting = []
        undiscovered = set(range(len(events)))
        while undiscovered:
            root = undiscovered.pop()
            visiting.append(root)
            while visiting:
                event = visiting.pop()
                groups.setdefault(root, []).append(event)
                for new in set(neighbors[event]).intersection(undiscovered):
                    undiscovered.remove(new)
                    visiting.append(new)

        #print('group indices:', [g for g in list(groups.values())])
        return [[events[gi] for gi in g] for g in list(groups.values())]
