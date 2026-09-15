"""
This is a CHORD/FRB prototype version, modified from CHIME/FRB.
This actor buffers input from L1, gathering all events from a given time-chunk of data.
"""
import time
import traceback
from datetime import datetime

import numpy as np

from chord_frb_sifter.actors import Actor
from chord_frb_sifter.event import EventGroup

class BeamBuffer(Actor):
    """
    This actor groups EventGroups received from all the Pirate search machines for a single time chunk.
    """
    def __init__(self, sifter=None, **kwargs):
        super().__init__(**kwargs)
        self.sifter = sifter
        self.pipe_id = 0

        self.current_chunk = None
        self.buffered_events = []

        self.current_beamsets = set()
        self.expecting_beamsets = None

    def __str__(self):
        return 'BeamBuffer'

    '''
    This actor receives EventGroups from the Pirate machines (via the
    gRPC endpoint).  In each time chunk, each pirate node sends a gRPC
    that becomes an EventGroup, which may have zero or more events in
    it.

    Once an EventGroup has been received from all the Pirates, a new
    EventGroup is created for the time chunk and it is sent
    downstream.

    Each pirate machine handles a set of beams (a "beamset"), which is
    the terminology used below.
    '''
    def _perform_action(self, event_group):
        # Assume the event_groups we get are from a single time-chunk and beamset.

        # We will group these into (one or more) event_groups each from a single time-chunk.

        # for debugging purposes, tag events...
        tnow = time.monotonic()
        for e in event_group.events:
            e['pipeline_timestamp'] = tnow
            e['pipeline_id'] = self.pipe_id
            self.pipe_id += 1

        # Return value - flushed event groups
        rtn = []

        def _flush():
            # Only send event-groups downstream if there are actually events!
            if not any([len(g.events)>0 for g in self.buffered_events]):
                return
            chunk_group = EventGroup(chunk_utc = self.current_chunk,
                                     events = [],
                                     n_beamsets = len(self.buffered_events),
                                     )
            chunk_group.n_live_beams = 0
            for g in self.buffered_events:
                chunk_group.events.extend(g.events)
                chunk_group.n_live_beams += len(self.sifter.beamset_to_beam_ids(g.beamset))
            rtn.append(chunk_group)

        # This is a "while True" loop because there's one case where we want to loop twice;
        # the normal case is to "break" out
        while True:
            if self.current_chunk is None:
                #print('Starting to collect event_group for chunk %s' % event_group.chunk_utc)
                self.current_chunk = event_group.chunk_utc
    
            if event_group.chunk_utc == self.current_chunk:
                self.buffered_events.append(event_group)
                self.current_beamsets.add(event_group.beamset)
                if self.current_beamsets == self.expecting_beamsets:
                    # Received all beamsets - flush events!
                    #print('Received all beamsets expected for chunk %s' % event_group.chunk_utc)
                    _flush()
                    self.buffered_events = []
                    self.current_chunk = None

            elif event_group.chunk_utc < self.current_chunk:
                # Slowpoke
                if len(event_group.events):
                    rtn.append(EventGroup(chunk_utc = event_group.chunk_utc,
                                          events = event_group.events))
                # we should expect this beamset next time
                self.current_beamsets.add(event_group.beamset)

            elif event_group.chunk_utc > self.current_chunk:
                # Got the first event_group of a new chunk.
                # If this is the first chunk, this is expected.  Otherwise, we expect to receive
                # all beamsets that we were expecting.  So this possibly indicates one beamset
                # running fast?
                if self.expecting_beamsets is None:
                    # If we're starting up -- next chunk, we expect the same beamsets
                    self.expecting_beamsets = self.current_beamsets
                    #print('Received the first event_group for the next chunk')
                else:
                    print('Received an unexpected event_group for chunk %s while processing chunk %s' %
                          (event_group.chunk_utc, self.current_chunk))
                # Flush the current_chunk
                _flush()
                # Set my state ready for this first event_group in the next chunk & repeat the loop
                self.buffered_events = []
                self.current_chunk = None
                continue

            # Normal case - one time through the loop
            break

        return rtn
