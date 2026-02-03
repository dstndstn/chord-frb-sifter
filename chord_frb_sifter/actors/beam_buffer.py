"""
This is a CHORD/FRB prototype version, modified from CHIME/FRB.
This actor buffers input from L1, gathering all events from a given time-chunk of data.
"""

from os import path
import traceback
from datetime import datetime

import numpy as np
import msgpack

import time

from chord_frb_sifter.actors.actor import Actor

class BeamBuffer(Actor):
    """
    The purpose of this class is to accumulate events from individual beams
    into a single frame, such that events may be grouped.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipe_id = 0

        self.current_chunk = None
        self.previous_chunk = None
        self.buffered_events = []

        self.slowpoke_events = []

        # FIXME -- for CHORD, we could do this at the node level instead.
        self.current_beams = set()
        self.expecting_beams = None

    def _perform_action(self, events):

        # for debugging purposes, tag events...
        tnow = time.monotonic()
        for e in events:
            e['pipeline_timestamp'] = tnow
            e['pipeline_id'] = self.pipe_id
            self.pipe_id += 1
        
        # First, group events into per-timestamp, per-beam event groups.  This makes life easier below.
        event_sets = []
        current_chunk = None
        current_beam = None
        current_events = None
        for e in events:
            chunk = e['chunk_utc']
            beam = e['beam']
            if current_chunk is None:
                current_chunk = chunk
                current_beam = beam
                current_events = [e]
            elif (chunk != current_chunk) or (beam != current_beam):
                # flush
                event_sets.append((current_chunk, current_beam, current_events))
                current_chunk = chunk
                current_beam = beam
                current_events = [e]
            else:
                current_events.append(e)
        if current_chunk is not None:
            event_sets.append((current_chunk, current_beam, current_events))


        rtn = []
        def _flush_events():
            if len(self.slowpoke_events):
                print('Flushing', len(self.slowpoke_events), 'slow-pokes')
                rtn.append(self.slowpoke_events)
                self.slowpoke_events = []
            if len(self.buffered_events):
                print('Flushing', len(self.buffered_events), 'events')
                rtn.append(self.buffered_events)
                self.buffered_events = []
            self.expecting_beams = self.current_beams
            self.current_beams = set()
            self.previous_chunk = self.current_chunk

        def _append_events(lst, evts):
            for e in evts:
                #if not e.get('null_event', False):
                #    lst.append(e)
                lst.append(e)

        for chunk, beam, events in event_sets:
            if chunk == self.previous_chunk:
                # slowpoke -- at startup, or when replaying from a file, you could get:
                #  chunk 0, beam 1
                #
                #  chunk 1, beam 2
                #  chunk 1, beam 1 --> flush!
                #  chunk 1, beam 3 --> slowpoke!
                #
                #  chunk 2, beam 2
                #  chunk 2, beam 3
                #  chunk 2, beam 1 --> flush!
                #
                #  etc.
                self.expecting_beams.add(beam)
                # FIXME -- append to previous event set?
                print('Got slow-poke event: chunk', chunk, 'current', self.current_chunk,
                      'previous', self.previous_chunk, 'beam:', beam)
                if len(rtn):
                    print('adding to last batch')
                    _append_events(rtn[-1], events)
                else:
                    print('adding to slow-pokes')
                    _append_events(self.slowpoke_events, events)
                continue

            if self.current_chunk is None:
                self.current_chunk = chunk
                print('Starting new batch: chunk', chunk, 'expecting beams:', self.expecting_beams)
                if len(self.slowpoke_events):
                    print('Flushing', len(self.slowpoke_events), 'slow-pokes')
                    rtn.append(self.slowpoke_events)
                    self.slowpoke_events = []

            if chunk > self.current_chunk:
                # We got the first event from a new chunk -- flush our current event list!
                print('New chunk - flushing %i events for chunk %s; beams %s' %
                      (len(self.buffered_events), self.current_chunk, self.current_beams))
                _flush_events()
                self.current_chunk = chunk

            if chunk < self.current_chunk:
                # FIXME -- very slow event... do something??
                print('Ignoring old events: chunk', chunk, 'but current is', self.current_chunk,
                      'and previous was', self.previous_chunk, '; events are', events)
                continue

            self.current_beams.add(beam)
            _append_events(self.buffered_events, events)

            if self.expecting_beams is not None:
                if self.current_beams.issuperset(self.expecting_beams):
                    # All expected beams have been received -- flush our current event list!
                    print('Got all beams (%s) - expected beams (%s) - flushing %i events for chunk %s' %
                          (self.current_beams, self.expecting_beams, len(self.buffered_events), self.current_chunk))
                    _flush_events()
                    self.current_chunk = None
                    print('Next chunk, will expect beams %s' % self.expecting_beams)

        return rtn
