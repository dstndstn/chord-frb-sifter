from chord_frb_sifter.actors import Actor
from chord_frb_sifter.event import L1Event, L2Event

from chord_frb_db.utils import get_db_engine
from sqlalchemy.orm import Session

from queue import Queue
from threading import Thread

import numpy as np

import concurrent.futures as cf

# Should the database stuff be in a separate actor?

class ActionPicker(Actor):
    def __init__(self, database_engine=None, **kwargs):
        super().__init__(**kwargs)
        self.db_queue = Queue()
        self.db_thread = Thread(target=ActionPicker.run_db, args=(self, database_engine))
        self.db_thread.start()
        #self.intensity_queue = Queue()
        #self.intensity_thread = Thread(target=ActionPicker.run_intensity,
        #                               args=(self,))
        #self.intensity_thread.start()
        #self.db_executor = cf.ThreadPoolExecutor(max_workers=3)
        self.intensity_executor = cf.ThreadPoolExecutor(max_workers=10)
        self.intensity_futures = []

    def shutdown(self):
        #self.db_executor.shutdown(wait=True)
        print('Shutting down ActionPicker... sending None on db queue')
        self.db_queue.put(None)
        #self.intensity_queue.put(None)
        print('Joining db thread')
        self.db_thread.join()
        #print('Joining intensity callback thread')
        #self.intensity_thread.join()
        print('Shutting down intensity callback thread pool...')
        self.intensity_executor.shutdown(wait=True)
        print('Shutdown of intensity callback thread pool finished')
        print('done shutdown of ActionPicker')

    def _perform_action(self, event):
        # Log everything in db?
        self.save_to_db(event)

        if event.is_frb() or event.is_ambiguous():
            print('FRB or Ambiguous event -- sending intensity callback!')
            self.send_intensity_callback(event)
        # if event.is_rfi():
        #     return event
        # 
        # if event.is_frb():
        #     pass
        return [event]

    def save_to_db(self, event):
        # FIXME -- put_nowait ? queue size? timeout?
        #print('Saving event to db:', event)
        self.db_queue.put(event)

    def send_intensity_callback(self, event):
        #self.intensity_queue.put(event)
        future = self.intensity_executor.submit(ActionPicker.intensity_callback, self, event)
        print('future:', future)
        try:
            e_immediate = future.exception(timeout=0)
            if e_immediate is not None:
                raise e_immediate
        except TimeoutError:
            pass
        self.intensity_futures.append(future)

    # There's no real reason this needs to be a class method...
    def run_db(self, database_engine):
        print('Starting database interaction thread.')
        with Session(database_engine) as session:
            while True:
                print('Waiting for event from db queue.  Approx size: %i' % self.db_queue.qsize())
                event = self.db_queue.get()
                if event is None:
                    break
                self.save_event_to_db(session, event)
                session.flush()
                session.commit()
        print('Database thread ending')

    def save_event_to_db(self, session, event):
        from chord_frb_db.models import EventBeam, Event

        # Save L1 events
        l1_events = event.get('l1_events', [])
        l1_db_objs = []
        l1_payload = l1_events.database_payloads()
        for args in l1_payload:
            db_obj = EventBeam(**args)
            session.add(db_obj)
            session.flush()
            # Now we know the L1 event's unique id
            assert(db_obj.id is not None)
            l1_db_objs.append(db_obj)

        # Save L2 event
        l2_payload = event.database_payload()
        l2_db_obj = Event(**l2_payload)
        # Add L2 event to db
        session.add(l2_db_obj)

        # Now we can associate the L1 events with the L2 event.
        for e in l1_db_objs:
            l2_db_obj.beams.append(e)
        # DEBUG
        session.flush()
        print('Saved L2 event id', l2_db_obj.event_id)

    # # There's no real reason this needs to be a class method...
    # def run_intensity(self):
    #     print('Starting intensity callback thread.')
    #     while True:
    #         print('Waiting for event from intensity queue.  Approx size: %i' %
    #               self.intensity_queue.qsize())
    #         event = self.intensity_queue.get()
    #         if event is None:
    #             break
    #         self.intensity_callback(event)
    #     print('Intensity callback thread ending')

    def intensity_callback(self, event):
        print('Intensity callback: event_id', event.event_id)
