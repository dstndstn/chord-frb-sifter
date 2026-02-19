from queue import Queue
from threading import Thread

from chord_frb_sifter.actors import Actor
from chord_frb_sifter.event import L1Event, L2Event

from chord_frb_db.models import get_next_event_id

from sqlalchemy.orm import Session

class EventIdStamper(Actor):
    def __init__(self, database_engine=None, **kwargs):
        super().__init__(**kwargs)
        self.event_id_queue = Queue(maxsize=10)
        self.db_thread = Thread(target=EventIdStamper.run_db, args=(self, database_engine), daemon=True)
        self.db_thread.start()

    def run_db(self, database_engine):
        print('database_engine:', database_engine)
        if database_engine.is_sqlite:
            # FAKE IT!
            eid = 1
            while True:
                self.event_id_queue.put(eid)
                eid += 1
        else:
            with Session(database_engine) as session:
                while True:
                    eid = get_next_event_id(session)
                    self.event_id_queue.put(eid)

    def _perform_action(self, event):
        l1 = event['l1_events']
        for i in range(len(l1)):
            eid = self.event_id_queue.get()
            l1['id'][i] = eid
        eid = self.event_id_queue.get()
        print('Stamped L2 event id', eid)
        event.event_id = eid
        return [event]
