from chord_frb_sifter.actors import Actor
from chord_frb_sifter.event import L1Event, L2Event

class ActionPicker(Actor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _perform_action(self, event):
        # Log everything in db?
        event.send_to_db = True

        if event.is_rfi():
            return event

        if event.is_frb():
            pass

        
