"""
This actor identifies bright pulsar events based on a narrow window in DM and
a wide window in HA based on when we expect to detect known bright pulsars in 
sidelobes.
"""

from chord_frb_sifter.actors.actor import Actor
import numpy as np
from datetime import datetime

# import cfbm # Only if using cfbm for LST calculation

# Imports for astropy LST calculation
from astropy.time import Time
from astropy.coordinates import EarthLocation
from chord import Chord
import yaml
import os.path


class BrightPulsarSifter(Actor):
    """
    Identifies bright pulsar events based on known DM and HA windows.
    """

    def __init__(self, **kwargs):
        # Only a few pulsars that are bright enough to be detected in sidelobes.
        # If the list grows significantly, consider loading from config file or DB.
        self.bright_pulsars = {
            'B0329+54': {'dm': 26.8, 'dm_tol': 0.2, 'ha_window': (-3, 3), 'ra': 3.5498},
            'B0531+21': {'dm': 56.8, 'dm_tol': 0.2, 'ha_window': (-8, 8), 'ra': 5.5755},
            'B1933+16': {'dm': 158.6, 'dm_tol': 0.2, 'ha_window': (-2, 2), 'ra': 19.5966},
            # Add more known bright pulsars as needed
        }

        conf = yaml.load(open(os.path.join(
            os.path.dirname(__file__), 
            '../config', 
            'testChordTelescope.yaml'
        ),'r'), Loader=yaml.Loader)
        self.tele = Chord(conf["telescope"])

    def _perform_action(self, event):
  
        dm = event['dm']
        t = datetime.utcfromtimestamp(event["timestamp_utc"] / 1e6)

        # # Getting LST from ephem object for CHIME in cfbm. Bit clunky.
        # # Repalce w/ astropy or some CHORD utility?
        # cfbm.config.chime.date = t
        # lst = cfbm.config.chime.sidereal_time() * (12 / np.pi) # radians -> hours

        # astropy version using Chord object for telescope location
        loc = EarthLocation(
            lat=self.tele.origin_itrs_lat_deg, 
            lon=self.tele.origin_itrs_lon_deg
            )
        lst = Time(t, scale='utc', location=loc).sidereal_time('apparent').hour

        for pulsar, params in self.bright_pulsars.items():
            ha = lst - params['ra']
            ha = (ha + 12) % 24 - 12 # wrap HA

            if params['ha_window'][0] < ha < params['ha_window'][1]:
                if abs(dm - params['dm']) < params['dm_tol']:
                    event['is_bright_pulsar'] = True
                    event['bright_pulsar_name'] = pulsar
                    return [event]

        event['is_bright_pulsar'] = False
        return [event]
