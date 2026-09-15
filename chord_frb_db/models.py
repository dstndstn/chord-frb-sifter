from datetime import datetime
import os

from sqlalchemy.orm import DeclarativeBase
from typing import List
from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Table, Column, Integer, String, SmallInteger, BigInteger, Double, REAL, ForeignKey, Float, Boolean, DateTime, ARRAY

import numpy as np

class Base(DeclarativeBase):
    pass

# Note -- float32 = REAL
#         float64 = Double
#         int16   = SmallInteger
#         int32   = Integer
#         int64   = BigInteger

class Event(Base):
    __tablename__ = 'event'
    event_id:  Mapped[int] = mapped_column(primary_key=True)
    # ... FRB time at infinite frequency?  What time format?  Unix: seconds since 1970.0, UTC
    ##... actually I think this is the time at the bottom of the frequency band
    ## (based on the frb_sifter.proto grpc call documentation)
    timestamp: Mapped[Optional[float]] = mapped_column(Double)
    is_rfi:    Mapped[bool] = mapped_column(default=False)
    # matches a known pulsar
    is_known_pulsar:  Mapped[bool] = mapped_column(default=False, server_default='false')
    # is a new event (FRB, incl repeats, new pulsar candidates)
    is_new_burst:     Mapped[bool] = mapped_column(default=False, server_default='false')
    # Is a verified new FRB (subset of is_new_burst)
    is_frb:           Mapped[bool] = mapped_column(default=False)
    is_repeating_frb:           Mapped[bool] = mapped_column(default=False, server_default='false')

    # CHIME/FRB's rfi_grade_level2
    # values are 0 to 10, with RFI:0 and Astrophysical:10.
    rfi_grade: Mapped[float] = mapped_column(REAL)
    # beam_activity: something like the number of beams that were lit up by this event = nbeams??
    beam_activity: Mapped[int] = mapped_column(SmallInteger)

    # ??
    best_beam: Mapped[Optional[int]] = mapped_column(SmallInteger)
    nbeams:    Mapped[int] = mapped_column(SmallInteger, default=0)

    beams:     Mapped[List['EventBeam']] = relationship(back_populates='event')
    best_snr:  Mapped[Optional[float]] = mapped_column(REAL)
    # multi-beam
    total_snr: Mapped[Optional[float]] = mapped_column(REAL)

    dm:        Mapped[Optional[float]] = mapped_column(REAL)
    dm_error:  Mapped[Optional[float]] = mapped_column(REAL)
    # in deg
    ra:        Mapped[Optional[float]] = mapped_column(REAL)
    ra_error:  Mapped[Optional[float]] = mapped_column(REAL)
    # in deg
    dec:       Mapped[Optional[float]] = mapped_column(REAL)
    dec_error: Mapped[Optional[float]] = mapped_column(REAL)

    dm_ne2025:  Mapped[Optional[float]] = mapped_column(REAL)
    dm_ymw2016: Mapped[Optional[float]] = mapped_column(REAL)
    
    spectral_index: Mapped[Optional[float]] = mapped_column(REAL)
    scattering:     Mapped[Optional[float]] = mapped_column(REAL)
    # in Jy-ms
    fluence:        Mapped[Optional[float]] = mapped_column(REAL)
    # in Jy
    flux:           Mapped[Optional[float]] = mapped_column(REAL)
    # in millisec
    pulse_width:    Mapped[Optional[float]] = mapped_column(REAL)

    # Best known source match
    known_id:       Mapped[Optional[int]] = mapped_column(ForeignKey('known_source.id'))
    known:     Mapped['KnownSource'] = relationship(back_populates='events')

    intensity_files:     Mapped[List['IntensityFile']] = relationship(back_populates='event')

    #def __repr__(self) -> str:
    #    return f"User(id={self.id!r}, name={self.name!r}, fullname={self.fullname!r})"

    def get_datetime_string(self):
        from datetime import datetime, UTC
        d = datetime.fromtimestamp(self.timestamp, tz=UTC)
        d = d.isoformat(sep=' ')[:23]
        return d

    @property
    def n_intensity_files(self):
        return len(self.intensity_files)

# Individual-beam measurements for a grouped multi-beam event
# Aka an "L1 event"
class EventBeam(Base):
    __tablename__ = 'event_beam'
    # WTF sqlite3 doesn't support big-integer primary keys?
    #id:   Mapped[int] = mapped_column(BigInteger, primary_key=True)
    id:   Mapped[int] = mapped_column(primary_key=True)

    beam_id: Mapped[int]

    snr:  Mapped[float] = mapped_column(REAL)

    timestamp_utc:  Mapped[float] = mapped_column(Double)
    timestamp_fpga: Mapped[int] = mapped_column(BigInteger)

    time_error: Mapped[float] = mapped_column(REAL)
    
    tree_index: Mapped[int]
    #spectral_index: Mapped[float] = mapped_column(REAL)
    #scattering_measure: Mapped[float] = mapped_column(REAL)

    rfi_grade: Mapped[int]
    rfi_mask_fraction: Mapped[float] = mapped_column(REAL)
    rfi_clip_fraction: Mapped[float] = mapped_column(REAL)

    # Arrays: https://docs.sqlalchemy.org/en/20/core/type_basics.html#sqlalchemy.types.ARRAY
    # snr_vs_dm:
    # snr_vs_tree_index:
    # snr_vs_spectral_index:

    dm:        Mapped[float] = mapped_column(REAL)
    dm_error:  Mapped[float] = mapped_column(REAL)

    ra:        Mapped[float] = mapped_column(REAL)
    ra_error:  Mapped[float] = mapped_column(REAL)

    dec:       Mapped[float] = mapped_column(REAL)
    dec_error: Mapped[float] = mapped_column(REAL)

    event_id: Mapped[Optional[int]] = mapped_column(ForeignKey("event.event_id"))
    event:     Mapped['Event'] = relationship(back_populates='beams')

class IntensityFile(Base):
    __tablename__ = 'intensity_file'
    filename:    Mapped[str] = mapped_column(String(1024), primary_key=True)
    succeeded:   Mapped[bool] = mapped_column(default=False, server_default='false')
    failed:      Mapped[bool] = mapped_column(default=False, server_default='false')
    error_message: Mapped[Optional[str]] = mapped_column(String(1024))

    event_id: Mapped[Optional[int]] = mapped_column(ForeignKey("event.event_id"))
    event:     Mapped['Event'] = relationship(back_populates='intensity_files')

    def status_color(self):
        if self.succeeded:
            return 'green'
        if self.failed:
            return 'red'
        return 'yellow'

    # HACK -- these two functions assume the filename pattern
    # like event-00010833/frame_b11_t75.asdf
    def get_beam_id(self):
        beam,_ = self.get_beam_id_and_time()
        return beam

    def get_time_chunk(self):
        _,time = self.get_beam_id_and_time()
        return time

    def get_beam_id_and_time(self):
        fn = self.filename
        # grab just the filename part --> "frame_b11_t75.asdf"
        fn = os.path.basename(fn)
        # drop the ".*" suffix --> "frame_b11_t75"
        fn = fn.split('.')[0]
        words = fn.split('_')
        # assume X_b(BEAM)_t(TIME)
        beam = words[1][1:]
        time = words[2][1:]
        beam = int(beam)
        time = int(time)
        return beam,time

class KnownSource(Base):
    __tablename__ = 'known_source'
    id:          Mapped[int] = mapped_column(primary_key=True)
    name:        Mapped[str] = mapped_column(String(64))
    # "Pulsar", "FRB" ?
    source_type: Mapped[str] = mapped_column(String(32))
    # reference, etc - human readable
    origin:      Mapped[str] = mapped_column(String(32))
    ra:          Mapped[float] = mapped_column(REAL)
    ra_error:    Mapped[Optional[float]] = mapped_column(REAL)
    dec:         Mapped[float] = mapped_column(REAL)
    dec_error:   Mapped[Optional[float]] = mapped_column(REAL)
    dm:          Mapped[float] = mapped_column(REAL)
    dm_error:    Mapped[Optional[float]] = mapped_column(REAL)
    # S400:       Mean flux density at 400 MHz (mJy)
    s400:        Mapped[Optional[float]] = mapped_column(REAL)
    s400_error:  Mapped[Optional[float]] = mapped_column(REAL)
    # S1400:       Mean flux density at 1400 MHz (mJy)
    s1400:       Mapped[Optional[float]] = mapped_column(REAL)
    s1400_error: Mapped[Optional[float]] = mapped_column(REAL)
    events: Mapped[List['Event']] = relationship(back_populates='known')

    def __str__(self):
        return ('KnownSource: %s, %s, RA,Dec %.4f,%.4f, DM %.2f' %
                (self.name, self.source_type, self.ra, self.dec, self.dm))

'''
Populated by the sifter from info it receives in the first message from a Pirate
instance.
'''
class PirateConfig(Base):
    __tablename__ = 'pirate_config'
    id:          Mapped[int] = mapped_column(primary_key=True)
    # beamset
    beamset:     Mapped[int]
    # start_time
    start_time:  Mapped[datetime]
    # xengine_config
    xengine_config: Mapped[str]  # = mapped_column(String(10240))
    # pirate_config
    pirate_config: Mapped[str] = mapped_column(nullable=True) # = mapped_column(String(10240))
    # beam_x
    beam_x: Mapped[List[float]] = mapped_column(ARRAY(REAL, dimensions=1, zero_indexes=True))
    # beam_y
    beam_y: Mapped[List[float]] = mapped_column(ARRAY(REAL, dimensions=1, zero_indexes=True))
    # beam_id
    beam_id: Mapped[List[int]] = mapped_column(ARRAY(Integer, dimensions=1, zero_indexes=True))

'''
This is populated by the sifter, by periodically dumping the max SN received per
beam, from each Pirate instance (beamset).  That is, in each reporting period
there will be ~28 BeamSNR entries, one per beamset, with the same timestamp.
If a Pirate does not report, it gets no entry(?)
'''
class BeamSNR(Base):
    __tablename__ = 'beam_snr'
    id:          Mapped[int] = mapped_column(primary_key=True)
    # PirateConfig
    pirate_config_id: Mapped[int] = mapped_column(ForeignKey("pirate_config.id"))
    pirate_config:    Mapped['PirateConfig'] = relationship()#back_populates='events')

    # timestamp
    timestamp: Mapped[datetime]# = mapped_column(DateTime)
    # beam_snr
    beam_snr: Mapped[List[float]] = mapped_column(ARRAY(REAL, dimensions=1, zero_indexes=True))

class DumbTest(Base):
    __tablename__ = 'dumb_test'
    id:   Mapped[int] = mapped_column(primary_key=True)
    #id:   Mapped[int] = mapped_column(BigInteger, primary_key=True)
    x: Mapped[int]

#class EventId(Base):
#    id:
from sqlalchemy.schema import Sequence

event_id_sequence = Sequence('event_id_sequence', start=1, metadata=Base.metadata)
def get_next_event_id(session):
    return session.scalar(event_id_sequence)

if __name__ == '__main__':
    import os
    from sqlalchemy import create_engine

    # db_url = 'postgresql+psycopg2://frb:PASSWORD@localhost:5432/frb'
    # db_pass = os.environ.get('CHORD_FRB_DB_PASSWORD', 'PASSWORD')
    # db_url = db_url.replace('PASSWORD', db_pass)

    #db_url = "sqlite+pysqlite:///:memory:"

    #db_url = "sqlite+pysqlite:///db.sqlite3"
    #engine = create_engine(db_url, echo=True)

    from chord_frb_db.utils import get_db_engine
    engine = get_db_engine()
    
    Base.metadata.create_all(engine)

    from sqlalchemy.orm import Session
    import sqlalchemy as sa

    with Session(engine) as session:
        events = session.execute(sa.select(Event)).scalars()
        for e in events:
            print(e.event_id)
    import sys
    sys.exit(0)

    with Session(engine) as session:
        d = DumbTest(x=42)
        session.add(d)
        session.flush()
        print('d id', d.id)
    
        e = EventBeam(dm=42, beam_id=900, snr=20, timestamp_utc=1400000, timestamp_fpga=10000,
                      time_error=0., tree_index=0, rfi_grade=0, rfi_mask_fraction=0.,
                      rfi_clip_fraction=0., dm_error=0.1, ra=0., dec=0., ra_error=0., dec_error=0.)
        session.add(e)
        session.flush()
        print('e id', e.id)

        #s = event_id_sequence()
        #s = session.execute(event_id_sequence)
        s = session.scalar(event_id_sequence)
        print('s:', s)
