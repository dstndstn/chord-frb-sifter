# "Sifting" pipeline for the CHORD/Fast Radio Burst project

## Context / background

The CHORD FRB system is made up of the "FRB Search" stage (running on
the FRB cluster), and the "FRB Sifting" pipeline (this code).

The "FRB Search" stage receives intensity inputs from the correlator /
X-engine and runs the "pirate" search to generate multiple
DM-versus-time maps.  It then detects peaks and produces "events" that
get sent to the "FRB Sifting" pipeline.

Some of the names or aliases come from the CHIME world.  In CHIME, the
correlator / X-engine is called "level 0" or L0.  The FRB search
system is called "L1".  The part that goes from DM-versus-time maps to
events is called "L1b".  L1b sends events to the "L2/L3" system, which
sends events to the "L4" or database system.  Because of reasons,
there is another database system called "frb_master" that stores a
subset of events.

In the CHIME terminology, this repository corresponds to the "L4" system.
The "L2/L3" equivalent is in https://github.com/chord-observatory/frb-l2l3/tree/test

(FIXME - rename / import into a CHORD repo!)


## Architecture

The `chord_frb_db` module provides a database interface for CHORD/FRB events.
It uses the `sqlalchemy` package for database interactions, and the `alembic` package to
handle changes to the data model over time.

In our current testing setup, the actual database is `postgres`.

FIXME - rename

The `web` directory contains a very basic web service that can read
information from the database and display it in a web site.  It uses
the `flask` web framework.

In our current testing setup, the web service is fronted by the
`nginx` web server.  The flask web app is run as an `uwsgi` service
that runs via `systemd`.


For testing purposes, we have captured a large number of events from
the CHIME/FRB system.  These can be ingested into the database using a
prototype of our CHORD/FRB Sifting pipeline -- which is currently
basically just a subset of the CHIME/FRB L2/L3 system, but calling the
core functionality directly rather than using the (perhaps overly)
elaborate framework used in CHIME/FRB.


## Setup / initialization:

```
# Install CHIME/FRB beam model
python -m pip instal cfbm
# Get data for cfbm
python -c "from cfbm.bm_data import get_data; get_data()"

# Install other packages required
python -m pip install threadpoolctl
```

## Scripts

* load-chime-events.py

This script reads saved CHIME/FRB events, sends them through a pipeline of 'actors', and loads the results into our CHORD/FRB database.

## Simplification

As part of the CHORD/FRB effort, we want to simplify everything.  In this spirit, I have started copying actors from the old L2/L3 repo into
the `chord_frb_sifter` directory.  So far, have only done the `BeamBuffer`!


## Actors

The `BeamBuffer` class used to do several things:
* track "exposure" - which beams were reporting during each 10-second interval, written to one file per day
* send beam status (dead/alive) to frb-master
* send a heartbeat to frb-master
Now, it just buffers events from a single chunk of data, waiting for data from all beams to arrive; when the next chunk of data arrives, it flushes
the buffer.

## Open design questions

* how do we want to track liveness / effective exposure time?
* does the FRB Search system send null results to the FRB Sifter if no event is found? (eg, for the purposes of tracking liveness / exposure time)
* early triggers - need a new actor!





# OLD STUFF BELOW HERE

# Notes about Sqlalchemy & Alembic Setup

Ubuntu 24.04 packages are older than we want (v2.0+) -- use `pip` to install new versions:

```
sudo pip install sqlalchemy --break-system-packages
sudo pip install alembic --break-system-packages
sudo pip install flask_sqlalchemy --break-system-packages
```

Along with
```
sudo apt install postgresql-common libpq-dev
sudo pip install psycopg2 --break-system-packages
```

The database URL is specified in the file `chord_frb_db/alembic.ini` as the `sqlalchemy.url` variable.

If you set the environment variable `CHORD_FRB_DB_PASSWORD`, that will get plugged into
the default database URL.

If you set the environment variable `CHORD_FRB_DB_URL`, that database URL will get used instead.  For example, for local `sqlite3` testing,

```
export CHORD_FRB_DB_URL=sqlite+pysqlite:///db.sqlite3
```


# Notes about alembic in normal use

```
cd chord_frb_db && alembic revision --autogenerate -m "add initial models"
cd chord_frb_db && alembic upgrade head
```

You should then `git add` the `chord_frb_db/alembic/version/*.py` files.


# Notes about flask

Good tutorial

https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world

flask --app web.webapp run --reload
 (--debug)