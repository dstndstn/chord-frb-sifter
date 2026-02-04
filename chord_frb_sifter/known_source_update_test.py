'''

This is an experiment about doing async updates from a database, eg, for the Known Source matcher.

'''

from chord_frb_db.models import KnownSource
from chord_frb_db.utils import get_db_engine
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.sql.expression import func

import concurrent.futures as cf
import time

def get_known_sources(do_print=False):
    known_sources = []
    engine = get_db_engine()
    with Session(engine) as session:
        if do_print:
            print('Known sources:', session.execute(select(func.count(KnownSource.id))).scalar_one())
        query = select(KnownSource)
        result = session.execute(query)
        for i,r in enumerate(result):
            (r,) = r
            known_sources.append(r)
            if do_print:
                print('  %i:' % (i+1), r)
    return known_sources

def main():
    executor = cf.ThreadPoolExecutor(max_workers=3)

    known_sources = get_known_sources(do_print=True)
    t_last_update = time.time()
    update_status = None

    while True:
        time.sleep(1.)
        tnow = time.time()
        if (tnow - t_last_update > 5) and (update_status is None):
            # Start a new update.
            print('Launching a database query to update Known Sources...')
            update_status = executor.submit(get_known_sources, ())
        print('Update status:', update_status)
        if update_status is not None:
            if update_status.done():
                try:
                    ks = update_status.result()
                    print('Updating known sources:', len(ks))
                    known_sources = ks
                    t_last_update = tnow
                except Exception as e:
                    print('Known Source database query failed:', e)
                    import traceback
                    traceback.print_exc(e)
                update_status = None

if __name__ == '__main__':
    main()
