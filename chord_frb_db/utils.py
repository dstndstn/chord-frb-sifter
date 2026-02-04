def get_db_engine():
    import os
    from sqlalchemy import create_engine
    from chord_frb_db.models import Base
    db_url = os.environ.get('CHORD_FRB_DB_URL', 'sqlite+pysqlite:///db.sqlite3')
    #print('Using database URL:', db_url)
    #engine = create_engine(db_url, echo=True)
    engine = create_engine(db_url, echo=False)
    if 'sqlite' in db_url:
        # Make sure database tables exist
        Base.metadata.create_all(engine)
    return engine
