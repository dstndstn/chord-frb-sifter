from flask import Flask, request, make_response
from .config import Config
from flask import render_template
import sys
import time
import json
import os
import numpy as np

from flask_sqlalchemy import SQLAlchemy

from chord_frb_db.models import Event, EventBeam, IntensityFile
from chord_frb_db.models import PirateConfig, BeamSNR

import sqlalchemy as sa
#from sqlalchemy import func, select
from sqlalchemy.orm import Session

from datetime import timedelta

#from flask import session
#print('Config:', Config)

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

@app.route('/d3')
def d3():
    return render_template('d3.html')

@app.route('/d3b')
def d3b():
    return render_template('d3b.html')

@app.route('/beam-max-snr/latest')
def beam_max_snr():
    latest = db.session.scalar(sa.select(sa.func.max(BeamSNR.timestamp)))
    print('result:', latest)
    #print(type(latest))
    #
    # result = db.session.execute(sa.select(BeamSNR, PirateConfig).where(
    #     BeamSNR.timestamp == latest, BeamSNR.pirate_config_id == PirateConfig.id)).first()
    # beamsnr, pirate = result

    after = latest - timedelta(seconds=10)

    latest = db.session.execute(sa.select(sa.func.max(BeamSNR.timestamp), BeamSNR.pirate_config_id)
                                .where(BeamSNR.timestamp > after)
                                .group_by(BeamSNR.pirate_config_id))
    print('result:', latest)

    beamsets = []
    timestamps = []
    timestamps_nano = []
    beam_ids = []
    beam_xs = []
    beam_ys = []
    beam_snrs = []
    
    for r in latest:
        #print(r)
        timestamp, pirate_config_id = r

        result = db.session.execute(sa.select(BeamSNR, PirateConfig).where(
            BeamSNR.timestamp == timestamp,
            BeamSNR.pirate_config_id==pirate_config_id,
            BeamSNR.pirate_config_id == PirateConfig.id))
        print('result:', result)
        for r2 in result:
            beamsnr, pirate = r2

        beamsets.append(pirate.beamset)
        timestamps.append(beamsnr.timestamp)
        timestamps_nano.append(str(int(beamsnr.timestamp.timestamp() * 1e9)))
        beam_ids.append(pirate.beam_id)
        beam_xs.append(pirate.beam_x)
        beam_ys.append(pirate.beam_y)
        beam_snrs.append(beamsnr.beam_snr)

    # Return JSON:
    return dict(beamsets=beamsets,
                timestamps=timestamps,
                timestamps_nano=timestamps_nano,
                beam_ids=beam_ids,
                beam_xs=beam_xs,
                beam_ys=beam_ys,
                beam_snrs=beam_snrs)

@app.route('/mwe')
def mwe():
    return render_template('mwe.html')

@app.route('/fake-prometheus')
def fake_prometheus():
    return {
    }

@app.route('/fake-prometheus/api/v1/rules', methods=['POST'])
def fake_Prometheus_rules():
    return {}

@app.route('/fake-prometheus/api/v1/label/__name__/values')
def fake_prometheus_label():
    return dict(status='success', data=['100','200'])

@app.route('/fake-prometheus/api/v1/metadata')
def fake_prometheus_metadata():
    return dict(status='success',
                data=dict(
                    test=[dict(type='gauge', help='Test help text', unit='meters')],
                    test2=[dict(type='gauge', help='Test2 help text', unit='kilograms')],
                    ))

@app.route('/fake-prometheus/api/v1/query', methods=['POST'])
def fake_prometheus_query():
    if request.method == 'POST':
        print('Fake prometheus query data:', request.form)
        if request.form.get('query') == 'test':

            tnow = time.time()

            metric = dict(__name__='test',
                          job='prometheus',
                          instance='chordfrb')
            v1 = dict(metric=metric, value=[ [tnow, 42.], [tnow+1, 43.] ])
            
            return dict(status='success',
                        data=dict(resultType='matrix',
                                  result=[v1])
                        )
    return {}

@app.route('/fake-prometheus/api/v1/query_range', methods=['GET', 'POST'])
def fake_prometheus_query_range():
    if request.method == 'POST':
        args = request.form
    else:
        args = request.args
    print('Fake prometheus query range:', args)
    if args.get('query') == 'test':
        tnow = time.time()
        tnow = np.round(tnow, decimals=3)
        
        metric = dict(__name__='test')
                      #job='prometheus',
                      #instance='chordfrb')
        v1 = dict(metric=metric, values=[ [tnow-10, "42."], [tnow, "43."] ])
        rtn = dict(status='success',
                   data=dict(resultType='matrix',
                             result=[v1])
                   )
        print('Returning JSON:', json.dumps(rtn))
        return rtn

    if args.get('query') == 'test2':
        tnow = time.time()
        #tnow = np.round(tnow, decimals=3)
        times = int(tnow) - 100 + np.arange(100)
        times = [int(x) for x in times]
        N = len(times)
        
        metric = dict(__name__='test2')
        #job='prometheus',
        #             instance='chordfrb')
        #v1 = dict(metric=metric, values=[ [tnow-10, "142."], [tnow, "143."] ])
        v1 = dict(metric=dict(__name__='test3a'),
                  values=[[t, "%.1f"%f] for t,f in zip(times, np.random.uniform(low=0, high=200, size=N))])
        rtn = dict(status='success',
                   data=dict(resultType='matrix',
                             result=[v1])
                   )
        print('Returning JSON:', json.dumps(rtn))
        return rtn

    if args.get('query') == 'test3':
        tnow = time.time()
        tnow = np.round(tnow, decimals=3)
        
        #job='prometheus',
        #             instance='chordfrb')
        times = int(tnow) - 100 + np.arange(100)
        times = [int(x) for x in times]
        N = len(times)
        
        v1 = dict(metric=dict(__name__='test3a'),
                  values=[[t, "%.1f"%f] for t,f in zip(times, np.random.uniform(low=0, high=200, size=N))])
        v2 = dict(metric=dict(__name__='test3b'),
                  #values=[ [tnow-10, "200", "300"], [tnow, "300", "400"] ])
                  #values=[ [tnow-10, "200"], [tnow, "300"] ])
                  #values=list(zip(times, np.random.uniform(low=0, high=200, size=N))))
                  values=[[t, "%.1f"%f] for t,f in zip(times, np.random.uniform(low=0, high=200, size=N))])
        v3 = dict(metric=dict(__name__='test3c'),
                  #values=[ [tnow-10, "1"], [tnow, "2"]])
                  #values=list(zip(times, np.random.uniform(low=0, high=200, size=N))))
                  values=[[t, "%.1f"%f] for t,f in zip(times, np.random.uniform(low=0, high=1, size=N))])
        rtn = dict(status='success',
                   data=dict(resultType='matrix',
                             result=[v1,v2,v3])
                   )
        print('Returning JSON:', json.dumps(rtn))
        return rtn
        
    return {}
    

@app.route('/beam-snr')
def beam_snr():
    # input: time range
    # returns JSON: list of tuples: (beamset, max-beam_snr-during-that-time-period array)
    # from querying the BeamSNR table
    pass

@app.route('/beamset/<int:beamset_id>')
def beamset():
    # input: beamset
    # output: most recent beam_x, beam_y, beam_id for that beamset
    # from the PirateConfig table
    pass

@app.route('/l1-events/<int:event_id>')
def l1_event_list(event_id):
    query = sa.select(EventBeam).filter_by(event_id=event_id)
    r = db.session.execute(query).scalars()
    print('r:', r)

    query = sa.select(Event).filter_by(event_id=event_id)
    event = db.session.execute(query).scalar_one()
    print('event:', event)

    fields = ['beam_id', 'snr', 'timestamp_utc', 'timestamp_fpga']
    return render_template('l1_event_list.html', event_id=event_id,
                           event=event, l1_events=r, fields=fields)

@app.route('/intensity-file-list/<int:event_id>')
def intensity_file_list(event_id):
    query = sa.select(IntensityFile).filter_by(event_id=event_id)
    r = db.session.execute(query).scalars()
    print('r:', r)
    r = list(r)

    # Sort -- assuming the filename pattern!
    # filenames are like event-00010833/frame_b11_t75.asdf 
    beams,times = [],[]
    for ifile in r:
        fn = ifile.filename
        fn = os.path.basename(fn)
        fn = fn.split('.')[0]
        words = fn.split('_')
        # yuck, man
        beam = words[1][1:]
        time = words[2][1:]
        beam = int(beam)
        time = int(time)
        beams.append(beam)
        times.append(time)
    I = np.lexsort((beams, times))
    ifiles = [r[i] for i in I]
    
    query = sa.select(Event).filter_by(event_id=event_id)
    event = db.session.execute(query).scalar_one()
    print('event:', event)

    fields = ['filename', 'status', 'error_message', 'view link']
    return render_template('intensity_file_list.html', event_id=event_id,
                           event=event, intensity_files=ifiles, fields=fields)

@app.route('/intensity-file-view/<int:event_id>/<path:filename>')
def intensity_file_view(event_id, filename):
    query = sa.select(IntensityFile).filter_by(filename=filename)
    ifile = db.session.execute(query).scalar_one()
    print('ifile:', ifile)
    print('app.config is', app.config, type(app.config))
    path = os.path.join(app.config['PIRATE_INTENSITY_DIR'], ifile.filename)

    import pylab as plt
    import tempfile
    
    import asdf
    with asdf.open(path) as af:

        so = af['scales_offsets']
        data = af['data']
        # load?
        data = data[:,:]

        with tempfile.NamedTemporaryFile(suffix='.png') as tf:
            plt.clf()
            plt.imshow(data, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.savefig(tf.name)
            plotdata = open(tf.name, 'rb')

            resp = make_response(plotdata)
            resp.headers['Content-Type'] = 'image/png'
            return resp

@app.route('/')
def event_list(): #(name=None):
    query = sa.select(Event).order_by(Event.event_id)#.desc())
    #order_by(Event.timestamp.desc())
    print('Query:', query)
    #print(dir(query))

    # #print('Count:', query.count())
    # #count_query = query.statement.with_only_columns([sa.func.count()]).order_by(None)
    # #count = q.session.execute(count_query).scalar()
    # count = sa.select(sa.func.count(Event.event_id))#.scalar()
    # print('Count:', type(count), count)
    # r = db.session.execute(count).scalar()
    # print(type(r), r)
    # n_events = r

    page = request.args.get("page")
    #print('page:', page)
    try:
        page = int(page)
    except:
        page = 1
    
    event_pager = db.paginate(query, page=page, per_page=20, error_out=False)
    events = event_pager.items
    
    #fields = [ 'event_id', 'timestamp', 'rfi_grade', 'total_snr', 'dm', 'ra', 'dec', 'nbeams', 'dm_ne2001', 'dm_ymw2016', 'flux', 'fluence', 'pulse_width' ]
    fields = [ 'event_id', 'timestamp', 'rfi_grade', 'best_snr', 'dm', 'ra', 'dec', 'nbeams', 'dm_ne2025', 'dm_ymw2016', 'n_intensity_files' ]#, 'flux', 'fluence', 'pulse_width' ]

    return render_template('event_list.html', event_pager=event_pager, events=events, fields=fields)

@app.route('/events.png')
def event_plot():
    from datetime import datetime

    query = sa.select(Event).order_by(Event.event_id.desc()).limit(1000)
    print('Query:', query)
    r = db.session.execute(query)#.scalar()
    print('Result:', r)

    xx = []
    yy = []
    cc = []

    for e in r:
        (e,) = e
        #print('  event:', e)
        d = datetime.fromtimestamp(e.timestamp)
        print('timestamp:', e.timestamp, '-> date', d)
        xx.append(d)
        #xx.append(e.timestamp)
        #xx.append(e.event_id)
        yy.append(e.dm)
        cc.append(e.rfi_grade)


    from io import BytesIO
    from matplotlib.figure import Figure
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    fig = Figure()
    ax = fig.subplots()
    scat = ax.scatter(xx, yy, c=cc, s=4, vmin=0, vmax=10, cmap='inferno')#copper')
    ax.set_yscale('log')
    ax.set_xlabel('Date')
    ax.set_ylabel('DM')
    ax.set_facecolor('0.6')
    #divider = make_axes_locatable(0)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    #fig.colorbar(scat, cax=cax, orientation='vertical')
    cb = fig.colorbar(scat, cax=None, ax=ax)
    cb.set_label('RFI grade')
    buf = BytesIO()
    fig.savefig(buf, format="png")
    #buf = buf.getbuffer()
    buf = buf.getvalue()

    resp = make_response(buf)
    resp.headers['Content-type'] = 'image/png'
    return resp

if __name__ == '__main__':
    from flask import request
    with app.test_request_context('/beam-max-snr/latest', method='GET'):
        beam_max_snr()

