import numpy as np

from chord_frb_sifter.actors import Actor

def radec_to_xyz(ra, dec):
    decrad = np.deg2rad(dec)
    z = np.sin(decrad)
    cosdec = np.cos(decrad)
    rarad = np.deg2rad(ra)
    x = np.cos(rarad) * cosdec
    y = np.sin(rarad) * cosdec
    return np.array((x, y, z)).T

def xyz_to_radec(xyz):
    if len(xyz.shape) == 1:
        rs,ds = xyz_to_radec(xyz[np.newaxis,:])
        return (rs[0], ds[0])
    assert(xyz.shape[1] == 3)
    ra = np.arctan2(xyz[:,1], xyz[:,0])
    ra += (ra < 0) * 2.*np.pi
    norm = np.sqrt(np.sum(xyz**2, axis=1))
    dec = np.arcsin(xyz[:,2] / norm)
    return np.rad2deg(ra), np.rad2deg(dec)

class SimpleLocalizer(Actor):
    '''
    Averages the "L1" individual events, weighting by S/N, to get the "L2" localization.
    '''
    def __init__(self, **kwargs):
        pass

    def _perform_action(self, item):
        #print('SimpleLocalizer: got', item)
        l1_events = item['l1_events']
        if len(l1_events) == 1:
            item['average_dra'] = item['max_beam_dra']
            item['average_ddec'] = item['max_beam_ddec']
            return item
        else:
            print('Averaging L1 positions to get L2 position')
            #print(item)
            # Take the SNR-weighted average position -- in xyz unit-sphere coords
            xyz_snr = 0.
            sum_snr = 0.
            for e in l1_events:
                print('  snr %.1f: dra,ddec %.2f, %.2f' % (e['snr'], e['beam_dra'], e['beam_ddec']))
                xyz = radec_to_xyz(e['beam_dra'], e['beam_ddec'])
                snr = e['snr']
                xyz_snr = xyz_snr + xyz * snr
                sum_snr += snr
            xyz_snr /= sum_snr
            dra,ddec = xyz_to_radec(xyz_snr)
            print('  -> average dra,ddec %.2f, %.2f' % (dra,ddec))
            item['average_dra'] = dra
            item['average_ddec'] = ddec
            return item
