import os
import importlib

import chord_frb_sifter.real_actions

def git_vesion(dirname):
    import subprocess
    p = subprocess.run(['git', 'describe', '--dirty'], cwd=dirname,
                       capture_output=True, check=True, text=True)
    return p.stdout

class ActionPicker(object):

    def __init__(self):
        fn = chord_frb_sifter.real_actions.__file__
        print('real_actions: file', fn)
        st = os.stat(fn)
        self.last_reload = st.st_mtime
        self.real = None

    def perform_action(self, event):
        # only check for updates periodically?

        fn = chord_frb_sifter.real_actions.__file__
        st = os.stat(fn)
        old_state = None
        old_real = None
        if st.st_mtime > self.last_reload:
            print('Reloading real_actions!')
            if self.real is not None:
                old_state = self.real.get_state()
                old_real = self.real
            importlib.reload(chord_frb_sifter.real_actions)
            print('Reloaded real_actions!')
            self.last_reload = st.st_mtime
            v = chord_frb_sifter.real_actions.version()
            print('Version:', v)
            self.real = None

        if self.real is None:
            self.real = chord_frb_sifter.real_actions.RealActionPicker(state=old_state,
                                                                       old_object=old_real)
            print('dir:', os.path.dirname(fn))
            gitver = git_vesion(os.path.dirname(fn))
            print('Git version:', gitver)

        return self.real.perform_action(event)




if __name__ == '__main__':
    import time
    ap = ActionPicker()
    for i in range(100):
        r = ap.perform_action(i)
        print('got', r)
        time.sleep(1.)
        
