import os
import importlib

import chord_frb_sifter.real_actions

'''

This is a demo of how we might use code reloading in, say, the Action Picker.

This ActionPicker class is a wrapper over a real implementation in
  real_actions.py

It watches the file real_actions.py on disk, and reloads the module
when the file is updated.

We keep a single instance of a RealActionPicker object; when the
module is reloaded we create a new one.
  
We _may_ want to pass state from the old version to the new version;
the new object gets passed the old object in its constructor.  I also
put in a get_state() method for the old object to form a state object
to pass forward.

If we end up using this, we will want some tooling or practice around
it, for example, always updating the code by doing a "git pull", never
by making modifications to the local code.

'''

# If you're paranoid, it would be possible that git is monkeying
# around with the filesystem at the same time as we're trying to
# reload the code and get its git tag, allowing an inconsistency.
def git_version(dirname):
    import subprocess
    p = subprocess.run(['git', 'describe', '--dirty'], cwd=dirname,
                       capture_output=True, check=True, text=True)
    return p.stdout

class ActionPicker(object):

    def __init__(self):
        # We're going to base our reload on the OS's last-modified timestamp on the
        # real_actions.py file.
        fn = chord_frb_sifter.real_actions.__file__
        print('real_actions: file', fn)
        st = os.stat(fn)
        self.last_reload = st.st_mtime
        # We'll initialize our real action picker object when the first event arrives.
        # Obviously we could fix this if there is some startup cost.
        self.real = None

    def perform_action(self, event):
        # only check for updates periodically?
        # Here we check the file timestamp on each event, which could take some time.
        fn = chord_frb_sifter.real_actions.__file__
        st = os.stat(fn)
        old_state = None
        old_real = None
        if st.st_mtime > self.last_reload:
            print('Reloading real_actions!')
            if self.real is not None:
                old_state = self.real.get_state()
                old_real = self.real
            # If there is, eg, a syntax error in "real_actions.py", the import can fail.
            # We can deal with that case no problem.
            success = False
            try:
                importlib.reload(chord_frb_sifter.real_actions)
                success = True
            except Exception as e:
                print('Failed to reload real_actions:', e)
                import traceback
                traceback.print_exc()

            self.last_reload = st.st_mtime
            if success:
                print('Reloaded real_actions!')
                v = chord_frb_sifter.real_actions.version()
                print('Version:', v)
                self.real = None

        if self.real is None:
            # We could stick a try/except around here to catch more bugs upon actually trying
            # to instantiate the RealActionPicker object...
            # ... I mean, we could also save the fallback object in case the perform_action()
            # fails, but at some point we have to give ourselves some credit, right?  Right?
            self.real = chord_frb_sifter.real_actions.RealActionPicker(state=old_state,
                                                                       old_object=old_real)
            gitver = git_version(os.path.dirname(fn))
            print('Git version:', gitver)

        return self.real.perform_action(event)

if __name__ == '__main__':
    import time
    ap = ActionPicker()
    for i in range(100):
        r = ap.perform_action(i)
        print('got', r)
        time.sleep(1.)
        
