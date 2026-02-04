
def version():
    return 44

#bombcat

class RealActionPicker(object):
    def __init__(self, state=None, old_object=None):
        print('RealActionPicker (version %s)' % version())
        if state is not None:
            self.count = state
        else:
            self.count = 0

    def perform_action(self, event):
        print('real action for', event)
        self.count += 1
        return 7, self.count

    def get_state(self):
        return self.count
