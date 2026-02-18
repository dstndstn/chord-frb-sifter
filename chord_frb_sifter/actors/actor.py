
class Actor(object):

    def __init__(self, **kwargs):
        pass

    def perform_action(self, item):
        try:
            return self._perform_action(item)
        except:
            import traceback
            traceback.print_exc()
            raise

    def _perform_action(self, item):
        raise RuntimeError('not implemented')

    def shutdown(self):
        pass
