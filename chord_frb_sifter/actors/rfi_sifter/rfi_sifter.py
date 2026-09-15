"""
RFISifter Documentation

This class applies filter rules to grouped L1 events to cull any that may not
be astrophysical. Requires rfi_filter_rules.py and the pipeline YAML
configuration.
"""


__author__ = "CHIME FRB Group"
__version__ = "0.1"
__maintainer__ = "Shriharsh Tendulkar"
__developers__ = "Shriharsh Tendulkar"
__email__ = "shriharsh@physics.mcgill.ca"
__status__ = "Epsilon"

from chord_frb_sifter.actors.actor import Actor
from chord_frb_sifter import config

from . import rfi_filter_rules

class RFISifter(Actor):
    """
    Class for handling multi-beam RFI filtering

    RFISifter is a subclass of the ActorBaseClass for applying the rfi
    filter rules for multi-beam L1 events. The filter functions are defined
    in rfi_filter_rules.py and the configuration parameters are defined
    in config/pipeline_config.yaml

    """

    def __init__(self, threshold, filters, **kwargs):
        """ Initialize the RFISifter class worker.

        Parameters
        ----------
        threshold: float (between 0 to 10)
           Threshold L2 grade above which an event is considered to be
           astrophysical. This is interpreted as an estimate of the probability
           of the event being astrophysical.

        filters: List
           List of filters defined for calculating the probability.
           The dict includes the filter names and a dictionary of keyword
           arguments. The filter functions of the same names
           must be defined in rfi_filter_rules.py

        Returns
        -------
        RFISifter : RFISifter
            Object that has all the information for applying RFI sifting
            rules at the L2 stage.

        """

        super(RFISifter, self).__init__(**kwargs)
        self.threshold = threshold

        self.version = ""
        self.config_init = False

        self.filters = []  # functional forms of the filters
        self.rfi_filter_names = []
        self.rfi_filter_argstrs = []
        self.load_filters(filters)

    def __str__(self):
        return 'RFISifter'

    def _perform_action(self, event_group):
        """ Performs L2 RFI Sifting in science mode.

        Parameters are same as perform_action()

        See Also
        --------
        perform_action

        """
        for event in event_group.events:
            self.set_L2_grades(event)

            if event.rfi_grade_level2 < self.threshold:
                # set event_category to RFI (=3)
                print("RFI Sifter: Event at time %s -> RFI" % str(event.timestamp_utc))
                #event.is_rfi = True
                # set event_category to RFI (=3)
                event.set_rfi()
                print("RFI Sifter: Event at time %s -> RFI" % str(event.timestamp_utc))

            else:
                print("RFI Sifter: Event at time %s -> Astro" % str(event.timestamp_utc))
                # For now if its not RFI say its unknown (KSS can change later)
                event.is_rfi = False
        return [event_group]

    def load_filters(self, filters):
        """ Initializes the filters defined in the configuration file.

        This function loads the RFI filters, weights and arguments as
        defined in the config file and L2_rfi_filter_rules.py to RFISifter.
        """
        # set sifter threshold
        try:
            self.threshold = float(self.threshold)
        except Exception as e:
            import traceback; print(traceback.format_exc())
            self.threshold = 10.0
            #self.logger.critical(e)
            #self.logger.critical(
            #    "Unable to load. "
            #    + "Setting threshold to "
            #    + "{0}".format(self.threshold)
            #)

        # initialize filters
        for i, filt in enumerate(filters):
            # check if this filter definition exists
            if hasattr(rfi_filter_rules, filt[0]):
                filt[1].update({"config_dir": config.config_dir})
                try:
                    # test out the function
                    # func =getattr(rfi_filter_rules, filt[0])(filt[1])
                    func = getattr(rfi_filter_rules, filt[0])(filt[1])
                
                    if filter_works(func):
                        self.filters.append(getattr(rfi_filter_rules, filt[0])(filt[1]))
                        self.rfi_filter_names.append(filt[0])
                        self.rfi_filter_argstrs.append(filt[1])
                        print("Filter '{0}' loaded".format(filt[0]))
                    else:
                        print(
                            "Filter '{0}' ".format(filt[0])
                            + "is not executing correctly. "
                            + "Check filter definition and config!"
                        )
                
                except Exception as e:
                    import traceback; print(traceback.format_exc())
                    print(e)
                    # self.logger.critical(
                    #     "Filter '{0}' ".format(filt[0])
                    #     + "is not initializing properly from  "
                    #     + "L2_rfi_filters.py!"
                    # )

            else:
                print(
                    "Filter '{0}' ".format(filt[0])
                    + "is not defined in "
                    + "L2_rfi_filters.py!"
                )

        nof = len(self.rfi_filter_names)
        if nof:
            print(
                "Read {nof} filters".format(nof=len(self.rfi_filter_names))
            )
        else:
            print(
                "There are no filters defined! "
                + "The RFI sifter will pass everything!"
            )

        self.config_init = True

    def set_L2_grades(self, event):
        """
        Applies the individual RFI filters to the event in order.

        The calculated grades and metrics are saved in the L2Event.
        """
        ## Event grade was originally initialized to best L1 by the event grouper.
        ## The L2 probability was then multiplied to the existing number.
        ## However, L1 grade is already accounted for in the ML Classifier.
        ## This we should set it to 10 so as to not penalize the marginal events
        ## L1 grade ~7 or 8 events.

        event.rfi_grade_level2 = 10.0

        for i, function in enumerate(self.filters):
            # get the function call.

            # function adds the grade to the object itself.
            function.grade(event)

            # print(
            #     "Applied filter %s" % self.rfi_filter_names[i]
            #     + " to L2_event at time %s" % str(event.timestamp_utc)
            # )
        return 0


def filter_works(filter_function):
    """
    Tests that a filter minimally works without crashing.
    """
    from chord_frb_sifter.event import simulate_l2_event
    try:
        filter_function.grade(simulate_l2_event())
        print("Filter {} works.".format(filter_function.__name__))
        return True
    except Exception:
        import traceback; traceback.print_exc()
        print("Filter {} doesn't work!".format(filter_function.__name__))
        return False

# test
if __name__ == "__main__":
    rfisifter = RFISifter()
