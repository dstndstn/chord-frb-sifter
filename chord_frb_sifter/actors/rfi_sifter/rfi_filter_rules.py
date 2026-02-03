"""
This module defines the RFI filter functions for L2 level.
The function call is always function(L2_object, **kwargs)

The grade of the L2 event is a scaled probability from 0 to 10 of the event
being astrophysical. 10 is 100% astrophysical. Eventually, we will use the L1
grade of the higest SNR beam as the starting point. This will be further
multiplied with the probabilities from the L2 rfi filter.
The function updates the grade of the L2 object by multiplying the probabilities
with the newly calculated probability.

All filters requested in the filter_rules_config file must be present here.
This is checked when the filter_rules_config file is read in.

An unused template function is defined below.
"""


from builtins import object
from builtins import str

__author__ = "CHIME FRB Group"
__version__ = "0.1"
__maintainer__ = "Shriharsh Tendulkar"
__developers__ = "Shriharsh Tendulkar"
__email__ = "shriharsh@physics.mcgill.ca"
__status__ = "Epsilon"


import joblib
import numpy as np
import yaml

from .rfi_features import (
    check_filter_labels_valid,
    print_available_features,
    features_to_vector,
    event_to_features,
)


class FunctionTemplate(object):
    """ Template for a filter class
    Should have init, reload and run methods.
    """

    def __init__(self, **kwargs):
        """ Overload this """

    def grade(self, event):
        """
        Grades the event.
        Overload this

        Returns grade
        """

    def reload(self, **kwargs):
        """ Overload this """


def is_valid_beam(beam_no):
    """
    Returns true if beam_no is in the valid range
    """

    # print type(beam_no), beam_no
    # assert type(beam_no) is int or type(beam_no) is float

    return (
        ((beam_no >= 0) and (beam_no < 256))
        or ((beam_no >= 1000) and (beam_no < 1256))
        or ((beam_no >= 2000) and (beam_no < 2256))
        or ((beam_no >= 3000) and (beam_no < 3256))
    )

class ML_Classifier(object):
    """
    Machine Learning Classifier based on scikit-learn

    This imports a pre-trained classifier saved with scikit joblib and runs
    the L2_event through it. The run function creates the test vector from the
    L2_event properties. This format MUST match the vector used to train the
    classifier.

    This classifier gives p(A | beam_pattern, snr, snr_ratio etc). It does not
    include the prior probabilities of the event being astrophysical or RFI.
    For example, if more RFI is expected at a particular time of day, we should
    add that information as a separate function.
    """

    def __init__(self, kwargs):
        """
        Initializes by trying to load the classifier.

        """
        assert "clf_filename" in kwargs
        assert "labels_filename" in kwargs
        assert "config_dir" in kwargs

        self.config_init = False
        self.clf_filename = ""
        self.clf = None
        self.labels_filename = ""
        self.labels = None
        self.__name__ = "ML Classifier"
        self.reload(kwargs)
        self.n_labels = len(self.labels)
        print("done")

    def grade(self, event):
        """
        Calculate the grade for ML filter.

        Parameters:
        event : L2_event object type.
        """
        features = event_to_features(event)
        grade = self.clf.predict_proba(
            features_to_vector(features, self.labels).reshape(1, self.n_labels)
        )[0, 1]

        if hasattr(event,"rfi_grade_level2") and isinstance(
            event.rfi_grade_level2, (float, int)
        ):  # Some grade already exists
            event.rfi_grade_level2 = event.rfi_grade_level2 * grade
        else:  # no grade exists, scale from 0 to 10.
            event.rfi_grade_level2 = 10.0 * grade

        print(event.rfi_grade_level2)
        # add to the dictionary (grade_metrics)
        # if the keys already exist then they are updated.
        if hasattr(event,"rfi_grade_metrics_level2") and isinstance(event.rfi_grade_metrics_level2, dict):
            event.rfi_grade_metrics_level2.update({"ML_Classifier_grade": grade})
        else:
            event.rfi_grade_metrics_level2 = {"ML_Classifier_grade": grade}
        #event.futures.rfi_features = features

        return event

    def reload(self, kwargs):
        """
        (Re)-initializes by trying to load the classifier.

        """

        # self.logger = logger

        clf_filename = kwargs["config_dir"] + kwargs["clf_filename"]
        labels_filename = kwargs["config_dir"] + kwargs["labels_filename"]
        print(clf_filename)
        print(labels_filename)
        try:
            self.clf = joblib.load(clf_filename)
            self.clf_filename = clf_filename
            print(
                ("Successfully loaded %s from file %s" % (str(self.clf), clf_filename))
            )

            with open(labels_filename) as fd:
                self.labels = yaml.safe_load(fd)
            self.labels_filename = self.labels_filename

            print(
                ("Successfully loaded %s from file %s" % (self.labels, labels_filename))
            )

            self.labels_filename = self.labels_filename

            print(
                "Successfully loaded %s from file %s" % (self.labels, labels_filename)
            )
            assert check_filter_labels_valid(self.labels)
            self.config_init = True

        except Exception as e:
            import traceback; print(traceback.format_exc())
            self.clf = None
            #           self.scaler = None
            print(
                (
                    "Could not load classifier or labels from file {}, {}!".format(
                        clf_filename, labels_filename
                    )
                )
            )
            raise IOError
        return 0

class ML_Classifier_Ensemble(object):
    """
    Machine Learning Classifier based on scikit-learn

    This imports a pre-trained classifier saved with scikit joblib and runs
    the L2_event through it. The run function creates the test vector from the
    L2_event properties. This format MUST match the vector used to train the
    classifier.

    This classifier gives p(A | beam_pattern, snr, snr_ratio etc). It does not
    include the prior probabilities of the event being astrophysical or RFI.
    For example, if more RFI is expected at a particular time of day, we should
    add that information as a separate function.
    """

    def __init__(self, kwargs):
        """
        Initializes by trying to load the classifier.

        """
        assert "clf_filename" in kwargs
        assert "labels_filename" in kwargs
        assert "config_dir" in kwargs
        assert "svm_filename" in kwargs
        assert "svm_labels_filename" in kwargs
        assert "clf_scaler" in kwargs

        self.config_init = False
        self.clf_filename = ""
        self.svm_filename = ""
        self.svm = None
        self.scaler = None
        self.svm_labels_filename = ""
        self.clf_scaler = ""
        self.clf = None
        self.labels_filename = ""
        self.labels = None
        self.__name__ = "ML Classifier"
        self.reload(kwargs)
        self.n_labels = len(self.labels)
        print("done")

    def grade(self, event):
        """
        Calculate the grade for ML filter.

        Parameters:
        event : L2_event object type.
        """
        features = event_to_features(event)
        vector_clf = self.scaler.transform(features_to_vector(features, self.labels).reshape(1, self.n_labels))
        vector_svm = features_to_vector(features, self.svm_labels).reshape(1, len(self.svm_labels))
        try:
            grade = self.ensemble_inference(self.clf, vector_clf, self.svm, vector_svm)
            #print('grading using ensemble model')
        except Exception as e:
            print("failed grading")
            import traceback; traceback.print_exc()

        print(f"sucessfully graded: {grade}")
        if hasattr(event,"rfi_grade_level2") and isinstance(
            event.rfi_grade_level2, (float, int)
        ):  # Some grade already exists
            if event.rfi_grade_level2!=10:
                print(f"not accepting new grade: {grade},{event.rfi_grade_level2}")
            event.rfi_grade_level2 = event.rfi_grade_level2 * grade

        else:  # no grade exists, scale from 0 to 10.
            event.rfi_grade_level2 = 10.0 * grade

        # add to the dictionary (grade_metrics)
        # if the keys already exist then they are updated.
        if hasattr(event,"rfi_grade_metrics_level2") and isinstance(event.rfi_grade_metrics_level2, dict):
            event.rfi_grade_metrics_level2.update({"ML_Classifier_grade": grade})
        else:
            event.rfi_grade_metrics_level2 = {"ML_Classifier_grade": grade}
        event.futures.rfi_features = features
        return event

    def reload(self, kwargs):
        """
        (Re)-initializes by trying to load the classifier.

        """

        # self.logger = logger
        print(kwargs)
        clf_filename = kwargs["config_dir"] + kwargs["clf_filename"]
        labels_filename = kwargs["config_dir"] + kwargs["labels_filename"]
        svm_filename = kwargs["config_dir"] + kwargs["svm_filename"]
        svm_labels_filename = kwargs["config_dir"] + kwargs["svm_labels_filename"]
        clf_scaler = kwargs["config_dir"] + kwargs["clf_scaler"]
        try:
            self.clf = joblib.load(clf_filename)
            self.svm = joblib.load(svm_filename)
            self.scaler = joblib.load(clf_scaler)

            self.clf_filename = clf_filename
#            print(
#                ("Successfully loaded %s from file %s" % (str(self.scaler), clf_scaler))
#            )
            print(
                ("Successfully loaded %s from file %s" % (str(self.clf), clf_filename))
            )
#            print(
#                ("Successfully loaded %s from file %s" % (str(self.svm), svm_filename))
#            )
            with open(labels_filename) as fd:
                self.labels = yaml.safe_load(fd)
            with open(svm_labels_filename) as fd:
                self.svm_labels = yaml.safe_load(fd)

            print(
                ("Successfully loaded %s from file %s" % (self.labels, labels_filename))
            )

            self.labels_filename = self.labels_filename

            print(
            "Successfully loaded %s from file %s" % (self.labels, labels_filename)
            )
            assert check_filter_labels_valid(self.labels)
            assert check_filter_labels_valid(self.svm_labels)

            self.config_init = True

        except IOError:
            self.clf = None
            #           self.scaler = None
            print(
                (
                    "Could not load classifier or labels from file {}, {}!".format(
                        clf_filename, labels_filename
                    )
                )
            )
            raise IOError
        return 0

    def ensemble_inference(self,xgboost, xgboost_data, svm, svm_data):
        # compute the predictions of the SVM model
        svm_prediction = svm.predict_proba(svm_data)

        # We compute the predictions for the XGBoost model
        xgboost_prediction = xgboost.predict_proba(xgboost_data)
        # Apply a constant to weight the classes 1:4 ratio with

        xgboost_prediction =xgboost_prediction*[1,4]
        # We renormalize this
        axis_sum = np.sum(xgboost_prediction, axis=1)
        # Apply the normalization
        xgboost_prediction[:,0] = xgboost_prediction[:,0]/axis_sum
        xgboost_prediction[:,1] = xgboost_prediction[:,1]/axis_sum

        # We collect the predicted class in list
        prediction_classes = []
        # We compute the SVM prediction ratios
        ratios = svm_prediction[:,1]/svm_prediction[:,0]

        inverse_ratios = svm_prediction[:,0]/svm_prediction[:,1]
        if np.sum(xgboost_prediction)!=1:
            print('not 1')
            print(xgboost_prediction)
        #return the grade
        for i in range(xgboost_data.shape[0]):
            # SVM OVERRIDE CONDITION
            if ratios[i] > 4:
                return svm_prediction[i,1]
            # Regular XGBoost conditions
            else:
                return xgboost_prediction[i,1]

class AntiCoincidence(object):
    """
    Anticoincidence filter class.
    Needs incoherent beam input. Not currently used.

    """

    def __init__(self, time_window):
        """
        Init function for AntiCoincidence filter.

        Parameters:

        time_window : float
        Time window in seconds for anti-coincidence veto. Must be positive.
        """

        self.time_window = 0.0
        self.config_init = False

        assert isinstance(time_window, float)
        assert time_window > 0

        self.time_window = time_window

        self.config_init = True

    def grade(self, event):
        """
        Calculate the grade for AntiCoincidence filter.

        Parameters:
        L2_event : L2_event object type.

        """
        ## do stuff to calculate a grade
        grade = 1.0

        if hasattr(event,"rfi_grade_level2") and isinstance(event.rfi_grade_level2, (float, int)) and (
            event.rfi_grade_level2 > 0
        ):
            # Some grade already exists
            event.rfi_grade_level2 = event.rfi_grade_level2 * grade
        else:  # no grade exists.
            event.rfi_grade_level2 = grade * 10.0

        # add to the dictionary (grade_metrics)
        # if the keys already exist then they are updated.
        if hasattr(event,"rfi_grade_metrics_level2") and isinstance(event.rfi_grade_metrics_level2, dict):
            event.rfi_grade_metrics_level2.update(
                {"anti_coincidence" + "_grade": grade}
            )
        else:
            event.rfi_grade_metrics_level2 = {"anti_coincidence" + "_grade": grade}

        return event

    def reload(self, time_window):
        """
        Reload the time_window variable for the AntiCoincidence filter class.

        Parameters:

        time_window : float
        Time window in seconds for anti-coincidence veto. Must be positive.

        """
        assert isinstance(time_window, float)
        assert time_window > 0

        self.time_window = time_window

        self.config_init = True

        return 0


class High_SNR_Override(object):
    """ 
    Allows very high SNR events to be bypassed by setting L2_grade to 10.
    """

    def __init__(self, kwargs):
        """
        Initializes by trying to load the threshold.

        """
        assert "snr_threshold" in kwargs
        assert "only_coherent" in kwargs

        self.config_init = False
        self.snr_threshold = 1000.0  # set very high in case of fault.
        self.only_coherent = False
        self.__name__ = "High_SNR_Override"

        self.reload(kwargs)

    def grade(self, event):
        """
        Sets the grade to 10 if the coherent SNR is above snr_threshold.
        If only_coherent is false then the incoherent SNR is also considered.
        """

        l1_events = event.l1_events
        num_beams = len(l1_events)
        num_incoherent_beams = np.count_nonzero(l1_events["is_incoherent"])
        filt = np.where(l1_events["is_incoherent"] == False)

        if num_beams == num_incoherent_beams:
            if self.only_coherent:
                return event  # without modification
            else:
                max_snr = np.max(l1_events["snr"])
        elif num_beams > num_incoherent_beams:
            if self.only_coherent:
                max_snr = np.max(l1_events["snr"][filt])
            else:
                max_snr = np.max(l1_events["snr"])
        else:  # this should NEVER happen
            print(
                "For num_beams < num_incoherent_beams for event! This should never happen."
            )
            print(("{}".format(event)))

        if max_snr < self.snr_threshold:
            return event  # without modification
        else:
            event.rfi_grade_level2 = 10.0

            # add to the dictionary (grade_metrics)
            # if the keys already exist then they are updated.
            if hasattr(event,"rfi_grade_metrics_level2") and isinstance(event.rfi_grade_metrics_level2, dict):
                event.rfi_grade_metrics_level2.update({"High_SNR_Override_grade": 10.0})
            else:
                event.rfi_grade_metrics_level2 = {"High_SNR_Override_grade": 10.0}

        return event

    def reload(self, kwargs):
        """
        Loads the snr_threshold and the only_coherent flag.
        """

        self.snr_threshold = float(kwargs["snr_threshold"])
        self.only_coherent = bool(kwargs["only_coherent"])
        self.config_init = True

        return 0


class Sanity_Check(object):
    """ 
    A final sanity check for grades. This is to mitigate the effects of 
    the ML classifier misbehaving. This applies hard cuts to the mean 
    L1 grade and beam activity

    """

    def __init__(self, kwargs):
        """ Overload this """
        assert "mean_L1_grade_threshold" in kwargs
        assert "beam_activity_threshold" in kwargs

        self.config_init = False
        self.mean_L1_grade_threshold = 5.0  # set defaults
        self.beam_activity_threshold = 25
        self.__name__ = "Sanity_Check"

        self.reload(kwargs)

    def grade(self, event):
        """
        Sets the grade to zero if the mean_L1_grade is below the 
        configured threshold or if the beam activity is above the 
        configured threshold.
        """
        l1_events = event.l1_events

        num_coh = np.count_nonzero(l1_events["is_incoherent"] == False)
        filt = np.where(l1_events["is_incoherent"] == False)

        if num_coh >= 1:
            mean_L1_grade = np.average(
                l1_events["rfi_grade_level1"][filt], weights=l1_events["snr"][filt]
            )
        else:
            mean_L1_grade = np.mean(l1_events["rfi_grade_level1"])

        if (
            mean_L1_grade < self.mean_L1_grade_threshold
            or event.beam_activity > self.beam_activity_threshold
        ):
            event.rfi_grade_level2 = 0.0

            # add to the dictionary (grade_metrics)
            # if the keys already exist then they are updated.
            if hasattr(event,"rfi_grade_metrics_level2") and isinstance(event.rfi_grade_metrics_level2, dict):
                event.rfi_grade_metrics_level2.update({"Sanity_Check_grade": 0.0})
            else:
                event.rfi_grade_metrics_level2 = {"Sanity_Check_grade": 0.0}

        return event

    def reload(self, kwargs):
        """ Overload this """
        self.config_init = False
        self.mean_L1_grade_threshold = float(kwargs["mean_L1_grade_threshold"])
        self.beam_activity_threshold = float(kwargs["beam_activity_threshold"])
