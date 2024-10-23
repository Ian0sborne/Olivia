import numpy  as np
import tables as tb

from invisible_cities.core import tbl_functions as tbl


class Histogram:
    def __init__(self, title, bins, labels, scale, values=None):
        """This class represents a histogram with is a parameter holder that
        contains data grouped by bins.

        Attributes
        ----------
        title     : str
        Histogram title.
        bins      : sequence
        Histogram binning.
        data      : np.array
        Accumulated entries on each bin.
        out_range : np.array
        Accumulated counts out of the bin range.
        Values are n-dim arrays of lenght 2
        (first element is underflow, second oveflow).
        errors    : np.array
        Assigned uncertanties to each bin.
        labels    : sequence
        Axis labels.
        scale     : str
        Scale of the y axis. This can take one of the
        scale values allowed by matplotlib: linear, log ...

        Parameters
        ----------
        bins   : sequence
        List containing the histogram binning.
        values : np.array, optional
        Initial values. If not passed, the initial bin content is set to zero.
        """
        self.title     = title
        self.bins      = bins
        self.data      = self.init_from_bins()
        self.errors    = self.init_from_bins()
        self.out_range = np.zeros(shape=(2, len(self.bins)))
        self.labels    = labels
        self.scale     = scale

        if values is not None:
            self.fill(np.asarray(values))

    def init_from_bins(self):
        """Encapsulation for histogram initialization to 0
        """
        return np.zeros(shape=tuple(len(x) - 1 for x in self.bins))

    def fill(self, additive, data_weights=None):
        """Given datapoints, bins and adds thems to the stored bin content.

        Parameters
        ----------
        additive     : np.array or sequence
        Data to fill the histogram.
        data_weights : np.array or sequence
        Weights of the data.
        """
        additive     = np.array(additive)
        data_weights = np.ones(len(additive.T)) if data_weights is None else np.array(data_weights)
        if len(data_weights) != len(additive.T):
            raise ValueError("Dimensions of data and weights is not compatible.")

        binnedData, outRange = self.bin_data(additive, data_weights)

        self.data      += binnedData
        self.out_range += outRange
        self.update_errors()

    def bin_data(self, data, data_weights):
        """Bins the given data and computes the events out of range.

        Parameters
        ----------
        data         : np.array
        Data to be binned.
        data_weights : np.array
        Weights for the data points.
        """
        binned_data, *_ = np.histogramdd(data.T, self.bins,
                                         weights=data_weights)
        out_of_range    = self.count_out_of_range(np.array(data, ndmin=2))

        return binned_data, out_of_range

    def count_out_of_range(self, data):
        """
        Returns an array with the number of events out of the
        Histogram's bin range of the given data.

        Parameters
        ----------
        data : np.array
        """
        out_of_range = []
        for i, bins in enumerate(self.bins):
            lower_limit = bins[0]
            upper_limit = bins[-1]
            out_of_range.append([np.count_nonzero(data[i] < lower_limit),
                                 np.count_nonzero(data[i] > upper_limit)])
        return np.asarray(out_of_range).T

    def update_errors(self, errors=None):
        """
        Updates the errors with the passed list/array.
        If nothing is passed, the square root of the
        counts is computed and assigned as error.

        Parameters
        ----------
        errors : np.array or sequence
        """
        self.errors = np.asarray(errors) if errors is not None else np.sqrt(self.data)

    def _check_valid_binning(self, bins):
        if len(self.bins) != len(bins) or not np.all(a == b for a, b in zip(self.bins, bins)):
            raise ValueError("Histogram binning is not compatible")

    def __radd__(self, other):
        return self + other

    def __add__ (self, other):
        if other is None:
            return self
        self._check_valid_binning(other.bins)
        if self.title  !=  other.title:
            print(f"""Warning: Histogram titles are different.
                      {self.title}, {other.title}""")
        if self.labels != other.labels:
            print(f"""Warning: Histogram titles are different.
                      {self.labels}, {other.labels}""")
        new_histogram           = Histogram(self.title, self.bins,
                                            self.labels, self.scale)
        new_histogram.data      =           self.data        + other.data
        new_histogram.out_range =           self.out_range   + other.out_range
        new_histogram.errors    = np.sqrt  (self.errors ** 2 + other.errors ** 2)
        new_histogram.scale     =           self.scale
        return new_histogram


class HistoManager:
    def __init__(self, histograms=None):
        """
        This class is a parameter holder that contains
        a dictionary of Histogram objects.

        Attributes
        ----------
        histos : dict
        Histogram objects.
        The keys are the Histograms' name.

        Parameters
        ----------
        histograms : sequence
        Initial Histogram objects.
        """
        self.histos = {}

        if histograms is not None:
            values = histograms.values() if isinstance(histograms, dict) else iter(histograms)
            for histogram in values:
                self.new_histogram(histogram)

    def new_histogram(self, histogram):
        """Adds a new Histogram to the HistoManager.

        Parameters
        ----------
        histogram : Histogram object
        """
        self[histogram.title] = histogram

    def fill_histograms(self, additives):
        """Fills several Histograms of the Histomanager.

        Parameters
        ----------
        additives: dict
        The keys are the Histograms names.
        The values are the data to fill the Histogram with.
        """
        for histoname, additive in additives.items():
            if histoname in self.histos:
                self[histoname].fill(np.asarray(additive))
            else:
                print(f"Histogram with name {histoname} does not exist")

    def __getitem__(self, histoname):
        return self.histos[histoname]

    def __setitem__(self, histoname, histogram):
        self.histos[histoname] = histogram
