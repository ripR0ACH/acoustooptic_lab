import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../../lhillber/brownian/src")
from time_series import CollectionTDMS as ctdms
from acoustic_entrainment import mic_response

class System():

    def __init__(self, name = "", data_files = [], power = 19, SNR_freq_cut = 0, phis = [], SNR_resolution = 10, SNR_freq_range = [10000, 2e6]) -> None:
        self.set_name(name)
        self.set_df(np.array(data_files))
        self.set_data(self.__df)
        self.set_power(power)
        self.set_phis(phis)
        self.set_SNR_resolution(SNR_resolution)
        self.set_SNR_freq_cutoff(SNR_freq_cut)
        self.set_SNR_freq_range(SNR_freq_range)
        self.set_SNR_at_cutoff(self.calc_SNR_at_cutoff())
        self.reset_SNR_vs_freq()

    def get_name(self) -> str:
        """
        
        get_name gets the name of the system.
        :return: the name of the system.
        
        """
        return self.__name

    def set_name(self, n) -> None:
        """
        
        set_name sets the name of the system
        :param n: name of the system.
        :return: None.
        
        """
        self.__name = n
        return None

    def get_df(self) -> np.array([]):
        """
        
        get_df gets the list of data files that the system has.
        :return: the list of data files in the system
        
        """
        return self.__df

    def set_df(self, df) -> None:
        """
        
        set_df sets the list of data files.
        :param df: list of data files.
        :return: None.
        
        """
        self.__df = df
        return None

    def get_data(self) -> np.array([]):
        """
        
        get_data gets the data collections for the system.
        :return: list of data collections.
        
        """
        return self.__data

    def set_data(self, df) -> None:
        """
        
        set_data sets the data collections for every data file provided for the system.
        :param df: list of data files provided in the system object.
        :return: None.
        
        """
        self.__data = np.empty((len(df), ), dtype=object)
        for d in range(len(df)):
            self.__data[d] = ctdms(df[d])
            if self.get_name()[:3] == "mic":
                self.__data[d].set_collection("Y")
                self.__data[d].apply("correct", response = mic_response, recollect = True)
            else:
                self.__data[d].set_collection("X")
                self.__data[d].apply("calibrate", cal = -1, inplace = True)
        return None

    def get_power(self) -> int:
        """
        
        get_power gets the power of the pulsed laser.
        :return: integer value of the laser power.
        
        """
        return self.__power

    def set_power(self, p) -> None:
        """
        
        set_power sets the power of the pulsed laser used in the system.
        :param p: power of the pulsed laser.
        :return: None.
        
        """
        self.__power = p
        return None

    def get_SNR_freq_cutoff(self) -> tuple:
        """
        
        get_SNR_freq_cutoffs gets the frequency cutoffs range.
        :return: tuple containing the starting and ending frequency of the cutoffs.
        
        """
        return self.__SNR_freq_cutoff

    def set_SNR_freq_cutoff(self, freq) -> None:
        """
        
        set_SNR_freq_cutoffs sets the frequency cutoffs range.
        :freqs: frequency cutoff that the SNR will be run at.
        :return: None.
        
        """
        self.__SNR_freq_cutoff = freq
        return None

    def get_phis(self) -> list:
        """
        
        get_phis gets the phis that the system scans over in the minimum detectable setup.
        :return: None.
        
        """
        return self.__phis

    def set_phis(self, p) -> None:
        """
        
        set_phis sets the phis that the system scans over to an array.
        :param p: list of phis.
        :return: None.
        
        """
        self.__phis = p
        return None

    # add comments for these functions down!
    def set_SNR_resolution(self, res) -> None:
        """"""
        self.__SNR_res = res
        return None

    def get_SNR_resolution(self) -> int:
        """"""
        return self.__SNR_res
        
    def set_SNR_freq_range(self, ran) -> None:
        """"""
        self.__SNR_freq_range = np.linspace(ran[0], ran[1], self.get_SNR_resolution())
        return None
    def get_SNR_freq_range(self) -> list:
        """"""
        return self.__SNR_freq_range

    def local_detrend(self, col, tmin = None, tmax = None, inplace = False) -> None:
        """"""
        for c in col.collection:
            t, x = c.time_gate(tmin = tmin, tmax = tmax)
            x_bar = np.mean(x)
            m, b = np.polyfit(t, x, 1)
            if inplace:
                c.x = c.x - (m * c.t) - b
        return None

    def reset_SNR_at_cutoff(self) -> None:
        """"""
        self.__SNR_at_cutoff = np.array([])
        return None
    def calc_SNR_at_cutoff(self, f = 0, bins = False) -> np.array([]):
        """"""
        if f == 0:
            f = self.get_SNR_freq_cutoff()
        snr = []
        for dat in self.get_data()[:]:
            peaks = np.array([])
            rms = np.array([])
            if bins:
                N = int(dat.r / (2 * f))
                dat.apply("bin_average", Npts = N, inplace = True)
            else:
                dat.apply("lowpass", cutoff = f, inplace = True)
            self.local_detrend(dat, 1.7e-4, 3.5e-4, True)
            for s in dat.collection[1:]:
                peaks = np.append(peaks, max(s.time_gate(tmin = 3.5e-4, tmax = 5e-4)[1]))
                rms = np.append(rms, np.std(s.time_gate(tmin = 1.7e-4, tmax = 3.5e-4)[1]))
            snr.append(np.mean(peaks / rms))
            if self.get_name()[:3] == "mic":
                dat.set_collection("Y")
                dat.apply("correct", response = mic_response, recollect = True)
            else:
                dat.set_collection("X")
                dat.apply("calibrate", cal = -1, inplace = True)
        return snr
    def set_SNR_at_cutoff(self, snr) -> None:
        """"""
        self.__SNR_at_cutoff = snr
        return None
    def get_SNR_at_cutoff(self) -> list:
        """"""
        return self.__SNR_at_cutoff

    def reset_SNR_vs_freq(self) -> None:
        """"""
        self.__SNR_vs_freq = []
        return None
    def calc_SNR_vs_freq(self, freq = [0, 0], bins = False) -> None:
        """"""
        if freq[0] == 0 and freq[1] == 0:
            freq = self.get_SNR_freq_range()
        else:
            freq = np.linspace(freq[0], freq[1], self.get_SNR_resolution())
        for i in range(len(freq)):
            self.__SNR_vs_freq.append(self.calc_SNR_at_cutoff(freq[i], bins))
#         if self.get_name()[:3] != "mic":
#             self.__SNR_vs_freq.reverse()
        self.__SNR_vs_freq = np.array(self.__SNR_vs_freq)
        return None
    def get_SNR_vs_freq(self) -> np.array([]):
        """"""
        return self.__SNR_vs_freq