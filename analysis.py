import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfinv
import sys
sys.path.append("../../lhillber/brownian/src")
from time_series import CollectionTDMS as ctdms
from acoustic_entrainment import mic_response
import time
import threading

def calc_cal_factor(l_col, m_col, deviation):
    """
    
    calc_cal_factor calculates the calibration factor for an individual
    shot in a collection array. Calibration is to the microphone data
    which we have a pretty accurate mV/Pa conversion.
    :param l_col: laser collection object for a single shot.
    :param m_col: microphone collection object for a single shot.
    :param deviation: the number of points to each side of the trough 
                      that the calibration is to include in its calculation.
    :return: calibration factor for a shot.
    
    """
    def find_nearest(array, value):
        return (np.abs(np.asarray(array) - value)).argmin()
    
    m_trough = np.where(m_col.x == min(m_col.time_gate(tmin = 3.5e-4, tmax = 5e-4)[1]))[0][0]
    l_trough = find_nearest(l_col.t, m_col.t[m_trough])
    if deviation != 0:
        return np.mean(m_col.x[m_trough - deviation : m_trough + deviation] / l_col.x[l_trough - deviation : l_trough + deviation])
    return m_col.x[m_trough] / l_col.x[l_trough]

def mic_tau_shift(s1, s2, dat_i, col_i, filter = False) -> float: 
    # assuming s1 will always be laser and s2 will always be microphone
    # also assume that the data will be preprocessed at this point:
    # - the data must already be detrended
    # - the data doesn't have to be lowpassed or bin_averaged, but it can be beforehand
    x0 = []
    for s in [s1, s2]:
        c = s.get_data()[dat_i].collection[col_i]
        mx = c.t[np.where(c.x == max(c.time_gate(tmin = 3.5e-4, tmax = 5e-4)[1]))[0][0]]
        mn = c.t[np.where(c.x == min(c.time_gate(tmin = 3.5e-4, tmax = 5e-4)[1]))[0][0]]
        x = [c.time_gate(tmin = mx, tmax = mn)[0][np.where(np.diff(np.sign(c.time_gate(tmin = mx, tmax = mn)[1])))[0][0]], c.time_gate(tmin = mx, tmax = mn)[0][np.where(np.diff(np.sign(c.time_gate(tmin = mx, tmax = mn)[1])))[0][0] + 1]]
        y = [c.x[np.where(c.t == x[0])[0][0]], c.x[np.where(c.t == x[1])[0][0]]]
        x0.append(x[0] - y[0] * ((x[1] - x[0]) / (y[1] - y[0])))
    return x0[1] - x0[0]

def graph_systems(systems = [], title = "", save = False) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (54, 15)
    fig, ax = plt.subplots(2, 9)
    for s in systems:
        ax = ax.flatten()
        fig.suptitle("setups vs frequency (strong shot and phi 82-160)")
        for i in range(len(s.get_SNR_vs_freq()[0])):
            if i == 0:
                ax[i].set_title("Strong shot")
                ax[i].plot(s.get_SNR_freq_range(), s.get_SNR_vs_freq()[:, i], label = s.get_name())
                ax[i].set_ylabel("SNR")
                ax[i].set_xlabel("frequency")
            else:
                ax[i].set_title("Phi " + str(s.get_phis()[i - 1]))
                ax[i].plot(s.get_SNR_freq_range(), s.get_SNR_vs_freq()[:, i], label = s.get_name())
                ax[i].set_ylabel("SNR")
                ax[i].set_xlabel("frequency")
    fig.tight_layout(pad = 1.5)
    ax[0].legend()
    if save:
        fig.savefig(title)
    plt.show()
    return None

def phi(p):
    return np.sqrt(2) * erfinv(2 * p - 1)

def expected_max(N):
    mun = phi(1 - 1 / N)
    sigman = phi(1 - 1 / (N * np.e)) - mun
    return mun + sigman * 0.577
    

def std_max(N, mu):
    mun = phi(1 - 1 / N) + mu
    sigman = phi(1 - 1 / (N * np.e)) - mun
    return sigman * np.pi * np.sqrt(1/6)
    

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
        self.set_SNR_at_cutoff(self.calc_SNR_at_cutoff(bins = True, lowpass = True))
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

    def set_data(self, df = [], ind = 0, mic_correct = True) -> None:
        """
        
        set_data sets the data collections for every data file provided for the system.
        :param df: list of data files provided in the system object.
        :return: None.
        
        """
        if len(df) != 0:
            self.__data = np.empty((len(df), ), dtype=object)
            for d in range(len(df)):
                self.__data[d] = ctdms(df[d])
                if self.get_name()[:3] == "mic":
                    self.__data[d].set_collection("Y")
                    if mic_correct:
                        self.__data[d].apply("correct", response = mic_response, recollect = True)
                else:
                    self.__data[d].set_collection("X")
                    if self.get_name()[:] == "sagnac":
                        self.__data[d].apply("calibrate", cal = -1, inplace = True)
        else:
            self.__data[ind] = ctdms(self.get_df()[ind])
            if self.get_name()[:3] == "mic":
                self.__data[ind].set_collection("Y")
                if mic_correct:
                    self.__data[ind].apply("correct", response = mic_response, recollect = True)
            else:
                self.__data[ind].set_collection("X")
                if self.get_name()[:] == "sagnac":
                    self.__data[ind].apply("calibrate", cal = -1, inplace = True)
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

    def get_SNR_freq_cutoff(self) -> float:
        """
        
        get_SNR_freq_cutoffs gets the frequency cutoff.
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

    def local_detrend(self, col = [], index = 0, tmin = None, tmax = None, inplace = False) -> None:
        """"""
        if col == []:
            d = self.get_data()[index]
            for c in d.collection:
                t, x = c.time_gate(tmin = tmin, tmax = tmax)
                x_bar = np.mean(x)
                m, b = np.polyfit(t, x, 1)
                if inplace:
                    c.x = c.x - (m * c.t) - b
        else:
            for c in col.collection:
                t, x = c.time_gate(tmin = tmin, tmax = tmax)
                x_bar = np.mean(x)
                m, b = np.polyfit(t, x, 1)
                if inplace:
                    c.x = c.x - (m * c.t) - b
        return None

    def reset_SNR_at_cutoff(self) -> None:
        """"""
        self.__SNR_at_cutoff = []
        return None
    def calc_SNR_at_cutoff(self, f = 0, bins = False, lowpass = False) -> np.array([]):
        """"""
        if f == 0:
            f = self.get_SNR_freq_cutoff()
        snr = []
        for i in range(len(self.__data)):
            if lowpass and bins:
                self.__data[i].apply("lowpass", cutoff = f, inplace = True)
                self.__data[i].apply("bin_average", Npts = int(self.__data[i].r / (2 * f)), inplace = True)
            elif bins:
                self.__data[i].apply("bin_average", Npts = int(self.__data[i].r / (2 * f)), inplace = True)
            else:
                self.__data[i].apply("lowpass", cutoff = f, inplace = True)
            self.local_detrend(col = self.__data[i], tmin = 0, tmax = 3.5e-4, inplace = True)
            peaks = np.array([])
            rms = np.array([])
            for s in self.__data[i].collection[1:]:
                peaks = np.append(peaks, np.abs(max(s.time_gate(tmin = 3.5e-4, tmax = 5e-4)[1])))
                rms = np.append(rms, np.std(s.time_gate(tmin = 2e-4, tmax = 3.5e-4)[1]))
            snr.append(np.mean(peaks / (rms * expected_max(len(s.time_gate(tmin = 2e-4, tmax = 3.5e-4)[1])))))
            self.set_data(ind = i)
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
    def calc_SNR_vs_freq(self, freq = [0, 0], bins = False, lowpass = False) -> None:
        """"""
        if freq[0] == 0 and freq[1] == 0:
            freq = self.get_SNR_freq_range()
        else:
            freq = np.linspace(freq[0], freq[1], self.get_SNR_resolution())
        for i in range(len(freq)):
            self.__SNR_vs_freq.append(self.calc_SNR_at_cutoff(freq[i], bins, lowpass))
        self.__SNR_vs_freq = np.array(self.__SNR_vs_freq)
        return None
    def get_SNR_vs_freq(self) -> np.array([]):
        """"""
        return self.__SNR_vs_freq