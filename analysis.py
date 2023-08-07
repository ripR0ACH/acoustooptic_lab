from acoustic_entrainment import mic_response
import matplotlib.pyplot as plt
from time_series import CollectionTDMS as ctdms
import numpy as np
import sys
sys.path.append("../../lhillber/brownian/src")


class System():

    def __init__(self, name="", data_files=[], power=19, SNR_freq_cut=0, phis=[], SNR=[], SNR_resolution=10) -> None:
        self.set_name(name)
        self.set_df(np.array(data_files))
        self.set_data(self.__df)
        self.set_power(power)
        self.set_phis(phis)
        self.set_SNR_freq_cutoffs(SNR_freq_cut)
        self.set_SNR(SNR)
        self.set_SNR_resolution(SNR_resolution)

    #         self.__SNR
    #         self.__system_frequency_cutoff
    #         self.__SNR_cutoff

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
        data = np.empty((len(df), ), dtype=object)
        for d in range(len(df)):
            data[d] = ctdms(df[d])
            if self.get_name()[:3] == "mic":
                data[d].set_collection("Y")
            else:
                data[d].set_collection("X")
        self.__data = data
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
    def set_SNR(self, SNR) -> None:
        """"""
        self.__SNR = SNR
        return None

    def get_SNR(self) -> tuple:
        """"""
        return self.__SNR

    def set_SNR_resolution(self, res) -> None:
        """"""
        self.__SNR_res = res
        return None

    def get_SNR_resolution(self) -> int:
        """"""
        return self.__SNR_res
