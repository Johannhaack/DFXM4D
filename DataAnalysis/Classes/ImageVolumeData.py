from ImageSliceData import ImageSliceData
import numpy as np
from typing import Optional

class ImageVolumeData:

    __slots__ = ['volume', 'chi_com_array', 'chi_fwhm_array', 'chi_kurtois_array', 'chi_skewness_array', 'phi_com_array', 'phi_fwhm_array', 'phi_kurtois_array', 'phi_skewness_array', 'chi_feature_array', 'phi_feature_array', 'feature_array']

    def __init__(self, volume: list[ImageSliceData]) -> None:
        self.volume = volume
        #This is a bit heavy on memmory haha
        self.set_volume()

    def set_volume(self, volume: Optional[ImageSliceData] = None) -> None:
        if volume is not None:
            self.volume = volume
        volume = self.volume
        self.chi_com_array = np.array([slice_data.numpy_feature_array[0] for slice_data in volume])
        self.chi_fwhm_array = np.array([slice_data.numpy_feature_array[2] for slice_data in volume])
        self.chi_kurtois_array = np.array([slice_data.numpy_feature_array[4] for slice_data in volume])
        self.chi_skewness_array = np.array([slice_data.numpy_feature_array[6] for slice_data in volume])

        self.phi_com_array = np.array([slice_data.numpy_feature_array[1] for slice_data in volume])
        self.phi_fwhm_array = np.array([slice_data.numpy_feature_array[3] for slice_data in volume])
        self.phi_kurtois_array = np.array([slice_data.numpy_feature_array[5] for slice_data in volume])
        self.phi_skewness_array = np.array([slice_data.numpy_feature_array[7] for slice_data in volume])

        self.chi_feature_array = np.array([slice_data.numpy_feature_array[0:4] for slice_data in volume])
        self.phi_feature_array = np.array([slice_data.numpy_feature_array[4:] for slice_data in volume])

        self.feature_array = np.array([slice_data.numpy_feature_array for slice_data in volume]) 
