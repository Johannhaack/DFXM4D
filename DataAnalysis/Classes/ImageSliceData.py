import h5py
import numpy as np

class ImageSliceData:
    """
    ImageSlice is a class that represents a slice of a 3D image, the Image is produced by DFXM
    slots defnies the variables that are allowed
    numpy feature array is [8,row_size,col_size] array that contains the 8 features of the image
    """
    __slots__ = ['path', 'h5_file', 'com_moment', 'fwhm_moment', 'kurtois_moment', 'skewness_moment','numpy_feature_array']

    def __init__(self):
        pass
    
    def extract_data(self, varient_angle_name = False):
        try:
            dset_com_chi = self.h5_file['entry/chi/Center of mass/Center of mass']
            dset_com_phi = self.h5_file['entry/diffry/Center of mass/Center of mass']
            dset_fwhm_chi = self.h5_file['entry/chi/FWHM/FWHM']
            dset_fwhm_phi = self.h5_file['entry/diffry/FWHM/FWHM']
            dset_kurtois_chi = self.h5_file['entry/chi/Kurtosis/Kurtosis']
            dset_kurtois_phi = self.h5_file['entry/diffry/Kurtosis/Kurtosis']
            dset_skewness_chi = self.h5_file['entry/chi/Skewness/Skewness']
            dset_skewness_phi = self.h5_file['entry/diffry/Skewness/Skewness']
        except Exception as e:
            raise RuntimeError(f"An error occurred while extracting data, most likely keys of h5 are wrong: {e}")
        
        try:
            dset_com_chi,dset_com_phi = dset_com_chi[:],dset_com_phi[:]
            dset_fwhm_chi,dset_fwhm_phi = dset_fwhm_chi[:],dset_fwhm_phi[:]
            dset_kurtois_chi,dset_kurtois_phi = dset_kurtois_chi[:],dset_kurtois_phi[:]
            dset_skewness_chi,dset_skewness_phi = dset_skewness_chi[:],dset_skewness_phi[:]
            
        except Exception as e:
            raise RuntimeError(f"Error when converting to numpy: {e}")
        
        self.com_moment = MomentWrapper(dset_com_chi, dset_com_phi)
        self.fwhm_moment = MomentWrapper(dset_fwhm_chi, dset_fwhm_phi)
        self.kurtois_moment = MomentWrapper(dset_kurtois_chi, dset_kurtois_phi)
        self.skewness_moment = MomentWrapper(dset_skewness_chi, dset_skewness_phi)
        self.set_numpy_feature_array()

    def set_h5_file(self, h5_file_path):
        self.h5_file = h5py.File(h5_file_path)
        self.path = h5_file_path

    def set_numpy_feature_array(self):
        self.numpy_feature_array = np.zeros((8,self.com_moment.motors_dict['Chi'].col_size,self.com_moment.motors_dict['Chi'].row_size))
        self.numpy_feature_array[0] = self.com_moment.motors_dict['Chi'].Img
        self.numpy_feature_array[4] = self.com_moment.motors_dict['Phi'].Img
        self.numpy_feature_array[1] = self.fwhm_moment.motors_dict['Chi'].Img
        self.numpy_feature_array[5] = self.fwhm_moment.motors_dict['Phi'].Img
        self.numpy_feature_array[2] = self.kurtois_moment.motors_dict['Chi'].Img
        self.numpy_feature_array[6] = self.kurtois_moment.motors_dict['Phi'].Img
        self.numpy_feature_array[3] = self.skewness_moment.motors_dict['Chi'].Img
        self.numpy_feature_array[7] = self.skewness_moment.motors_dict['Phi'].Img

class MotorWrapper:
    def __init__(self, motor):
        self.set_single_motor_data(motor)

    def set_single_motor_data(self, moment_motor)-> None:
        self.Img = moment_motor
        self.max = np.nanmax(moment_motor)
        self.min = np.nanmin(moment_motor)
        self.average = np.nanmean(moment_motor)
        self.row_size, self.col_size = moment_motor.shape
        B1 = self.Img.T
        B = np.flipud(B1)
        self.TF = np.isnan(B)

class MomentWrapper:
    def __init__(self ,moment_motor_1_numpy, moment_motor_2_numpy):
        self.motors_dict = {'Chi': MotorWrapper(moment_motor_1_numpy), 'Phi': MotorWrapper(moment_motor_2_numpy)}