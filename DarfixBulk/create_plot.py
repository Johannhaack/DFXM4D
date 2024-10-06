import glob
import os
import numpy
from darfix.core.dataset import Dataset
import h5py
from matplotlib import pyplot as plt
from darfix.core.dimension import POSITIONER_METADATA
from darfix.core.dataset import Dataset
from darfix.io.utils import create_nxdata_dict
from matplotlib.colors import hsv_to_rgb
from silx.io.dictdump import dicttonx
from silx.utils.enum import Enum as _Enum
import joblib
import psutil
import time
import threading

def monitor_memory():
    # Get the current process
    process = psutil.Process()
    
    # Initialize max memory usage
    max_memory_in_gb = 0
    
    # Continuously monitor memory usage as long as the process is running
    while process.is_running():
        # Get the current memory usage in GB
        current_memory_in_gb = process.memory_info().rss / (1024 ** 3)
        
        # If current memory usage is greater than the max recorded, update and print
        if current_memory_in_gb > max_memory_in_gb:
            max_memory_in_gb = current_memory_in_gb
            print(f"New max memory usage: {max_memory_in_gb:.2f} GB")
        
        # Wait for 10 seconds before checking again
        time.sleep(3)

def find_directories_with_name(root_dir, search_name):
    matching_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if search_name in dirname:
                matching_dirs.append(os.path.join(dirpath, dirname))
    return matching_dirs

# Function to find numbered keys in an h5 file
def find_numbered_keys(h5_file):
    numbered_keys = []
    for key in h5_file.keys():
        if key.replace('.', '').isdigit() and key.endswith('.1') :  # Check if the key is numeric
            numbered_keys.append(key)
    return numbered_keys

# Function to iterate over directories and process each .h5 file
def process_directories(directories, raw_string,metadata_string):
    file_metadata_dict = {}
    for directory in directories:
        # Find the h5 file in the current directory
        h5_files = glob.glob(os.path.join(directory, "*.h5"))
        
        if len(h5_files) == 0:
            print(f"No h5 file found in directory: {directory}")
            continue
        
        h5_file_path = h5_files[0]  # Assuming there is one .h5 file per directory
        
        # Open the h5 file and look for numbered keys
        with h5py.File(h5_file_path, 'r') as h5_file:
            numbered_keys = find_numbered_keys(h5_file)
            
            if numbered_keys:
                for key in numbered_keys:
                    raw_folder_location = f"{h5_file_path}?/{key}{raw_string}"
                    metadata_location =f"{h5_file_path}?/{key}{metadata_string}"
                    
                    # Add the file and metadata locations to the dictionary using the key as the dictionary key
                    file_metadata_dict[key] = {"raw_folder_location": raw_folder_location, "metadata_location": metadata_location}
            else:
                print(f"No numbered keys found in {h5_file_path}")

    return file_metadata_dict

class Method(_Enum):
    """
    Different maps to show
    """

    COM = "Center of mass"
    FWHM = "FWHM"
    SKEWNESS = "Skewness"
    KURTOSIS = "Kurtosis"
    ORI_DIST = "Orientation distribution"
    MOSAICITY = "Mosaicity"

def exportMaps(dataset, mosaicity, hsv_key, moments, filename):
        """
        Creates dictionay with maps information and exports it to a nexus file
        """
        if dataset.transformation:
            print("Step 2")
            axes = [
                dataset.transformation.yregular,
                dataset.transformation.xregular,
            ]
            axes_names = ["y", "x"]
            axes_long_names = [
                dataset.transformation.label,
                dataset.transformation.label,
            ]
        else:
            print("Step 3")
            axes = None
            axes_names = None
            axes_long_names = None

        if dataset and dataset.dims.ndim > 1:

            print("Step 4")
            nx = {
                "entry": {"@NX_class": "NXentry"},
                "@NX_class": "NXroot",
                "@default": "entry",
            }

            for axis, dim in dataset.dims:
                nx["entry"][dim.name] = {"@NX_class": "NXcollection"}
                for i in range(4):
                    nx["entry"][dim.name][Method.values()[i]] = create_nxdata_dict(
                        moments[axis][i],
                        Method.values()[i],
                        axes,
                        axes_names,
                        axes_long_names,
                    )
        else:

            print("Step 5")
            nx = {
                "entry": {"@NX_class": "NXentry"},
                "@NX_class": "NXroot",
                "@default": "entry",
            }

            for i in range(4):
                nx["entry"][Method.values()[i]] = create_nxdata_dict(
                    moments[0][i],
                    Method.values()[i],
                    axes,
                    axes_names,
                    axes_long_names,
                )
            nx["entry"]["@default"] = Method.COM.value

        dicttonx(nx, f'/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/Johann_Haack/{filename}.h5')

# Create a separate thread to run the monitor_memory function
memory_thread = threading.Thread(target=monitor_memory)

# Start the memory monitoring thread
memory_thread.start()

load_data = True

if load_data:

    # Load Dataset object
    dataset = joblib.load('/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/Johann_Haack/saved_dataset.joblib')
    moments = dataset.apply_moments()
    print("Moments calculated")

    # Load moments object
    #moments = joblib.load('/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/Johann_Haack/saved_moments.joblib') 

    
else:
    root_directory = "/dtu/3d-imaging-center/projects/2022_QIM_PMP/raw_data_extern/2024_06_beamtime/RAW_DATA/111_cells_2/"  # Replace with your root directory
    search_term = "mosalayers"  # Replace with the name you're looking for
    directories = find_directories_with_name(root_directory, search_term)
    for directory in directories:
        print(directory)


    file_metadata_dict = process_directories(directories, '/measurement/pco_ff', '/instrument/positioners')

    darfix_dataset_lst = []

    for key, values in file_metadata_dict.items():
        darfix_dataset = Dataset(
            _dir = "/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/Johann_Haack",
            in_memory=True,
            first_filename=values["raw_folder_location"],
            metadata_url=values["metadata_location"],
            isH5=True
        )
        darfix_dataset_lst.append(darfix_dataset)
        #Take out when it ran for one file
        break

    dataset = darfix_dataset_lst[0]
    dataset.find_dimensions(POSITIONER_METADATA)

    #This sets the dimesions of the angles
    dataset.dims.set_size(0,37)
    dataset.dims.set_size(1,26)
    dataset = dataset.reshape_data()

    #Noise removal
    print("Starting noise removal")
    dataset = dataset.apply_background_subtraction(method="median")
    print("Starting threshold removal") 
    dataset = dataset.apply_threshold_removal(bottom=10,top=64000)
    
    print("Starting moments")
    moments = dataset.apply_moments()

exportMaps(dataset, None, None, moments, "test1")

memory_thread.join()
