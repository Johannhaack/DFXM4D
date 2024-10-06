import glob
import os
from darfix.core.dataset import Dataset
import h5py
from matplotlib import pyplot as plt
from darfix.core.dimension import POSITIONER_METADATA
from darfix.core.dataset import Dataset
from darfix.io.utils import create_nxdata_dict
from silx.io.dictdump import dicttonx
from silx.utils.enum import Enum as _Enum
import concurrent.futures
import shutil



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

def find_directories_with_name(root_dir, search_name):
    matching_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if search_name in dirname:
                matching_dirs.append(os.path.join(dirpath, dirname))
    return matching_dirs

def find_numbered_keys(h5_file):
    numbered_keys = []
    for key in h5_file.keys():
        if key.replace('.', '').isdigit() and key.endswith('.1'):
            numbered_keys.append(key)
    return numbered_keys

def process_directories(directories, raw_string, metadata_string):
    file_metadata_dict = {}
    for directory in directories:
        h5_files = glob.glob(os.path.join(directory, "*.h5"))

        if len(h5_files) == 0:
            print(f"No h5 file found in directory: {directory}")
            continue

        h5_file_path = h5_files[0]  # Assuming one .h5 file per directory
        with h5py.File(h5_file_path, 'r') as h5_file:
            numbered_keys = find_numbered_keys(h5_file)

            if numbered_keys:
                for key in numbered_keys:
                    raw_folder_location = f"{h5_file_path}?/{key}{raw_string}"
                    metadata_location = f"{h5_file_path}?/{key}{metadata_string}"
                    file_metadata_dict[directory+key] = {
                        "raw_folder_location": raw_folder_location,
                        "metadata_location": metadata_location
                    }
            else:
                print(f"No numbered keys found in {h5_file_path}")

    return file_metadata_dict

def process_dataset(key, values, dir_save):
    try:
        raw_location = values['raw_folder_location']
        desired_part = raw_location.split('/')[-4]
        number = key.split('/')[-1]
        file_name = f"{desired_part}_{number}"
        file_name = file_name.replace(".h5?", "")
        file_name = file_name.replace(".", "_")

        dir = f"{dir_save}/{file_name}"
        print(f"Processing key {file_name}")
        # Initialize dataset
        #Its important here to have different dictionary for each dataset, otherwise we overwrite stuff
        darfix_dataset = Dataset(
            _dir=dir,
            in_memory=True,
            first_filename=values["raw_folder_location"],
            metadata_url=values["metadata_location"],
            isH5=True
        )
        dataset = darfix_dataset
        

        # Set dimensions and reshape data
        dataset.find_dimensions(POSITIONER_METADATA)

        # Start with specific values first
        initial_dims_0 = 37
        initial_dims_1 = 26



        # Create a list of tuples for the initial values and the ±2 range
        dim_combinations = [(initial_dims_0, initial_dims_1)] + [
            (dims_0, dims_1)
            for dims_0 in range(initial_dims_0 - 2, initial_dims_0 + 2)
            for dims_1 in range(initial_dims_1 - 2, initial_dims_1 + 2)
        ]

        # Iterate over the list of dimension combinations
        for i, (dims_0, dims_1) in enumerate(dim_combinations):
            dataset.dims.set_size(0, dims_0)
            dataset.dims.set_size(1, dims_1)
            try:
                dataset = dataset.reshape_data()
            except Exception as e:
                if i == len(dim_combinations) - 1: 
                    print(f"Reshape failed")
                    raise  # Reraise the exception to stop the loop
                else:
                    continue  # Continue the loop for non-final iterations
            if dims_0 == initial_dims_0 and dims_1 == initial_dims_1:
                raise Exception("Initial dimensions")
            print(f"Reshape succeeded with dims 0: {dims_0}, dims 1: {dims_1}")
            break  # Exit if successful
        # Noise removal
        print(f"Starting noise removal for key {file_name}")
        dataset = dataset.apply_background_subtraction(method="median")
        dataset = dataset.apply_threshold_removal(bottom=10, top=64000)

        # Apply moments
        print(f"Starting moments for key {file_name}")
        moments = dataset.apply_moments()

        # Export results

        exportMaps(dataset, None, None, moments, file_name)

        if os.path.exists(dir):
            try:
                shutil.rmtree(dir)
                print(f"Directory {dir} has been deleted.")
            except OSError as e:
                print(f"Error: {e.strerror} - {e.filename}")
                return f"Key {file_name} processed successfully"
    except Exception as e:
        print(f"Error processing key {file_name}: {str(e)}")
        return f"Error processing key {file_name}"

if __name__ == "__main__":

    print("Starting script")
    root_directory_data = "/dtu/3d-imaging-center/projects/2022_QIM_PMP/raw_data_extern/2024_06_beamtime/RAW_DATA/111_cells_2/"  # Update with your root directory
    root_directory_save = "/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/Johann_Haack/3D_Volume"  # Update with your root directory
    search_term = "mosalayers_2x"  # Search term
    directories = find_directories_with_name(root_directory_data, search_term)
    del directories[2]  # Removing unwanted directory
    #for directory in directories:
        #print(directory)
    file_metadata_dict = process_directories(directories, '/measurement/pco_ff', '/instrument/positioners')
    #to try with two entries
    #file_metadata_dict = dict(list(file_metadata_dict.items())[:2])

    for key, values in file_metadata_dict.items():
        process_dataset(key, values,root_directory_save)

    
        
