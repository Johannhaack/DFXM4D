import sys
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from skimage.measure import label
from random import randint
from skimage.color import label2rgb
from matplotlib.animation import FuncAnimation, FFMpegWriter
from skimage import morphology

from DFXM.scan_functions import *
from DFXM.image_processor import inv_polefigure_colors 


def create_binary_masks(image_3d, num_bins):
    '''This function takes a 3D image and creates binary masks for each intensity bin'''
    hist_bins = np.linspace(np.nanmin(image_3d), np.nanmax(image_3d), num_bins + 1)
    binary_masks = []
    
    for i in range(num_bins):
        # Create binary mask for each bin
        mask = ((image_3d >= hist_bins[i]) & (image_3d < hist_bins[i + 1])).astype(int)
        binary_masks.append(mask)

    return binary_masks, hist_bins

def connected_components_on_masks(binary_masks, connectivity=3):
    '''This function takes a list of binary masks and performs connected component labeling on each mask'''
    labeled_masks = []
    for mask in binary_masks:
        print(mask.shape)
        #This next line is the actual connected component labeling, where we can also define a structering element and so on 
        labeled_slice, n_labels = label(mask,background=0,return_num=True,connectivity=connectivity)
        labeled_masks.append((labeled_slice, n_labels))
    return labeled_masks

def postprocess_connected_components(labeled_mask, closing_size, num_components):
    '''This function takes a labeled mask and performs a closing operation on the largest components'''
    component_sizes = np.bincount(labeled_mask.ravel())  
    component_sizes[0] = 0  
    largest_components = np.argsort(component_sizes)[-num_components:]  
    biggest_regions_maks = np.isin(labeled_mask, largest_components)

    #closing the mask
    struct_elem = morphology.ball(closing_size)
    closed_volume = morphology.binary_closing(biggest_regions_maks, footprint=struct_elem)
    closed_volume = morphology.binary_closing(closed_volume, footprint=struct_elem)
    #need to return labels not binary mask
    return labeled_mask*closed_volume

def create_random_colors(number_of_colors):
    '''
    Returns a list of number_of_colors random RGB colors
    where each channel has values between 0 and 1,
    e.g. red is (1,0,0) and blue is (0,1,0). 
    '''
    colors = []
    for i in range(number_of_colors):
        col_hex = '#%06X' % randint(0, 0xFFFFFF)
        h = col_hex.lstrip('#')
        col_rgb =  tuple(int(h[i:i+2], 16)/256 for i in (0, 2, 4))
        colors.append(col_rgb)
    return colors

def visualize_3d(labeled_processed_mask, original_volume, bin_number,num_components, save = False):	
    colors = create_random_colors(num_components)
    image_label_overlay = label2rgb(labeled_processed_mask, colors=colors,bg_label=0)

    #Gives the option to save the animation
    if save:
        fig, ax = plt.subplots(figsize=(6, 6))
        
        def update(frame):
            ax.clear()
            ax.imshow(original_volume[frame, :, :], cmap='gray')
            ax.imshow(image_label_overlay[frame, :, :], alpha=0.7)
            ax.set_title(f"Frame {frame + 1}/{original_volume.shape[0]}")
            ax.axis('off')

        # Create the animation
        anim = FuncAnimation(fig, update, frames=np.arange(0, original_volume.shape[0]), interval=200, repeat=True)
        # Specify the writer explicitly
        writer = FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)

        # Save the animation as an MP4 video
        anim.save(f'animation_layers{bin_number}_.mp4', writer=writer)

    interactive_viewer(labeled_processed_mask, original_volume, num_components)

def interactive_viewer(labeled_mask, original_volume, num_components):
    colors = create_random_colors(num_components)
    image_label_overlay = label2rgb(labeled_mask, colors=colors, bg_label=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    frame_idx = [0]  # Use a mutable object to keep track of the frame index

    def update_plot():
        ax.clear()
        ax.imshow(original_volume[frame_idx[0], :, :], cmap='gray')
        ax.imshow(image_label_overlay[frame_idx[0], :, :], alpha=0.7)
        ax.set_title(f"Frame {frame_idx[0] + 1}/{original_volume.shape[0]}")
        ax.axis('off')
        fig.canvas.draw()
    
    def on_key(event):
        if event.key == 'right':
            frame_idx[0] = (frame_idx[0] + 1) % original_volume.shape[0]
        elif event.key == 'left':
            frame_idx[0] = (frame_idx[0] - 1) % original_volume.shape[0]
        update_plot()

    fig.canvas.mpl_connect('key_press_event', on_key)
    
    update_plot()  # Display the initial frame
    plt.show()

def film_2D(frames):

    fig, ax = plt.subplots(figsize=(6, 6))
    frame_idx = [0]  # Use a mutable object to keep track of the frame index

    def update_plot():
        ax.clear()
        ax.imshow(frames[frame_idx[0], :, :], cmap='gray')
        ax.set_title(f"Frame {frame_idx[0] + 1}/{frames.shape[0]}")
        ax.axis('off')
        fig.canvas.draw()
    
    def on_key(event):
        if event.key == 'right':
            frame_idx[0] = (frame_idx[0] + 1) % frames.shape[0]
        elif event.key == 'left':
            frame_idx[0] = (frame_idx[0] - 1) % frames.shape[0]
        update_plot()

    fig.canvas.mpl_connect('key_press_event', on_key)
    
    update_plot()  # Display the initial frame
    plt.show()

def load_2D_time_data(path, img_size, type='FWHM'):   

    #For the new Data we need upper case letters for the old lower case
    com_phi, com_chi = load_data(path, 'COM')
    fwhm_phi, fwhm_chi = load_data(path, 'FWHM')
    order = com_chi


    # Placeholder for storing images
    img_chi_list,cropped_chi_list = [],[]
    img_phi_list,cropped_phi_list = [],[]

    # Define your target size
    target_row_size = img_size[0]  # Example size, set to desired value
    target_col_size = img_size[1]   # Example size, set to desired value

    # Define the cropping bounds (0.49 is the max at 0.5 there is no image left)
    crop_fraction = 0.45
    crop_row_start = int(crop_fraction * target_row_size)
    crop_row_end = target_row_size - crop_row_start
    crop_col_start = int(crop_fraction * target_col_size)
    crop_col_end = target_col_size - crop_col_start

    
    if(len(com_chi) != len(fwhm_chi)):
        ValueError("The number of COM and FWHM images do not match")

    for i in range(len(com_chi)):
        #we need the grain mask to get the FWHM images so always get COM
        Img_chi, maximum_chi, minimum_chi, average_chi, TF_chi, row_size_chi, col_size_chi, _ = process_data(path, com_chi[i], method='COM')
        Img_phi, maximum_phi, minimum_phi, _, _, _, _, _ = process_data(path, com_phi[i], method='COM')
        if type == 'FWHM':
            grain = find_grain(TF_chi)
            _, _, grain_mask = values_histogram(Img_phi, maximum_phi, grain)
            Img_chi, maximum_chi, minimum_chi, average_chi, TF_chi, row_size_chi, col_size_chi, _ = process_data(path, fwhm_chi[i], method='FWHM', grain_mask=grain_mask)
            Img_phi, maximum_phi, minimum_phi, _, _, _, _, _ = process_data(path, fwhm_phi[i], method='FWHM', grain_mask=grain_mask)
        
        #TODO: If they are not the same size we need to do registration, for now we just skip them 
        if Img_chi.shape[0] == target_row_size and Img_chi.shape[1] == target_col_size:
            Img_chi = Img_chi[:target_row_size, :target_col_size]
            img_chi_list.append(Img_chi)
            #Now for the cropped images
            cropped_chi = Img_chi[crop_row_start:crop_row_end, crop_col_start:crop_col_end]
            cropped_chi_list.append(cropped_chi)
        else:
            print(f"Skipping Img_chi at index {i} due to insufficient size: {Img_chi.shape}")
        
        if Img_phi.shape[0] == target_row_size and Img_phi.shape[1] == target_col_size:
            Img_phi = Img_phi[:target_row_size, :target_col_size]
            img_phi_list.append(Img_phi)
            #Now for the cropped images
            cropped_phi = Img_phi[crop_row_start:crop_row_end, crop_col_start:crop_col_end]
            cropped_phi_list.append(cropped_phi)
        else:
            print(f"Skipping Img_phi at index {i} due to insufficient size: {Img_phi.shape}")
    

    # Convert the lists to NumPy arrays for original images
    Img_chi_array = np.stack(img_chi_list) if img_chi_list else np.empty((0, target_row_size, target_col_size))
    Img_phi_array = np.stack(img_phi_list) if img_phi_list else np.empty((0, target_row_size, target_col_size))

    # Convert the lists to NumPy arrays for cropped images
    Img_chi_array_cropped = np.stack(cropped_chi_list) if cropped_chi_list else np.empty((0, crop_row_end - crop_row_start, crop_col_end - crop_col_start))
    Img_phi_array_cropped = np.stack(cropped_phi_list) if cropped_phi_list else np.empty((0, crop_row_end - crop_row_start, crop_col_end - crop_col_start))
    
    return Img_chi_array, Img_phi_array, Img_chi_array_cropped, Img_phi_array_cropped, order
    