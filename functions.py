import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import Tensor
import os

def open_print_image(path):
    """Input a path of the image to be opened and the name of image(string), 
    It opens, normalizes, prints and ouputs the image"""
    #open the image
    img=Image.open(path)
    #Normalize the image
    img= np.array(img, dtype='float')/255.0
    img = img[:,:,0:3]
    print('Input image shape', img.shape)
    return img

def extract_mask(image_scribble,avg_colour = 1):
    """Input a scribbled image and name for the mask, avg colour is colour used for scribble,
    In general white is being used so, given default avg_colour to 1.
    It generates a mask of the scribble and prints the mask"""
    #eps is to accomodate error in the white colour of scribble
    eps = 0.01
    mask = ((image_scribble.mean(axis=2) >= avg_colour-eps) & (image_scribble.mean(axis=2) <= avg_colour+eps))   
    return mask

def extract_data_from_image(original_image, mask):
    """Input the original image and the generated fore ground/ back ground mask,
    it generates the data (x,y,r,g,b) of the all pixels in the mask and no of pixels """
    indices = np.nonzero(mask)
    No_of_pixels = len(indices[1])
    pixel_info = np.zeros((5,No_of_pixels)) 
    # store x,y,R,G,B values of all pixels the user marked as foreground
    nx,ny,nc = original_image.shape
    pixel_info[0,:] = indices[0] / nx
    pixel_info[1,:] = indices[1] / ny
    pixel_info[2:5,:] = original_image[indices[0], indices[1],:].T
    return pixel_info,No_of_pixels

def neuralnet_eval(Trained_network,img_org,threshold = 0.5):
    """Give a trained network and input image,
    It generates the mask"""

    full_mask = np.ones((img_org.shape[0],img_org.shape[1]), dtype=bool)
    #extract data from the image using the function "extract_data_from_image"
    allPixels_data,allPixels_data_len = extract_data_from_image(img_org,full_mask)
    #testing with network
    with torch.no_grad():
        inferenceResult = Trained_network(Tensor(allPixels_data.T))
    #converting the tensor ouput to numpy, reshaping to size of image
    inferenceResult = inferenceResult.numpy()
    inferenceResult = inferenceResult.reshape((img_org.shape[0],img_org.shape[1]))
    inferenceResult_mask = np.where(inferenceResult < threshold,0,1)
    return inferenceResult_mask

#functons useful for segmentation net
def add_padding_to_divisible_by(image, divisor):
    # Get the original width and height of the image
    original_height, original_width, channels = image.shape
    # Calculate the amount of padding needed for width and height
    padding_width = (divisor - (original_width % divisor)) % divisor
    padding_height = (divisor - (original_height % divisor)) % divisor
    # Calculate the new width and height after adding padding
    new_width = original_width + padding_width
    new_height = original_height + padding_height
    # Create a new blank image with the desired width, height, and padding color
    padded_image = np.zeros((new_height, new_width, channels))
    # Paste the original image onto the new blank image at the top-left corner
    padded_image[:original_height,:original_width,:] = image
    return padded_image

def segmentationet_eval(model, inp, threshold = 0.5):
    model.eval()
    seg_out = model(inp)
    seg_out_squeezed = seg_out.squeeze().detach().cpu().numpy()
    # Calculate the minimum and maximum values in the array
    min_value = np.min(seg_out_squeezed)
    max_value = np.max(seg_out_squeezed)
    # Apply min-max normalization
    normalized_seg_out_squeezed = (seg_out_squeezed - min_value) / (max_value - min_value)
    # plt.figure()
    # plt.title("normalized_seg_out_squeezed",fontsize = 20)
    # plt.imshow(normalized_seg_out_squeezed,cmap = 'gray')
    threshold_array = np.where(normalized_seg_out_squeezed < threshold,0,1)
    return threshold_array

#convex net functions
def convert_segmented_inference_to_covexnet_dataset(primary_segmented_inference):
    """Pass a primary_segmented_inference and it generates a dataset for the convex network"""
    nx,ny = primary_segmented_inference.shape
    indices = np.nonzero(np.ones((nx,ny)))
    n = len(indices[1])
    print('size of data: ',n)
    data = np.zeros((n,2)) # store x,y
    data[:,0] = indices[0] / nx                  
    data[:,1] = indices[1] / ny
    labels = np.zeros((n,1))
    labels[:,0] = primary_segmented_inference[indices[0], indices[1]]
    return data,labels

def extract_segemented_images_from_inference(img_org,inference):
    """Give image and inference from trained networks,
    It generates the foreground and background of the image"""
    inferenceResult_mask_fg = inference <= 0
    #fore ground image segmentation
    segmented_fg_image = img_org.copy()
    #assigning all background pixels to white
    segmented_fg_image[~inferenceResult_mask_fg] = [1, 1, 1]

    #back ground image segmentation
    segmented_bg_image = img_org.copy()
    #assigning all foreground pixels to white
    segmented_bg_image[inferenceResult_mask_fg] = [1, 1, 1]
    return segmented_fg_image,segmented_bg_image

def get_output_folder(base_path):
    # Look for existing output folders and find the highest numbered one
    existing_folders = [folder for folder in os.listdir(base_path) if folder.startswith("output")]
    max_num = 0
    for folder in existing_folders:
        try:
            num = int(folder.split("_")[-1])
            if num > max_num:
                max_num = num
        except ValueError:
            pass

    # Increment the number and create the new output folder
    new_folder_name = os.path.join(base_path, f"output_{max_num + 1}")
    os.makedirs(new_folder_name, exist_ok=True)
    return new_folder_name

