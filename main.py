# Load necessary Pytorch packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import argparse

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()

# inputs
parser.add_argument('-ip',
                    "--inputpath",
                    type = str,
                    default = 'data/single_apple/',
                    help="path of Image to segment",
                    )

#model
parser.add_argument("-pm",
                    "--primarymodel", 
                    type = str,
                    default= 'neuralnet',
                    choices = ['neuralnet','segmentationnet'],
                    help="choice of primary model",
                    )
#For Neural net
parser.add_argument("-nnbs",
                    "--nnbatchsize", 
                    type=int,
                    default= 64,
                    help="batch size for dataloader for neural net data",
                    )
parser.add_argument("-nne",
                    "--nnepochs", 
                    type=int,
                    default= 50,
                    help="no of epochs for neural net",
                    )
parser.add_argument("-nnlr",
                    "--nnlearningrate", 
                    type=float,
                    default= 0.00001,
                    help="learning rate for neural net",
                    )
parser.add_argument("-nnhu",
                    "--nnhiddenunits", 
                    type=int,
                    default= 256,
                    help="no of hidden neurons in neural net",
                    )

#for segementation network
parser.add_argument("-sm",
                    "--segmentationmodel", 
                    type = str,
                    default= 'PSPNet',
                    choices = ['PSPNet','Unetplusplus','MAnet','Linknet','FPN','DeepLabV3Plus'],
                    help="choice of segmentationmodel",
                    )
parser.add_argument("-sne",
                    "--snepochs", 
                    type=int,
                    default= 50,
                    help="no of epochs for segmentation net",
                    )
parser.add_argument("-snlr",
                    "--snlearningrate", 
                    type=float,
                    default= 0.0001,
                    help="learning rate for segmentation net",
                    )

#for convexnet
parser.add_argument("-cnbs",
                    "--cnbatchsize", 
                    type=int,
                    default= 64,
                    help="batch size for dataloader for convex net data",
                    )
parser.add_argument("-cne",
                    "--cnepochs", 
                    type=int,
                    default= 50,
                    help="no of epochs for convex net",
                    )
parser.add_argument("-cnlr",
                    "--cnlearningrate", 
                    type=float,
                    default= 0.00001,
                    help="learning rate for convex net",
                    )
parser.add_argument("-cnhu",
                    "--cnhiddenunits", 
                    type=int,
                    default= 128,
                    help="no of hidden neurons in convex net",
                    )

args = parser.parse_args()

from functions import open_print_image, extract_mask, extract_data_from_image, neuralnet_eval
from functions import add_padding_to_divisible_by, segmentationet_eval, convert_segmented_inference_to_covexnet_dataset
from functions import extract_segemented_images_from_inference, get_output_folder
from models.neuralnet import NeuralNet
from models.convexnet import ConvexNet
from nn_train_val import nn_train, nn_validate
from cn_train_val import cn_train, cn_validate

output_folder = get_output_folder(args.inputpath)

writer = SummaryWriter(output_folder+'/runs/')

config_file_path = os.path.join(output_folder, "configuration.txt")
    
with open(config_file_path, "w") as file:
    file.write("Input Path: {}\n".format(args.inputpath))
    file.write("Primary Model: {}\n".format(args.primarymodel))
    
    if args.primarymodel == 'neuralnet':
        file.write("Neural Net Batch Size: {}\n".format(args.nnbatchsize))
        file.write("Neural Net Epochs: {}\n".format(args.nnepochs))
        file.write("Neural Net Learning Rate: {}\n".format(args.nnlearningrate))
        file.write("Neural Net Hidden Units: {}\n".format(args.nnhiddenunits))
        file.write("Convex Net Batch Size: {}\n".format(args.cnbatchsize))
        file.write("Convex Net Epochs: {}\n".format(args.cnepochs))
        file.write("Convex Net Learning Rate: {}\n".format(args.cnlearningrate))
        file.write("Convex Net Hidden Units: {}\n".format(args.cnhiddenunits))
    elif args.primarymodel == 'segmentationnet':
        file.write("Segmentation Model: {}\n".format(args.segmentationmodel))
        file.write("Segmentation Net Epochs: {}\n".format(args.snepochs))
        file.write("Segmentation Net Learning Rate: {}\n".format(args.snlearningrate))
        file.write("Convex Net Batch Size: {}\n".format(args.cnbatchsize))
        file.write("Convex Net Epochs: {}\n".format(args.cnepochs))
        file.write("Convex Net Learning Rate: {}\n".format(args.cnlearningrate))
        file.write("Convex Net Hidden Units: {}\n".format(args.cnhiddenunits))

terminal_output_file = os.path.join(output_folder, "terminal_output.txt")
sys.stdout = open(terminal_output_file, "w")

# create figure
fig = plt.figure(figsize=(10, 7))
  
# setting values to rows and column variables
rows = 7
columns = 2

#Opening and printing the input image, foreground and background scribbles
img_org = open_print_image(args.inputpath+'input.png')
fig.add_subplot(rows, columns, 1)
# showing image
plt.imshow(img_org)
plt.axis('off')
plt.title("Input Image")

img_fg_scribble =open_print_image(args.inputpath+'scribble_fg.png')
fig.add_subplot(rows, columns, 3)
# showing image
plt.imshow(img_fg_scribble)
plt.axis('off')
plt.title("Image with Scribble on Foreground")

img_bg_scribble =open_print_image(args.inputpath+'scribble_bg.png')
fig.add_subplot(rows, columns, 4)
# showing image
plt.imshow(img_bg_scribble)
plt.axis('off')
plt.title('Image with Scribble on Background')

# Extracting fore and background masks of image for training
fg_mask = extract_mask(img_fg_scribble,1)
fig.add_subplot(rows, columns, 5)
# showing image
plt.imshow(fg_mask)
plt.axis('off')
plt.title('Foreground Mask')

bg_mask = extract_mask(img_bg_scribble,1)
fig.add_subplot(rows, columns, 6)
# showing image
plt.imshow(bg_mask)
plt.axis('off')
plt.title('Background Mask')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.primarymodel == 'neuralnet':
    fg_data,fg_data_len =extract_data_from_image(img_org,fg_mask)
    bg_data,bg_data_len =extract_data_from_image(img_org,bg_mask)
    fg_bg_data = np.concatenate((fg_data, bg_data), axis=1).T
    # print("no of foreground scribbles: ",fg_data_len)
    # print("no of background scribbles: ",bg_data_len)
    print("Total data shape for primary Network: ",fg_bg_data.shape)

    # Assigning fore ground labels to zero and back ground labels to 1
    fg_labels = np.zeros(fg_data_len)
    bg_labels = np.ones(bg_data_len)
    fg_bg_labels = np.concatenate((fg_labels, bg_labels))
    #print(fg_bg_labels.shape)

    # Split the features and labels into train and test sets
    nn_train_data, nn_test_data, nn_train_labels, nn_test_labels = train_test_split(Tensor(fg_bg_data), Tensor(fg_bg_labels), test_size=0.2, random_state=42)
    # Create dataset from several tensors with matching first dimension
    # Samples will be drawn from the first dimension (rows)
    nn_train_dataset = TensorDataset( nn_train_data, nn_train_labels )
    nn_test_dataset = TensorDataset( nn_test_data, nn_test_labels )
    # Create a data loader from the dataset
    # Type of sampling and batch size are specified at this step
    nn_trainloader = DataLoader(nn_train_dataset, args.nnbatchsize,shuffle=True)
    nn_valloader = DataLoader(nn_test_dataset, args.nnbatchsize,shuffle=False)
    
    # Create an instance of the network
    nn_model = NeuralNet(5,args.nnhiddenunits)

    # Define the binary cross-entropy loss
    nn_criterion = nn.BCELoss()

    # Define the optimizer (e.g. stochastic gradient descent)
    nn_optimizer = optim.AdamW(nn_model.parameters(), lr=args.nnlearningrate)

    nn_scheduler = optim.lr_scheduler.CosineAnnealingLR(nn_optimizer, T_max = args.nnepochs, eta_min=0, last_epoch=- 1, verbose=True)
    
    neuralnet_out_before_training = neuralnet_eval(nn_model,img_org)
    fig.add_subplot(rows, columns, 7)
    # showing image
    plt.imshow(neuralnet_out_before_training)
    plt.axis('off')
    plt.title('Neural Net Output Before Training')

    # Training and validation process
    nn_best_val_accuracy = 0.0
    nn_best_model_state = None

    for epoch in range(args.nnepochs):
        # Training
        nn_train_loss = nn_train(nn_model, nn_criterion, nn_optimizer, nn_trainloader)

        # Validation
        nn_val_loss, nn_val_accuracy = nn_validate(nn_model, nn_criterion, nn_valloader)

        # Save the model if the current validation accuracy is the best
        if nn_val_accuracy > nn_best_val_accuracy:
            nn_best_val_accuracy = nn_val_accuracy
            nn_best_model_state = nn_model.state_dict()

        # Step the scheduler
        nn_scheduler.step()
        
        # Print the progress
        print(f"Epoch [{epoch+1}/{args.nnepochs}] - Train Loss: {nn_train_loss:.4f} - Val Loss: {nn_val_loss:.4f} - Val Accuracy: {nn_val_accuracy:.4f}")
        
        writer.add_scalars("nn_graphs",{"nn_Loss/train" : nn_train_loss,"nn_Loss/val" : nn_val_loss,"nn_accuracy/val" : nn_val_accuracy}, epoch)

    # # Save the best model
    # torch.save(nn_best_model_state, 'nn_best_model.pth')

    neuralnet_out_after_training = neuralnet_eval(nn_model,img_org)
    fig.add_subplot(rows, columns, 8)
    # showing image
    plt.imshow(neuralnet_out_after_training)
    plt.axis('off')
    plt.title('Neural Net Output After Training')
    nn_segmented_fg_image, nn_segmented_bg_image = extract_segemented_images_from_inference(img_org, neuralnet_out_after_training)
    fig.add_subplot(rows, columns, 9)
    # showing image
    plt.imshow(nn_segmented_fg_image)
    plt.axis('off')
    plt.title('Neural Net Foreground segmented')

    fig.add_subplot(rows, columns, 10)
    # showing image
    plt.imshow(nn_segmented_bg_image)
    plt.axis('off')
    plt.title('Neural Net Background segmented')

    convex_data, convex_labels = convert_segmented_inference_to_covexnet_dataset(neuralnet_out_after_training)
    
elif args.primarymodel == 'segmentationnet':
    padded_image = add_padding_to_divisible_by(img_org, 32)
    padded_image = padded_image.astype(np.float32)
    # Apply the transformations needed
    trf = T.Compose([T.ToTensor(), 
                     T.Normalize(mean = [0.485, 0.456, 0.406], 
                                std = [0.229, 0.224, 0.225])])

    sn_input = trf(padded_image).unsqueeze(0)

    # Convert fg_mask and bg_mask to tensors
    fg_mask_tensor = torch.from_numpy(fg_mask)
    bg_mask_tensor = torch.from_numpy(bg_mask)

    # Collect indices of non-zero pixels from fg_mask_tensor and bg_mask_tensor
    fg_indices = torch.nonzero(fg_mask_tensor)
    bg_indices = torch.nonzero(bg_mask_tensor)

    # Create fg_labels and bg_labels tensors
    fg_labels = torch.zeros(len(fg_indices), dtype=torch.float32, requires_grad=True)
    bg_labels = torch.ones(len(bg_indices), dtype=torch.float32, requires_grad=True)

    # Create target_for_loss tensor
    target_for_loss = torch.cat((fg_labels, bg_labels))
    # print(target_for_loss)
    # print("target_for_loss shape",target_for_loss.shape)
    
    """using the segmentation_models_pytorch library to create a model for segmentation. It uses the ResNet34 encoder with pre-trained weights from ImageNet for encoder initialization."""
    if args.segmentationmodel == 'PSPNet':
        sn_model = smp.PSPNet(encoder_name='resnet34', encoder_weights='imagenet', 
                              encoder_depth=3, psp_out_channels=512, 
                              psp_use_batchnorm=False, #set false as we have only one image
                              psp_dropout=0.2, in_channels=3, 
                            classes=1, activation=None, upsampling=8, aux_params=None)
    elif args.segmentationmodel == 'Unetplusplus':
        sn_model = smp.UnetPlusPlus(encoder_name='resnet34', 
                                    encoder_depth=5, encoder_weights='imagenet', 
                                    decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), 
                                    decoder_attention_type=None, 
                                    in_channels=3, classes=1, activation=None, aux_params=None)
    elif args.segmentationmodel == 'MAnet':
        sn_model = smp.MAnet(encoder_name='resnet34', 
                             encoder_depth=5, encoder_weights='imagenet', 
                             decoder_use_batchnorm=True, 
                             decoder_channels=(256, 128, 64, 32, 16), 
                             decoder_pab_channels=64, 
                             in_channels=3, classes=1, activation=None, aux_params=None)
    elif args.segmentationmodel == 'Linknet':
        sn_model = smp.Linknet(encoder_name='resnet34', encoder_depth=5, 
                               encoder_weights='imagenet', decoder_use_batchnorm=True, 
                               in_channels=3, classes=1, activation=None, aux_params=None)
    elif args.segmentationmodel == 'FPN':
        sn_model = smp.FPN(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', 
                           decoder_pyramid_channels=256, decoder_segmentation_channels=128, 
                           decoder_merge_policy='add', decoder_dropout=0.2, 
                           in_channels=3, classes=1, activation=None, upsampling=4, aux_params=None)
    elif args.segmentationmodel == 'DeepLabV3Plus':
        sn_model = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_depth=5, 
                                     encoder_weights='imagenet', encoder_output_stride=16, 
                                     decoder_channels=256, decoder_atrous_rates=(12, 24, 36), 
                                     in_channels=3, classes=1, activation=None, upsampling=4, 
                                     aux_params=None)
    
    segmentationnet_out_before_training = segmentationet_eval(sn_model,sn_input,0.4)
    fig.add_subplot(rows, columns, 7)
    # showing image
    plt.imshow(segmentationnet_out_before_training)
    plt.axis('off')
    plt.title('Segmentation Net Output Before Training')

    sn_optimizer = optim.AdamW(sn_model.parameters(), lr=args.snlearningrate)
    sn_criterion = nn.BCEWithLogitsLoss()
    sn_scheduler = optim.lr_scheduler.CosineAnnealingLR(sn_optimizer, T_max = args.snepochs, eta_min=0, last_epoch=- 1, verbose=True)
    sn_input = sn_input.to(device)
    target_for_loss = target_for_loss.to(device)
    sn_model.to(device)
    for epoch in range(args.snepochs):
        sn_model.train()
        sn_output_mask = sn_model(sn_input)
        # Extract fg_pixels and bg_pixels from output_mask tensor
        sn_output_mask_fg_pixels = sn_output_mask[0, 0, fg_indices[:, 0], fg_indices[:, 1]]
        sn_output_mask_bg_pixels = sn_output_mask[0, 0, bg_indices[:, 0], bg_indices[:, 1]]

        # Concatenate fg_pixels and bg_pixels
        output_for_loss = torch.cat((sn_output_mask_fg_pixels, sn_output_mask_bg_pixels))
        # output_for_loss.to(device)
        #print("output_for_loss",output_for_loss)
        #print("output_for_loss shape",output_for_loss.shape)
        
        sn_optimizer.zero_grad()
        sn_model_loss = sn_criterion(output_for_loss, target_for_loss)
        sn_model_loss.backward()
        sn_optimizer.step()
        # Step the scheduler
        sn_scheduler.step()
        
        # Print the progress
        print(f"Epoch [{epoch+1}/{args.snepochs}] - Train Loss: {sn_model_loss:.4f}")

        writer.add_scalar("sn_Loss/train", sn_model_loss, epoch)
    sn_input = sn_input.to('cpu')
    target_for_loss = target_for_loss.to('cpu')
    sn_model.to('cpu')
    segmentationnet_out_after_training = segmentationet_eval(sn_model,sn_input)
    segmentationnet_out_after_training = segmentationnet_out_after_training[:img_org.shape[0],:img_org.shape[1]]
    fig.add_subplot(rows, columns, 8)
    # showing image
    plt.imshow(segmentationnet_out_after_training)
    plt.axis('off')
    plt.title('Segmentation Net Output After Training')
    sn_segmented_fg_image, sn_segmented_bg_image = extract_segemented_images_from_inference(img_org, segmentationnet_out_after_training)
    fig.add_subplot(rows, columns, 9)
    # showing image
    plt.imshow(sn_segmented_fg_image)
    plt.axis('off')
    plt.title('Segmentation Net Foreground Segmented')

    fig.add_subplot(rows, columns, 10)
    # showing image
    plt.imshow(sn_segmented_bg_image)
    plt.axis('off')
    plt.title('Segmentation Net Background Segmented')
    
    convex_data, convex_labels = convert_segmented_inference_to_covexnet_dataset(segmentationnet_out_after_training)

# Split the features and labels into train and test sets
cn_train_data, cn_test_data, cn_train_labels, cn_test_labels = train_test_split(Tensor(convex_data), Tensor(convex_labels), test_size=0.2, random_state=42)
# Create dataset from several tensors with matching first dimension
# Samples will be drawn from the first dimension (rows)
cn_train_dataset = TensorDataset( cn_train_data, cn_train_labels )
cn_test_dataset = TensorDataset( cn_test_data, cn_test_labels )
# Create a data loader from the dataset
# Type of sampling and batch size are specified at this step
cn_trainloader = DataLoader(cn_train_dataset, args.cnbatchsize,shuffle=True)
cn_valloader = DataLoader(cn_test_dataset, args.cnbatchsize,shuffle=False)

# Create an instance of the network
cn_model = ConvexNet(2,args.cnhiddenunits)
# print(cn_model)
# Define the binary cross-entropy loss
cn_criterion = nn.BCELoss()

# Define the optimizer (e.g. stochastic gradient descent)
cn_optimizer = optim.Adam(cn_model.parameters(), lr=args.cnlearningrate)

cn_scheduler = optim.lr_scheduler.CosineAnnealingLR(cn_optimizer, T_max = args.cnepochs, eta_min=0, last_epoch=- 1, verbose=True)

# Training and validation process
cn_best_val_accuracy = 0.0
cn_best_model_state = None

with torch.no_grad():
    cn_inference_bt = cn_model(Tensor(convex_data)).numpy().reshape((img_org.shape[0],img_org.shape[1]))

cn_inference_bt = np.where(cn_inference_bt < 0.5,0,1)
fig.add_subplot(rows, columns, 11)
# showing image
plt.imshow(cn_inference_bt)
plt.axis('off')
plt.title('Convex Net Output before Training')

for epoch in range(args.cnepochs):
    # Training
    cn_train_loss = cn_train(cn_model, cn_criterion, cn_optimizer, cn_trainloader)

    # Validation
    cn_val_loss, cn_val_accuracy = cn_validate(cn_model, cn_criterion, cn_valloader)

    # Save the model if the current validation accuracy is the best
    if cn_val_accuracy > cn_best_val_accuracy:
        cn_best_val_accuracy = cn_val_accuracy
        cn_best_model_state = cn_model.state_dict()

    # Step the scheduler
    cn_scheduler.step()
    
    # Print the progress
    print(f"Epoch [{epoch+1}/{args.cnepochs}] - Train Loss: {cn_train_loss:.4f} - Val Loss: {cn_val_loss:.4f} - Val Accuracy: {cn_val_accuracy:.4f}")
    
    writer.add_scalars("cn_graphs",{"cn_Loss/train" : cn_train_loss,"cn_Loss/val" : cn_val_loss,"cn_accuracy/val" : cn_val_accuracy}, epoch)

# # Save the best model
# torch.save(cn_best_model_state, 'cn_best_model.pth')

with torch.no_grad():
    cn_inference = cn_model(Tensor(convex_data)).numpy().reshape((img_org.shape[0],img_org.shape[1]))

cn_inference = np.where(cn_inference < 0.5,0,1)
fig.add_subplot(rows, columns, 12)
# showing image
plt.imshow(cn_inference)
plt.axis('off')
plt.title('Convex Net Output After Training')

cn_segmented_fg_image, cn_segmented_bg_image = extract_segemented_images_from_inference(img_org, cn_inference)

fig.add_subplot(rows, columns, 13)
# showing image
plt.imshow(cn_segmented_fg_image)
plt.axis('off')
plt.title('Convex Net Foreground segmented')

fig.add_subplot(rows, columns, 14)
# showing image
plt.imshow(cn_segmented_bg_image)
plt.axis('off')
plt.title('Convex Net Background segmented')

# Display the figure with all the subplots
plt.tight_layout()



# Save the plot to the new output folder
plt.savefig(os.path.join(output_folder, "result.png"))

writer.close()