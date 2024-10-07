print("Start Transfer Learning")
##############################################################################################################################################################
from Classification_3D import UNetForClassification

from monai.networks.nets import UNet
import torch
##############################################################################################################################################################

# STEP 1: Creating instances of the classification and segmentation models (FRAGE !! DONE!)
classification_model = UNetForClassification()  
segmentation_model = UNet(
    spatial_dims=3,
    in_channels=2,  # Adjust this based on how many input channels your segmentation tasks use
    out_channels=1,  # Typically 1 for binary segmentation tasks
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm='BATCH' #oder norm = Norm.Batch
).to(device) # oder to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


# STEP 2: Loading the pretrained/saved weights into the classification model from a file. (FRAGE !! DONE!)
# path = path_to_classification_model_weights.pth
path = "/lustre/groups/iterm/Hazem/MA/Runs/run_27339286__Phase_2/best_metric_model_classification3d_array_27339286.pth" #really the best one --> testing score = 75% (Evaluation score = 83%)
classification_model.load_state_dict(torch.load(path))

# STEP 3: Transfer Weights
# Transfer the weights from the classification model to the segmentation model (skipping any non-matching layers like the final layer OR focusing ONLY on the encoder OR ONLY the encoder&bottle-neck)

# STEP 3.1.: Copy the state_dict from classification_model
state_dict = classification_model.state_dict()
# an example approach to filter out only encoder weights (LATER)
# state_dict = {key: value for key, value in classification_model.state_dict().items() if 'fc' not in key}  # Assuming 'fc' is the fully connected layer

# # STEP 3.2.: Remove the final layer weights from the state_dict if they do not match in segmentation_model
# # Adjust the key names based on your actual model's layer names (FRAGE !!)
# state_dict.pop('fc.weight', None)  # Remove fully connected layer weights
# state_dict.pop('fc.bias', None)

# HOPEFULLY, the solution:
# when accessing the state_dict() of the model, each tensor (layer weights) is associated with a specific name that corresponds to its path within the model's architecture.
# for Transfer Learning, map the names from one state_dict to another if they differ. This is often necessary when the two models are not exactly the same but share some architectural similarities (e.g., the same types of layers in the encoder).

# print the keys of the state_dict for both models to see what the layer names are in the 2 models 
# This will give a list of all the parameter names and help identify which ones correspond to each other between the two models. 
# for name, param in classification_model.state_dict().items():
#     print(name)
print("Classification Model Parameters:") # Print parameter names and shapes for the classification model
for name, param in classification_model.named_parameters(): #This function iterates over each parameter in the model, where it returns a tuple containing the name of the parameter and the parameter tensor itself.
    print(f"{name}: {param.shape}") # param.shape returns the shape of the tensor, which is CRUCIAL for understanding how parameters align between the two models.
# iterates over all the parameters that are intended to be updated during training (i.e., parameters for which gradients will be computed).
# often used when it is needed to manipulate or specifically check trainable parameters, such as freezing certain layers or applying different strategies for different parameters during optimization.

# for name, param in segmentation_model.state_dict().items(): #print out all the parameters (weights and biases) in the model's state dictionary, including those that might not require gradients (i.e., non-trainable parameters if any).
#     print(name) #It returns a dictionary containing a whole state of the module, including not only parameters but potentially buffers and other states as well.
print("Segmentation Model Parameters:")
for name, param in segmentation_model.named_parameters(): # Print parameter names and shapes for the segmentation model
    print(f"{name}: {param.shape}")

# knowing the names, check the shape of the weights in both: the segmentation model and the classification model
# shape of weights/layer weights/tensor --> ChannelsxBatchxHeightxWidthxDepth (Batch is not important. The most relevant here is "Channels" for both: in_channels & out_channels)
# the potential challenge could be because of the different values in these 2 U-Net parameters: in_channels (seg: 2 & class: 1) & out_channels (seg: 1 & class: 32)




# # STEP 3.3.: Load the modified state_dict into segmentation_model
# # Removing Non-matching Weights --> If you wanna load weights where some keys do not match, you can either modify the keys of the incoming state dict or
# # ignore non-matching keys by using strict=False in the load_state_dict method
# segmentation_model.load_state_dict(state_dict, strict=False)



# # STEP 4: Optionally Freeze Layers (If you wish to freeze the transferred layers during training to keep the pre-trained features intact)
# # Preventing the backpropagation from updating the transferred weights during the training of the new task, which can be useful to preserve learned features and stabilize early training stages.
# for name, param in segmentation_model.named_parameters():
#     if 'output_layer' not in name:  # Assuming 'output_layer' is the segmentation-specific layer
#         param.requires_grad = False  # Freeze this parameter







##############################################################################################################################################################
# possibble outcomes and solution for undesired scenario:
# 1. Compare Shapes: Look at the output of this script to compare the shapes of corresponding layers in the two models. The shapes will tell which
# layers can potentially have their weights transferred directly.
# 2.
# 2.1. If the shapes match, transfer learning can be DIRECTLY applied.
# 2.2. If they don't, some engineering will be needed! --> either skip those layers (nah!) or adapt the weights programmatically (by averaging or reshaping).

# what engineering will be needed?
# 




# state_dict().items VS named_parameters()
# use named_parameters() --> If ONLY interested in the trainable parameters (those for which gradients are computed).
#   This is useful, for example, when you want to freeze certain layers or apply different optimization strategies.
#   This directly addresses the need to understand the trainable parameters within your model, which are the main focus when considering transfer learning.
# use state_dict().items() --> when you want to save/load the model state, OR when you need a comprehensive view of all parameters and buffers,
#   regardless of whether they are trainable.
# VORGEHEN --> for checking the names & shapes, I will use named_parameters(). THEN at saving/loading the model state, I will probably use state_dict().
