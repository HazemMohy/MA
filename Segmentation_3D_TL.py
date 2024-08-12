
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


# STEP 2: Loading the pretrained/saved weights into the classification model from a file. (FRAGE !!)
classification_model.load_state_dict(torch.load('path_to_classification_model_weights.pth'))

# STEP 3: Transfer Weights
# Transfer the weights from the classification model to the segmentation model (skipping any non-matching layers like the final layer OR focusing ONLY on the encoder OR ONLY the encoder&bottle-neck)

# STEP 3.1.: Copy the state_dict from classification_model
state_dict = classification_model.state_dict()

# STEP 3.2.: Remove the final layer weights from the state_dict if they do not match in segmentation_model
# Adjust the key names based on your actual model's layer names (FRAGE !!)
state_dict.pop('fc.weight', None)  # Remove fully connected layer weights
state_dict.pop('fc.bias', None)

# HOPEFULLY, the solution:
# for name, param in classification_model.state_dict().items():
#     print(name)

# for name, param in segmentation_model.state_dict().items():
#     print(name)



# STEP 3.3.: Load the modified state_dict into segmentation_model
segmentation_model.load_state_dict(state_dict, strict=False)



# STEP 4: Optionally Freeze Layers (If you wish to freeze the transferred layers during training to keep the pre-trained features intact)
# Preventing the backpropagation from updating the transferred weights during the training of the new task, which can be useful to preserve learned features and stabilize early training stages.
for name, param in segmentation_model.named_parameters():
    if 'output_layer' not in name:  # Assuming 'output_layer' is the segmentation-specific layer
        param.requires_grad = False  # Freeze this parameter







##############################################################################################################################################################