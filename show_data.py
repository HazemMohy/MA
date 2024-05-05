#see the whole patch!!

import nibabel as nib
import matplotlib.pyplot as plt

# Replace 'path_to_your_nii_file.nii' with the path to your .nii file
#file_path = r'C:\Users\PC\Desktop\MA\data\patchvolume_13.nii'
file_path = r'C:\Users\PC\Desktop\MA\First Phase\2x\bg\patchvolume_10.nii'

# Load the image file
nii_image = nib.load(file_path)

# Get the data array from the image
image_data = nii_image.get_fdata()

# Get the dimensions of the image data
print(f"Size of each image slice: {image_data.shape[0]} x {image_data.shape[1]}")
print(f"Number of slices available: {image_data.shape[2]}")

# Check if there are at least 50 slices
num_slices = min(50, image_data.shape[2])

# Plotting the first 50 or fewer slices
fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(15, 7.5))  # Adjust size as needed

for i, ax in enumerate(axes.flatten()):
    if i < num_slices:
        ax.imshow(image_data[:, :, i], cmap='gray')
        ax.axis('off')  # Hide axes
    else:
        ax.axis('off')  # Hide axes for empty subplots

plt.tight_layout()
plt.show()


#see each slice alone!!
import nibabel as nib
import matplotlib.pyplot as plt

# Replace 'path_to_your_nii_file.nii' with the path to your .nii file
#file_path = r'C:\Users\PC\Desktop\MA\data\patchvolume_13.nii'
file_path = r'C:\Users\PC\Desktop\MA\First Phase\2x\bg\patchvolume_10.nii'

# Load the image file
nii_image = nib.load(file_path)

# Get the data array from the image
image_data = nii_image.get_fdata()

# Display the first image slice
# Adjust the index if you want to see a different slice
plt.imshow(image_data[:, :, 0], cmap='gray')
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()
