import torch
from pathlib import Path
from evaluate import evaluate
from a4unet.dataloader.bratsloader import BRATSDataset3D  # Replace with your actual data loading module
from collections import OrderedDict
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

# Ensure that your model class is imported here
from a4unet.a4unet import create_a4unet_model  # Replace with your actual model module

# Paths and settings
dir_brats = 'C:/Users/admin/Desktop/Spring2025/AIPI540/CV/MICCAI_BraTS2020_TrainingData'
dir_val = 'testset'
out_files = 'C:/Users/admin/Desktop/Spring2025/AIPI540/CV/A4-Unet/evaluate_result'
checkpoint_path = Path('C:/Users/admin/Desktop/Spring2025/AIPI540/CV/A4-Unet/checkpoints/checkpoint_epoch5.pth')  # Replace with the actual path

# Initialize the model and load the checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_a4unet_model(image_size=128, num_channels=128, num_res_blocks=2, num_classes=2, learn_sigma=True, in_ch=4)
new_state_dict = OrderedDict()
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

for s, v in state_dict.items():
    name = s
    if s == 'mask_values':
        continue
    new_state_dict[name] = v
state_dict = new_state_dict

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# Define transformations for the dataset
train_list = [transforms.Resize((128, 128), antialias=True)]
transform_train = transforms.Compose(train_list)

# Load the BRATS 3D dataset
dataloader = BRATSDataset3D(dir_val, transform_train, test_flag=False)

# Set evaluation flags
datasets = 'Brats'
final_test = True

# Evaluate the model
dice_score, mIoU, hd95 = evaluate(model, dataloader, device, amp=False, datasets=datasets, final_test=final_test)

# Print evaluation results
print(f'Dice Score: {dice_score}')
print(f'mIoU: {mIoU}')
print(f'HD95: {hd95}')

# Note: If you want to save the predicted masks, they will be saved according to the logic inside the evaluate function.
