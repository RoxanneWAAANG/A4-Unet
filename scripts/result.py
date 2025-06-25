import torch
from pathlib import Path
import torchvision.transforms as transforms
from evaluate import evaluate
from a4unet.dataloader.bratsloader import BRATSDataset3D
from a4unet.a4unet import create_a4unet_model

# Directory paths for datasets and checkpoints
dir_brats = './datasets/brats/2020'
out_files = './result'
checkpoint_path = Path('./checkpoints')  # Replace with the actual path to the checkpoint

# Set up device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model
model = create_a4unet_model(image_size=128, num_channels=128, num_res_blocks=2, num_classes=2, learn_sigma=True, in_ch=4)

# Load the model weights from the checkpoint
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# Define transformations for the dataset
train_list = [transforms.Resize((128, 128), antialias=True)]
transform_train = transforms.Compose(train_list)

# Load the BRATS 3D dataset
dataloader = BRATSDataset3D(dir_brats, transform_train, test_flag=False)

# Set evaluation flags
datasets = False
final_test = True

# Evaluate the model
dice_score, mIoU, hd95 = evaluate(model, dataloader, device, amp=False, datasets=datasets, final_test=final_test)

# Print evaluation results
print(f'Dice Score: {dice_score}')
print(f'mIoU: {mIoU}')
print(f'HD95: {hd95}')