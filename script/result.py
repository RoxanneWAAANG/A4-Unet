import torch
from pathlib import Path
from evaluate import evaluate
from a4unet.dataloader.bratsloader import BRATSDataset3D  # Replace with your actual data loading module
from collections import OrderedDict

# Ensure that your model class is imported here
from a4unet.a4unet import create_a4unet_model  # Replace with your actual model module

# Paths and settings
dir_brats = '/autodl-tmp/MICCAI_BraTS2020_TrainingData'
out_files = 'env_result/'
checkpoint_path = Path('checkpoint_epoch34.pth')  # Replace with the actual path

# Initialize the model and load the checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_a4unet_model(image_size=128, num_channels=128, num_res_blocks=2, num_classes=2, learn_sigma=True, in_ch=4)
# new_state_dict = OrderedDict()
# state_dict = torch.load(checkpoint_path)
# for s, v in state_dict.items():
#     name = s
#     if s == 'mask_values':
#         continue
#     new_state_dict[name] = v
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# Replace with your actual data loading logic
train_list = [transforms.Resize((128, 128), antialias=True)]
transform_train = transforms.Compose(train_list)
dataloader = BRATSDataset3D(dir_brats, transform_train, test_flag=False)
datasets = False  # Set this to True if your dataset has additional information, else set to False
final_test = True  # Set this to True if you want to save predicted masks, else set to False

# Evaluate the model
dice_score, mIoU, hd95 = evaluate(model, dataloader, device, amp=False, datasets=datasets, final_test=final_test)

# Print or save the evaluation results
print(f'Dice Score: {dice_score}')
print(f'mIoU: {mIoU}')
print(f'HD95: {hd95}')

# Note: If you want to save the predicted masks, they will be saved according to the logic inside the evaluate function.
