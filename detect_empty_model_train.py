import torch
from PIL import Image, ImageFile
from pathlib import Path
from detect_empty_wells import EmptyWellModel 
from tqdm import tqdm

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class WellDataLoader(torch.utils.data.Dataset):
    def __init__(self, images, empty_well_images): # images is a list of image path, empty_well_images is a list of images that are empty_well
       self.images = images
       self.empty_well_images = empty_well_images
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], int(self.images[idx] in self.empty_well_images)
DATASET_DIR = Path("embryo_dataset")
images = []
    if DATASET_DIR.exists():
        for root, dirs, files in os.walk(DATASET_DIR):
            for file in files:
                if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, DATASET_DIR)
                    images.append(relative_path)
EMPTY_WELLS_FILE = Path("true_images.txt")
empty_well_images = []
with open(EMPTY_WELLS_FILE, 'r') as f:
    empty_well_images = f.readlines()
dataset = WellDataLoader(images, empty_well_images)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 25, shuffle = True, num_workers = 4)
MODEL_WEIGHTS_FILE = "empty_well_model_weights.pth")
model = EmptyWellModel().to(DEVICE)
if(os.path.exists(MODEL_WEIGHTS_FILE):
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE , weights_only=True))
    except Error:
        print("model has wrong shape") 
        torch.save(model.model.state_dict(), MODEL_WEIGHTS_FILE)
else:
    torch.save(model.model.state_dict(), MODEL_WEIGHTS_FILE)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(4):
    model.train()
    print("Epoch: ", str(epoch))
    for img, label in tqdm(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    torch.save(model.model.state_dict(), MODEL_WEIGHTS_FILE)
    
    
