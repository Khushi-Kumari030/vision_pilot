import torch
from torchvision import transforms
from Models.model_components.ego_lanes_network import EgoLanesNetwork
import numpy as np


class EgoLanesNetworkInfer():
    def __init__(
            self, 
            checkpoint_path: str = ""
    ):

        # Image loader
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406], 
                    std  = [0.229, 0.224, 0.225]
                )
            ]
        )
            
        # Checking devices (GPU vs CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Using {self.device} for inference")
            
        # Init model
        self.model = EgoLanesNetwork()
        if (checkpoint_path):
            print(f"Loading trained AutoSteer checkpoint: {checkpoint_path}")
            self.model.load_state_dict(
                torch.load(
                    checkpoint_path, 
                    weights_only = True,
                    map_location = self.device
                )
            )
        else:
            print("Loading vanilla AutoSteer model for training")
        
        # Load model to device, set eval mode
        self.model.to(self.device)
        self.model.eval()

    
    def inference(self, image):

        image_tensor = self.image_loader(image).unsqueeze(0).to(self.device)

        prediction = self.model(image_tensor)

        binary_seg = prediction.squeeze(0).cpu().detach().numpy()

        print("\nPrediction shape:", binary_seg.shape)

        for i in range(binary_seg.shape[0]):
            print(
                f"Lane {i}:",
                "min=", float(binary_seg[i].min()),
                "max=", float(binary_seg[i].max()),
                "mean=", float(binary_seg[i].mean()),
                "std=", float(binary_seg[i].std())
            )

        return binary_seg