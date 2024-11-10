import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Callable
from dataclasses import dataclass

from serl_launcher.vision.resnet_v1 import resnetv1_configs, PreTrainedResNetEncoder
from serl_launcher.common.encoding import EncodingWrapper


class BinaryClassifier(nn.Module):
    def __init__(self, encoder_def: nn.Module, hidden_dim: int = 256):
        super().__init__()
        self.encoder_def = encoder_def
        self.hidden_dim = hidden_dim
        
        self.dense1 = nn.Linear(encoder_def.output_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        x = self.encoder_def(x, train=train)
        x = self.dense1(x)
        x = self.dropout(x) if train else x
        x = self.layer_norm(x)
        x = torch.relu(x)
        x = self.dense2(x)
        return x


class NWayClassifier(nn.Module):
    def __init__(self, encoder_def: nn.Module, hidden_dim: int = 256, n_way: int = 3):
        super().__init__()
        self.encoder_def = encoder_def
        self.hidden_dim = hidden_dim
        self.n_way = n_way
        
        self.dense1 = nn.Linear(encoder_def.output_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, n_way)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        x = self.encoder_def(x, train=train)
        x = self.dense1(x)
        x = self.dropout(x) if train else x
        x = self.layer_norm(x)
        x = torch.relu(x)
        x = self.dense2(x)
        return x


@dataclass
class TrainState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    
    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])


def create_classifier(
    device: torch.device,
    sample: Dict,
    image_keys: List[str],
    pretrained_encoder_path: str = "../resnet10_params.pkl",
    n_way: int = 2,
) -> TrainState:
    pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
        pre_pooling=True,
        name="pretrained_encoder",
    )
    encoders = {
        image_key: PreTrainedResNetEncoder(
            pooling_method="spatial_learned_embeddings",
            num_spatial_blocks=8,
            bottleneck_dim=256,
            pretrained_encoder=pretrained_encoder,
            name=f"encoder_{image_key}",
        )
        for image_key in image_keys
    }
    encoder_def = EncodingWrapper(
        encoder=encoders,
        use_proprio=False,
        enable_stacking=True,
        image_keys=image_keys,
    )
    
    if n_way == 2:
        classifier_def = BinaryClassifier(encoder_def=encoder_def)
    else:
        classifier_def = NWayClassifier(encoder_def=encoder_def, n_way=n_way)
    
    classifier_def = classifier_def.to(device)
    optimizer = optim.Adam(classifier_def.parameters(), lr=1e-4)
    
    # Load pretrained encoder weights
    with open(pretrained_encoder_path, "rb") as f:
        encoder_params = pkl.load(f)
    param_count = sum(p.numel() for p in encoder_params.values())
    print(
        f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
    )
    
    # Update encoder parameters
    state_dict = classifier_def.state_dict()
    for image_key in image_keys:
        encoder_prefix = f"encoder_def.encoder_{image_key}.pretrained_encoder."
        for k, v in encoder_params.items():
            if k in state_dict[encoder_prefix]:
                state_dict[encoder_prefix + k] = torch.tensor(v)
                print(f"replaced {k} in encoder_{image_key}")
    
    classifier_def.load_state_dict(state_dict)
    return TrainState(model=classifier_def, optimizer=optimizer)


def load_classifier_func(
    device: torch.device,
    sample: Dict,
    image_keys: List[str],
    checkpoint_path: str,
    n_way: int = 2,
) -> Callable[[Dict], torch.Tensor]:
    """
    Return: a function that takes in an observation
            and returns the logits of the classifier.
    """
    classifier = create_classifier(device, sample, image_keys, n_way=n_way)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classifier.load_state_dict(checkpoint)
    
    classifier.model.eval()
    
    def inference_fn(obs: Dict) -> torch.Tensor:
        with torch.no_grad():
            return classifier.model(obs, train=False)
    
    return inference_fn
