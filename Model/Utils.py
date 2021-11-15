import torch
import torch
from skimage.io import imread
from torch.utils import data

class Option():
    """Configuration for training on Kaggle Data Science Bowl 2018
    Derived from the base Config class and overrides specific values
    """
    patchsize = 128
    num_workers = 4     	# number of threads for data loading
    shuffle = True      	# shuffle the data set
    batch_size = 16     	# GTX1060 3G Memory
    learning_rate = 1e-4    # learning rate
    weight_decay = 1e-4	# weight decay
    epochs = 50  # number of epochs to train
    Poled_train_path = r"C:\Users\guany\PycharmProjects\UDC\UDC_train\Train\Poled"
    Toled_train_path = r"C:\Users\guany\PycharmProjects\UDC\UDC_train\Train\Toled"
    DE_UNet_model_save_path = r"Model/Toled_DE_UNet_weight_decay_1e-4.pth"
    UNet_model_save_path = r"Model/UNet_weight_decay_1e-4.pth"
    model_save_path=r"Model/Toled_DE_UNet_weight_decay_1e-4.pth"
    logs_path=r"train_loss/DEUNet_weight_decay_1e-4_process.txt"

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = imread(input_ID), imread(target_ID)

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y



def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    """
    Center-crops the encoder_layer to the size of the decoder_layer,
    so that merging (concatenation) between levels/blocks is possible.
    This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
    """
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4:  # 2D
            encoder_layer = encoder_layer[
                :,
                :,
                ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
            ]
        elif encoder_layer.dim() == 5:  # 3D
            assert ds[2] >= es[2]
            encoder_layer = encoder_layer[
                :,
                :,
                ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2),
                ((ds[2] - es[2]) // 2):((ds[2] + es[2]) // 2),
            ]
    return encoder_layer, decoder_layer
