import numpy as np
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.autograd import Variable

from models import get_model_by_name, DATASETS

from PIL import Image

# EigenCAM timed out, others not included didn't work
# HiResCam was the best
from utility import get_half

other_cams = [GradCAM, GradCAMPlusPlus, XGradCAM]


def compute_gradcam(cam, input_tensor, label):
    """
    Computes gradcam on an input image.

    :param model: Model to compute with
    :param target_layers: Target layer of model
    :param input_tensor: Batch of [batch_size] copies of the input image. All images in batch should be identical.
    :param use_cuda: Whether to use cuda.
    :return: An rgb image with overlaid cam
    """

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.

    targets = [ClassifierOutputTarget(labels)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    input_img = input_tensor[0][0].numpy()

    # Normalize input image
    if np.max(input_img) > 1:
        input_img = np.divide(input_img, 255)

    # Order the dimensions correctly
    if input_img.shape != (256, 256, 3):
        input_img = np.transpose(input_img, (1, 2, 0))

    visualization = show_cam_on_image(input_img, grayscale_cam, use_rgb=True)
    return visualization


if __name__ == "__main__":
    print("Initializing model...")
    cuda_available = torch.cuda.is_available()
    model = get_model_by_name('prednet', class_weight=0.9, rec_weight=0.1, nt=10, noise_intensities=None, noise_type=None)
    model_path = f"./{model.get_name()}/model.pt"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda') if cuda_available else torch.device('cpu')))
    target_layers = [getattr(model, "conv{}".format(i)) for i in range(4)]  # Can I use gradcam on a sequential layer?
    print("Finished initializing model. Initializing dataset...")

    # Compute gradcam on model
    cam = HiResCAM
    current_cam = cam(model=model, target_layers=target_layers, use_cuda=cuda_available)

    dataset = DATASETS['CK']
    data_loader = dataset(10, False)
    input_tensor = None
    for i, (inputs, labels) in enumerate(data_loader):
        input_tensor = inputs
        input_tensor = Variable(input_tensor.to(torch.device('cuda') if cuda_available else torch.device('cpu')))
        input_tensor = get_half(input_tensor, half='top', dim=2)
        input_tensor = torch.unsqueeze(input_tensor, 0)
        result = compute_gradcam(current_cam, input_tensor, labels)
        Image.Image.show(Image.fromarray(result))

    print("Finished initializing dataset. Computing gradcam...")
