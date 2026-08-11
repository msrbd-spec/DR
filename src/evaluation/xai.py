import cv2
import numpy as np
from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch

def get_target_layer(model):
    """
    Dynamically finds the appropriate target layer for CAM methods.
    Checks model.named_modules() to ensure we target the exact nested attribute.
    If MSDA is used, we target the final MSDA block's deform_conv.
    Otherwise, we target the final Swin-V2 stage block's norm layer.
    """
    target_layer = None
    
    # 1. Try to find the final MSDA block if MSDA is enabled
    # We look for msda4.deform_conv
    for name, module in model.named_modules():
        if name == 'msda4.deform_conv':
            target_layer = module
            break
            
    if target_layer is not None:
        return [target_layer]
        
    # 2. If MSDA is not enabled, find the final block of the Swin-V2 backbone
    # The name could be 'backbone.layers.3.blocks.1.norm2' depending on Swin configuration
    # We will iterate and find the deepest 'norm2' in the last layer
    last_norm_name = None
    last_norm_module = None
    for name, module in model.named_modules():
        if 'backbone.layers.3.blocks' in name and 'norm2' in name:
            last_norm_name = name
            last_norm_module = module
            
    if last_norm_module is not None:
        return [last_norm_module]
        
    raise ValueError("Could not find a suitable target layer for XAI. Please inspect model.named_modules().")

def generate_heatmap(image_tensor, model, original_image, method='EigenCAM', out_name='xai_heatmap.png'):
    """
    Generates and saves the attention heatmap.
    
    Args:
        image_tensor (torch.Tensor): Preprocessed image tensor (1, C, H, W).
        model (nn.Module): The trained ICCIT_DR_Net model.
        original_image (np.ndarray): Original image in RGB (H, W, 3), values in [0, 1].
        method (str): 'EigenCAM' or 'GradCAM'.
    """
    model.eval()
    
    target_layers = get_target_layer(model)
    
    if method == 'EigenCAM':
        cam = EigenCAM(model=model, target_layers=target_layers)
    else:
        cam = GradCAM(model=model, target_layers=target_layers)
        
    # Generate the heatmap
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)
    
    # In this case, grayscale_cam has shape (batch_size, H, W)
    grayscale_cam = grayscale_cam[0, :]
    
    # Resize original_image if it doesn't match the CAM size
    if original_image.shape[:2] != grayscale_cam.shape:
        original_image = cv2.resize(original_image, (grayscale_cam.shape[1], grayscale_cam.shape[0]))
        
    # Overlay the cam on the image
    visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
    
    # Save the visualization
    cv2.imwrite(out_name, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    return visualization
