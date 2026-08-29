from PIL import Image, ImageDraw
import os

def visualize_result(image_path, pred_box, scale, save_path, original_size=(512, 512)):
    """
    Draws the predicted bounding box on the image and saves it with 
    the EXACT original (w, h) dimensions.
    
    Args:
        image_path (str): Path to original image.
        pred_box (list): [x, y, w, h] normalized to [-1, 1].
        scale (float): The resize scale used in preprocessing.
        save_path (str): Where to save the result.
        original_size (tuple): The (width, height) used for the model input.
                                  Example: (512, 512) or (640, 480).
    """
    # 1. Load Image to get original dimensions
    img = Image.open(image_path).convert("RGB")
    # w_source, h_source = img.size
    # if original_size is None:
    #     original_size = (w_source, h_source)
    draw = ImageDraw.Draw(img)
    
    # 2. Denormalize Coordinates
    # unpack normalized box (Range: -1 to 1)
    x_norm, y_norm, w_norm, h_norm = pred_box
    target_w, target_h = original_size
    
    # Shift to [0, 1] then scale to the Model Input Dimension
    # (We use target_w for x/w and target_h for y/h)
    x_in_model = ((x_norm + 1) / 2) * target_w
    y_in_model = ((y_norm + 1) / 2) * target_h
    w_in_model = ((w_norm + 1) / 2) * target_w
    h_in_model = ((h_norm + 1) / 2) * target_h
    
    # 3. Scale back to Original Image Space
    # We divide by 'scale' to undo the resizing.
    # Note: This assumes Top-Left padding (0,0). 
    # If you used Centered padding, you would subtract the padding offset here first.
    x = x_in_model / scale
    y = y_in_model / scale
    w = w_in_model / scale
    h = h_in_model / scale
    
    # 4. Convert Center-Format (x,y) to Top-Left-Format (x1,y1,x2,y2)
    # The model predicts the center of the box. PIL needs corners.
    x1 = x - (w / 2)
    y1 = y - (h / 2)
    x2 = x + (w / 2)
    y2 = y + (h / 2)
    
    # 5. Draw
    # width=5 makes the line thick enough to see on high-res images
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=5)
    
    # 6. Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)
    print(f"Saved result with size {img.size} to: {save_path}")