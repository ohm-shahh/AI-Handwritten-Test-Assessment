import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def deskew_image(gray_img):
    """
    Detects the skew angle of the text and rotates the image to straighten it.
    Uses a downscaled thumbnail to calculate angle with minimal memory footprint.
    """
    try:
        h, w = gray_img.shape[:2]
        
        # Downscale thumbnail for skew angle calculation (prevents huge memory allocation)
        max_thumb = 400
        scale = max_thumb / max(h, w) if max(h, w) > max_thumb else 1.0
        if scale < 1.0:
            thumb = cv2.resize(gray_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            thumb = gray_img
            
        # Binarize inverted thumbnail to find text pixels
        thresh = cv2.threshold(thumb, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Get coordinates of text pixels
        pts = cv2.findNonZero(thresh)
        if pts is None or len(pts) < 10:
            return gray_img, 0.0
            
        # Calculate skew angle using minAreaRect
        rect = cv2.minAreaRect(pts)
        angle = rect[-1]
        
        # OpenCv minAreaRect returns angle in range [-90, 0) in older versions
        # and [0, 90] in newer ones. Normalize it.
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle
            
        # Ignore extremely small skew angles to prevent unnecessary interpolation artifacts
        if abs(angle) < 0.5:
            return gray_img, 0.0
            
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            gray_img, M, (w, h), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REPLICATE
        )
        logger.info(f"Successfully deskewed image. Calculated skew angle: {angle:.2f} degrees")
        return rotated, round(float(angle), 2)
    except Exception as e:
        logger.warning(f"Deskewing skipped or encountered minor issue: {str(e)}")
        return gray_img, 0.0

def preprocess_pipeline(
    image_bytes: bytes,
    apply_grayscale: bool = True,
    apply_denoise: bool = True,
    apply_threshold: bool = True,
    apply_deskew: bool = True
):
    """
    Runs the OpenCV preprocessing pipeline on the input image bytes.
    Optimized for low-memory environments and fast OCR execution.
    Returns:
        preprocessed_image (numpy array),
        metadata (dict containing skew angle, original size, etc.)
    """
    metadata = {}
    
    # Load image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image bytes provided or image format not supported.")
        
    orig_h, orig_w = img.shape[:2]
    metadata["original_size"] = {"width": orig_w, "height": orig_h}
    
    # Safe dimension scaling to avoid multi-gigabyte RAM spikes on smartphone 12MP+ photos
    max_safe_dim = 1600
    if max(orig_h, orig_w) > max_safe_dim:
        scale = max_safe_dim / max(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 1. Grayscale Conversion
    if apply_grayscale:
        if len(img.shape) == 3:
            processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            processed = img.copy()
    else:
        processed = img.copy()
        
    # 2. Denoising / Smoothing (lightweight & memory safe)
    if apply_denoise:
        if len(processed.shape) == 2:
            processed = cv2.bilateralFilter(processed, 5, 50, 50)
        else:
            processed = cv2.bilateralFilter(processed, 5, 50, 50)
            
    # 3. Deskewing
    metadata["skew_angle"] = 0.0
    if apply_deskew:
        if len(processed.shape) == 2:
            processed, angle = deskew_image(processed)
            metadata["skew_angle"] = angle
        else:
            gray_temp = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            _, angle = deskew_image(gray_temp)
            metadata["skew_angle"] = angle
            if abs(angle) >= 0.5:
                (h, w) = processed.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                processed = cv2.warpAffine(
                    processed, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )

    # 4. Adaptive Thresholding (Binarisation)
    if apply_threshold:
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Gaussian thresholding
        processed = cv2.adaptiveThreshold(
            processed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15, 8
        )
        
    metadata["processed_size"] = {"width": processed.shape[1], "height": processed.shape[0]}
    
    return processed, metadata

