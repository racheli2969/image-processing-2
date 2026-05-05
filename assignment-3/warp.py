import numpy as np
import cv2

def warp_image(image: np.ndarray,
               angle_deg: float,
               scale_x: float,
               scale_y: float) -> np.ndarray:
    """
    Apply affine transformation to image with rotation and scaling around center.
    Vectorized implementation for fast performance.
    
    Uses backward mapping with bilinear interpolation.
    Pixel centers are at (j+0.5, i+0.5) in continuous space.
    
    Args:
        image: Input image (BGR, HxWxC)
        angle_deg: Rotation angle in degrees (counter-clockwise)
        scale_x: Scaling factor in x direction
        scale_y: Scaling factor in y direction
    
    Returns:
        Transformed image
    """
    
    H, W, C = image.shape
    
    # 1. Compute center of image
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    
    # 2. Build rotation matrix R(theta)
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    R = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ], dtype=np.float32)
    
    # 3. Build scaling matrix S(sx, sy)
    S = np.array([
        [scale_x, 0],
        [0, scale_y]
    ], dtype=np.float32)
    
    # 4. Compose A = R @ S (rotation followed by scaling)
    A = R @ S
    
    # 5. Build full affine transformation matrix
    # M = T(cx, cy) @ A @ T(-cx, -cy)
    tx = cx - A[0, 0] * cx - A[0, 1] * cy
    ty = cy - A[1, 0] * cx - A[1, 1] * cy
    
    M = np.array([
        [A[0, 0], A[0, 1], tx],
        [A[1, 0], A[1, 1], ty],
        [0,       0,       1]
    ], dtype=np.float32)
    
    # 6. Compute inverse transformation (for backward mapping)
    M_inv = np.linalg.inv(M)
    
    # 7. Vectorized coordinate transformation
    # Create meshgrid of output coordinates
    j_coords = np.arange(W, dtype=np.float32)
    i_coords = np.arange(H, dtype=np.float32)
    jj, ii = np.meshgrid(j_coords, i_coords)
    
    # Pixel centers in continuous space
    xx = jj + 0.5
    yy = ii + 0.5
    
    # Stack coordinates into homogeneous form: [x, y, 1]
    ones = np.ones_like(xx)
    coords = np.stack([xx, yy, ones], axis=-1)  # Shape: (H, W, 3)
    
    # Apply inverse transformation: src_coords = M_inv @ output_coords
    src_coords = np.dot(coords, M_inv.T)  # Shape: (H, W, 3)
    
    src_x = src_coords[..., 0]  # Shape: (H, W)
    src_y = src_coords[..., 1]  # Shape: (H, W)
    
    # 8. Perform vectorized bilinear interpolation
    output = vectorized_bilinear_interpolation(image, src_x, src_y)
    
    return np.clip(output, 0, 255).astype(np.uint8)


def vectorized_bilinear_interpolation(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Vectorized bilinear interpolation on entire image at once.
    
    Args:
        image: Input image (HxWxC)
        x: x-coordinates array (HxW)
        y: y-coordinates array (HxW)
    
    Returns:
        Interpolated image (HxWxC)
    """
    
    H, W, C = image.shape
    
    # Get integer and fractional parts
    x_int = np.floor(x - 0.5).astype(np.int32)
    y_int = np.floor(y - 0.5).astype(np.int32)
    
    dx = x - 0.5 - x_int
    dy = y - 0.5 - y_int
    
    # Clamp to valid range
    x_int = np.clip(x_int, 0, W - 1)
    y_int = np.clip(y_int, 0, H - 1)
    
    x_int_next = np.clip(x_int + 1, 0, W - 1)
    y_int_next = np.clip(y_int + 1, 0, H - 1)
    
    # Get the four nearest pixels
    p00 = image[y_int, x_int].astype(np.float32)  # (H, W, C)
    p10 = image[y_int, x_int_next].astype(np.float32)
    p01 = image[y_int_next, x_int].astype(np.float32)
    p11 = image[y_int_next, x_int_next].astype(np.float32)
    
    # Bilinear interpolation weights
    w00 = (1 - dx[..., np.newaxis]) * (1 - dy[..., np.newaxis])
    w10 = dx[..., np.newaxis] * (1 - dy[..., np.newaxis])
    w01 = (1 - dx[..., np.newaxis]) * dy[..., np.newaxis]
    w11 = dx[..., np.newaxis] * dy[..., np.newaxis]
    
    # Weighted sum
    result = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11
    
    # Handle out-of-bounds: set to black where original coordinates were out of bounds
    out_of_bounds = (x < 0) | (x >= W) | (y < 0) | (y >= H)
    result[out_of_bounds] = 0
    
    return result


def bilinear_interpolation(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """
    Perform bilinear interpolation on image at continuous coordinates (x, y).
    
    Args:
        image: Input image (HxWxC)
        x: x-coordinate (column) in continuous space
        y: y-coordinate (row) in continuous space
    
    Returns:
        Interpolated pixel value (C,) - for all channels
    """
    
    H, W, C = image.shape
    
    # Check if point is within bounds
    if x < 0 or x >= W or y < 0 or y >= H:
        # Return black for out-of-bounds
        return np.zeros(C, dtype=np.float32)
    
    # Get integer and fractional parts
    x_int = int(np.floor(x - 0.5))
    y_int = int(np.floor(y - 0.5))
    
    dx = x - 0.5 - x_int
    dy = y - 0.5 - y_int
    
    # Clamp to valid range
    x_int = np.clip(x_int, 0, W - 1)
    y_int = np.clip(y_int, 0, H - 1)
    
    x_int_next = np.clip(x_int + 1, 0, W - 1)
    y_int_next = np.clip(y_int + 1, 0, H - 1)
    
    # Get the four nearest pixels
    p00 = image[y_int, x_int].astype(np.float32)
    p10 = image[y_int, x_int_next].astype(np.float32)
    p01 = image[y_int_next, x_int].astype(np.float32)
    p11 = image[y_int_next, x_int_next].astype(np.float32)
    
    # Bilinear interpolation formula
    # f(x, y) = (1-dx)(1-dy)*f(0,0) + dx(1-dy)*f(1,0) + (1-dx)dy*f(0,1) + dx*dy*f(1,1)
    
    result = (1 - dx) * (1 - dy) * p00 + \
             dx * (1 - dy) * p10 + \
             (1 - dx) * dy * p01 + \
             dx * dy * p11
    
    return result
