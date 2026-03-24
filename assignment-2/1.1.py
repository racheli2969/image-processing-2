	
import numpy as np

def translation_matrix(a, b):
    """
    Create a 3x3 translation matrix in homogeneous coordinates.
    Translates a point (x, y) by (a, b) to (x+a, y+b).
    
    Args:
        a: Translation in x-direction
        b: Translation in y-direction
    
    Returns:
        3x3 translation matrix
    """
    matrix = np.array([
        [1, 0, a],
        [0, 1, b],
        [0, 0, 1]
    ], dtype=float)
    return matrix


def rotation_matrix(theta):
    """
    Create a 3x3 rotation matrix in homogeneous coordinates.
    Rotates a point by theta degrees around the origin.
    
    Args:
        theta: Rotation angle in degrees
    
    Returns:
        3x3 rotation matrix
    """
    theta_rad = np.radians(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=float)
    return matrix


def scale_matrix(sx, sy=None):
    """
    Create a 3x3 scale matrix in homogeneous coordinates.
    Scales a point by (sx, sy).
    
    Args:
        sx: Scale factor in x-direction
        sy: Scale factor in y-direction. If None, sy = sx (uniform scale)
    
    Returns:
        3x3 scale matrix
    """
    if sy is None:
        sy = sx
    
    matrix = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ], dtype=float)
    return matrix


def rotation_around_point(center_x, center_y, theta):
    """
    Create a 3x3 matrix that rotates theta degrees around a point.
    Uses composition of operations:
    1. Translate so the point is at origin
    2. Rotate
    3. Translate back
    
    Args:
        center_x: X coordinate of rotation center
        center_y: Y coordinate of rotation center
        theta: Rotation angle in degrees
    
    Returns:
        3x3 rotation matrix around the specified point
    """
    # Translate to origin
    T1 = translation_matrix(-center_x, -center_y)
    
    # Rotate
    R = rotation_matrix(theta)
    
    # Translate back
    T2 = translation_matrix(center_x, center_y)
    
    # Compose: T2 * R * T1 (apply in reverse order for matrix multiplication)
    result = T2 @ R @ T1
    return result


# Test: Rotation of 30 degrees around point (100, 200)
rotation_30_around_100_200 = rotation_around_point(100, 200, 30)

if __name__ == "__main__":
    print("Translation matrix T(5, 3):")
    print(translation_matrix(5, 3))
    print("\nRotation matrix R(45°):")
    print(rotation_matrix(45))
    print("\nScale matrix S(2, 1.5):")
    print(scale_matrix(2, 1.5))
    print("\nUniform scale S(2):")
    print(scale_matrix(2))
    print("\nRotation 30° around point (100, 200):")
    print(rotation_30_around_100_200)
