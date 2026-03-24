import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def get_rectangle_corners():
    """
    Create rectangle with height=1, width=2, centered at origin
    Returns corners in homogeneous coordinates
    """
    corners = np.array([
        [-1, -0.5, 1],  # bottom-left
        [1, -0.5, 1],   # bottom-right
        [1, 0.5, 1],    # top-right
        [-1, 0.5, 1]    # top-left
    ])
    return corners

def rotation_matrix(angle_degrees):
    """
    Create rotation matrix for counterclockwise rotation
    angle_degrees: rotation angle in degrees
    Returns 3x3 homogeneous transformation matrix
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    return matrix

def scaling_matrix(sx, sy):
    """
    Create scaling matrix
    sx, sy: scaling factors for x and y axes
    Returns 3x3 homogeneous transformation matrix
    """
    matrix = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])
    return matrix

def transform_corners(corners, transformation_matrix):
    """
    Apply transformation matrix to corners
    corners: Nx3 array of homogeneous coordinates
    transformation_matrix: 3x3 transformation matrix
    Returns Nx3 array of transformed corners
    """
    transformed = np.dot(corners, transformation_matrix.T)
    return transformed

def homogeneous_to_cartesian(homogeneous_coords):
    """
    Convert homogeneous coordinates back to Cartesian coordinates
    homogeneous_coords: Nx3 array
    Returns Nx2 array
    """
    return homogeneous_coords[:, :2]

# Create the original rectangle corners
original_corners = get_rectangle_corners()

# a. Original rectangle (identity transformation)
rect_a = original_corners

# b. Rotate by 30 degrees
rot_30 = rotation_matrix(30)
rect_b = transform_corners(original_corners, rot_30)

# c. Rotate by 45 degrees, then stretch by 2x on x-axis
rot_45 = rotation_matrix(45)
scale_2x = scaling_matrix(2, 1)
rect_c = transform_corners(original_corners, np.dot(rot_45, scale_2x))

# d. Stretch on x-axis, then rotate by 45 degrees
rect_d = transform_corners(original_corners, np.dot(scale_2x, rot_45))

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Homogeneous Coordinate Transformations', fontsize=14, fontweight='bold')

# Helper function to plot rectangle
def plot_rectangle(ax, corners, title):
    # Convert from homogeneous to Cartesian coordinates
    cart_coords = homogeneous_to_cartesian(corners)
    
    # Create polygon and plot
    polygon = Polygon(cart_coords, fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(polygon)
    
    # Plot corners as points
    ax.plot(cart_coords[:, 0], cart_coords[:, 1], 'ro', markersize=6)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight='bold')

# Plot all four rectangles
plot_rectangle(axes[0, 0], rect_a, 'a) Original Rectangle\n(height=1, width=2)')
plot_rectangle(axes[0, 1], rect_b, 'b) Rotated 30°')
plot_rectangle(axes[1, 0], rect_c, 'c) Rotated 45°\nthen Scaled 2x on x-axis')
plot_rectangle(axes[1, 1], rect_d, 'd) Scaled 2x on x-axis\nthen Rotated 45°')

plt.tight_layout()
plt.show()
