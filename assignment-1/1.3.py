import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define angle in radians
angle = 30 * np.pi / 180  # 30 degrees to radians

# rotation matrix for 30 degrees
r_30 = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
])
print("Rotation matrix r_30 (30 degrees):")
print(r_30)
print()

# scale matrix for stretching by 2 in x-direction
sx_2 = np.array([
    [2, 0],
    [0, 1]
])
print("Scale matrix sx_2 (stretch by 2 in x-direction):")
print(sx_2)
print()

# calculate the product of r_30 and sx_2
rs = r_30 @ sx_2
print("rs = r_30 @ sx_2:")
print(rs)
print()

# calculate the product of sx_2 and r_30
sr = sx_2 @ r_30
print("sr = sx_2 @ r_30:")
print(sr)
print()


# Rectangle: width=2, height=1, centered at origin
# Corners: (-1, -0.5), (1, -0.5), (1, 0.5), (-1, 0.5)
original_rect = np.array([
    [-1, -0.5],
    [1, -0.5],
    [1, 0.5],
    [-1, 0.5]
])

# Helper function to apply transformation matrix to rectangle corners
def apply_transformation(rect, matrix):
    """Apply transformation matrix to rectangle corners"""
    return np.array([matrix @ point for point in rect])

# rotate the rectangle by 30 degrees using r_30
rotated_rect = apply_transformation(original_rect, r_30)

#stretch the rectangle by 2 in x-direction using sx_2
scaled_rect = apply_transformation(original_rect, sx_2)

# stretch then rotate (sr) and rotate then stretch (rs)
sr_rect = apply_transformation(original_rect, sr)
rs_rect = apply_transformation(original_rect, rs)

# show the resulted rectangles in a 2x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Rectangle Transformations', fontsize=16)

rectangles = [
    (original_rect, "Original", axes[0, 0]),
    (rotated_rect, "Rotated (30°)", axes[0, 1]),
    (scaled_rect, "Scaled (2x in x)", axes[0, 2]),
    (sr_rect, "SR (Scale then Rotate)", axes[1, 0]),
    (rs_rect, "RS (Rotate then Scale)", axes[1, 1])
]

for rect, title, ax in rectangles:
    # Close the polygon by adding the first point at the end
    rect_closed = np.vstack([rect, rect[0]])
    ax.plot(rect_closed[:, 0], rect_closed[:, 1], 'b-', linewidth=2)
    ax.fill(rect_closed[:, 0], rect_closed[:, 1], alpha=0.3)
    
    # Plot points
    ax.plot(rect[:, 0], rect[:, 1], 'ro', markersize=8)
    
    # Set equal aspect ratio and add grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

# Hide the extra subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
