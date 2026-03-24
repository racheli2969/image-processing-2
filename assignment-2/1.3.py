"""
Interpolation functions for image processing
"""


def bilinear_interpolation(alpha, beta, I00, I01, I10, I11):
    """
    Bilinear interpolation between four pixel values
    
    Parameters:
    -----------
    alpha : float
        Interpolation parameter in x-axis (columns), 0 ≤ alpha ≤ 1
    beta : float
        Interpolation parameter in y-axis (rows), 0 ≤ beta ≤ 1
    I00 : float
        Pixel value at row 0, column 0
    I01 : float
        Pixel value at row 0, column 1 (change in x-axis direction)
    I10 : float
        Pixel value at row 1, column 0 (change in y-axis direction)
    I11 : float
        Pixel value at row 1, column 1
    
    Returns:
    --------
    float
        The interpolated value at point (alpha, beta)
    
    Formula:
    P(alpha, beta) = (1-alpha)(1-beta)*I00 + alpha(1-beta)*I01 + 
                     (1-alpha)*beta*I10 + alpha*beta*I11
    
    Note:
    The ordering of pixels accounts for the fact that in image[i,j], i is row and j is column,
    while alpha, beta measure in order (x, y) as in standard mathematics.
    """
    # Calculate weighted average
    # Weight for pixel I00 (bottom-left)
    w00 = (1 - alpha) * (1 - beta)
    
    # Weight for pixel I01 (top-left)
    w01 = alpha * (1 - beta)
    
    # Weight for pixel I10 (bottom-right)
    w10 = (1 - alpha) * beta
    
    # Weight for pixel I11 (top-right)
    w11 = alpha * beta
    
    # The interpolated value is the weighted average
    result = w00 * I00 + w01 * I01 + w10 * I10 + w11 * I11
    
    return result


def nearest_neighbor(alpha, beta, I00, I01, I10, I11):
    """
    Nearest neighbor interpolation between four pixel values
    
    Parameters:
    -----------
    alpha : float
        Interpolation parameter in x-axis (columns), 0 ≤ alpha ≤ 1
    beta : float
        Interpolation parameter in y-axis (rows), 0 ≤ beta ≤ 1
    I00 : float
        Pixel value at row 0, column 0
    I01 : float
        Pixel value at row 0, column 1
    I10 : float
        Pixel value at row 1, column 0
    I11 : float
        Pixel value at row 1, column 1
    
    Returns:
    --------
    float
        The value of the nearest pixel to point (alpha, beta)
    
    Method:
    Choose the pixel closest to point (alpha, beta):
    - If alpha < 0.5 choose column 0, otherwise choose column 1
    - If beta < 0.5 choose row 0, otherwise choose row 1
    """
    # Choose column based on alpha
    if alpha < 0.5:
        # Column 0
        if beta < 0.5:
            # Row 0
            return I00
        else:
            # Row 1
            return I10
    else:
        # Column 1
        if beta < 0.5:
            # Row 0
            return I01
        else:
            # Row 1
            return I11


# Example usage
if __name__ == "__main__":
    # Example: Bilinear interpolation
    I00 = 100  # row 0, col 0
    I01 = 150  # row 0, col 1
    I10 = 120  # row 1, col 0
    I11 = 180  # row 1, col 1
    alpha = 0.5  # x-axis
    beta = 0.5   # y-axis
    
    result_bilinear = bilinear_interpolation(alpha, beta, I00, I01, I10, I11)
    print(f"Bilinear interpolation:")
    print(f"I00={I00} (row 0, col 0)")
    print(f"I01={I01} (row 0, col 1)")
    print(f"I10={I10} (row 1, col 0)")
    print(f"I11={I11} (row 1, col 1)")
    print(f"alpha={alpha} (x-axis), beta={beta} (y-axis)")
    print(f"Result: {result_bilinear}\n")
    
    # Example: Nearest neighbor interpolation
    result_nn = nearest_neighbor(alpha, beta, I00, I01, I10, I11)
    print(f"Nearest neighbor interpolation:")
    print(f"Same pixel values")
    print(f"alpha={alpha} (x-axis), beta={beta} (y-axis)")
    print(f"Result: {result_nn}")
