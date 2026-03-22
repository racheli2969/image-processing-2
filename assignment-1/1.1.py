import math

# convert degrees to radians
def degrees_to_radians(degrees):
    """Convert degrees to radians"""
    # The formula to convert degrees to radians is: radians = degrees * (pi / 180)
    return degrees * math.pi / 180

# for each value the degrees will be converted to radians, and the sine and cosine will be calculated
degrees_values = [0, 90, 180, 45, 30, 10, 5, 1]
results = []

for deg in degrees_values:
    rad = degrees_to_radians(deg)
    sin_val = math.sin(rad)
    cos_val = math.cos(rad)
    results.append((deg, rad, sin_val, cos_val))

# print in csv format
print("degrees,radians,sin,cos")
for deg, rad, sin_val, cos_val in results:
    print(f"{deg},{rad},{sin_val},{cos_val}")
