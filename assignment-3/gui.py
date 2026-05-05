import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import cv2
import numpy as np
from tkinter import filedialog, Tk

import warp

class ImageApp:
    def __init__(self):
        self.original_image = None      # Original BGR image (cv2)
        self.current_image = None       # Image after transformation
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(12, 8))
        self.ax_img = plt.subplot(111)
        
        # Title
        self.fig.suptitle('Image Rotation and Scaling', fontsize=16)
        
        # Adjust layout to make room for sliders
        plt.subplots_adjust(bottom=0.35)
        
        # Create sliders
        self.ax_rot = plt.axes([0.2, 0.25, 0.6, 0.03])
        self.ax_sx = plt.axes([0.2, 0.20, 0.6, 0.03])
        self.ax_sy = plt.axes([0.2, 0.15, 0.6, 0.03])
        
        self.slider_rot = Slider(self.ax_rot, 'Rotation (°)', -180, 180, valinit=0, valstep=1)
        self.slider_sx = Slider(self.ax_sx, 'Scale X', 0.1, 3.0, valinit=1.0, valstep=0.01)
        self.slider_sy = Slider(self.ax_sy, 'Scale Y', 0.1, 3.0, valinit=1.0, valstep=0.01)
        
        # Connect slider events
        self.slider_rot.on_changed(self.on_slider_change)
        self.slider_sx.on_changed(self.on_slider_change)
        self.slider_sy.on_changed(self.on_slider_change)
        
        # Create buttons
        ax_load = plt.axes([0.2, 0.08, 0.1, 0.04])
        ax_reset = plt.axes([0.4, 0.08, 0.1, 0.04])
        
        self.btn_load = Button(ax_load, 'Load Image')
        self.btn_reset = Button(ax_reset, 'Reset')
        
        self.btn_load.on_clicked(lambda event: self.load_image())
        self.btn_reset.on_clicked(lambda event: self.reset_sliders())
        
    def load_image(self):
        # Use tkinter file dialog
        root = Tk()
        root.withdraw()  # Hide the root window
        
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )
        root.destroy()
        
        if not file_path:
            return
        
        img = cv2.imread(file_path)
        if img is None:
            return
        
        # Resize image if too large (for performance)
        h, w = img.shape[:2]
        if w > 800 or h > 600:
            scale = min(800 / w, 600 / h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        self.original_image = img
        self.current_image = img.copy()
        
        self.display_image()
    
    def reset_sliders(self):
        self.slider_rot.set(0)
        self.slider_sx.set(1.0)
        self.slider_sy.set(1.0)
    
    def on_slider_change(self, val):
        if self.original_image is None:
            return
        
        angle = self.slider_rot.val
        sx = self.slider_sx.val
        sy = self.slider_sy.val
        
        # Perform transformation
        self.current_image = warp.warp_image(self.original_image, angle, sx, sy)
        
        self.display_image()
    
    def display_image(self):
        # Convert BGR -> RGB
        img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Clear previous image
        self.ax_img.clear()
        
        # Display image
        self.ax_img.imshow(img_rgb)
        self.ax_img.set_title('Transformed Image')
        self.ax_img.axis('off')
        
        # Redraw
        self.fig.canvas.draw_idle()
    
    def show(self):
        plt.show()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    app = ImageApp()
    app.show()
