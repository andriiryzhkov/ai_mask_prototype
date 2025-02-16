import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import torch
import os

MODEL_TYPE = "vit_b"
MODEL_CHECKPOINT = "sam_vit_b_01ec64.pth"

class ClickPoint:
    def __init__(self, x, y, label):
        self.x = x  # x coordinate on original image
        self.y = y  # y coordinate on original image
        self.label = label  # 1 for positive, 0 for negative

class SAMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Object Mask Demo")
        self.root.geometry("1200x800")

        # Initialize state variables
        self.image_filename = None
        self.image = None
        self.photo = None
        self.image_scale = 1.0
        self.points = []
        self.computing = False
        self.mask = None
        self.mask_overlay = None
        self.is_encoding = False
        
        # Initialize SAM model
        self.setup_sam_model()
        
        # Create GUI elements
        self.create_gui()

    def setup_sam_model(self):
        # You'll need to download the model file from Meta's repository
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sam_checkpoint = os.path.join(parent_dir, "weights", MODEL_CHECKPOINT)
        
        try:
            sam = sam_model_registry[MODEL_TYPE](checkpoint=sam_checkpoint)
            self.predictor = SamPredictor(sam)
            if torch.cuda.is_available():
                self.predictor.model.to(device="cuda")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load SAM model: {str(e)}")
            raise

    def create_gui(self):
        # Button frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.load_button = ttk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = ttk.Button(button_frame, text="Clear Prompts", command=self.clear_prompts)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(button_frame, text="Save Mask", command=self.save_mask)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Canvas for drawing
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(self.canvas_frame, bg='grey80')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.canvas.bind("<Button-1>", lambda e: self.on_click(e, 1))  # Left click
        self.canvas.bind("<Button-3>", lambda e: self.on_click(e, 0))  # Right click

        # Loading indicator
        self.loading_label = ttk.Label(self.canvas_frame, text="Processing...", background='gray80')
        self.loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.loading_label.place_forget()

    def load_image(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")])
        
        if not filename:
            return

        try:
            # Load and display image
            self.image_filename = filename
            self.image = Image.open(filename)
            self.update_canvas()
            
            # Compute embeddings
            self.show_encoding_state(True)
            self.root.after(100, self.compute_image_embedding)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def update_canvas(self):
        if not self.image:
            return

        # Calculate scaling to fit image while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width, image_height = self.image.size

        scale_x = canvas_width / image_width
        scale_y = canvas_height / image_height
        self.image_scale = min(scale_x, scale_y)

        # Resize image for display
        new_size = (int(image_width * self.image_scale), int(image_height * self.image_scale))
        self.photo = ImageTk.PhotoImage(self.image.resize(new_size, Image.Resampling.LANCZOS))

        # Center the image
        x_offset = (canvas_width - new_size[0]) // 2
        y_offset = (canvas_height - new_size[1]) // 2

        # Clear canvas and draw image
        self.canvas.delete("all")
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo)

        # Draw points
        self.draw_points()

        # Draw mask overlay
        if self.mask is not None:
            self.draw_mask_overlay()

    def compute_image_embedding(self):
        try:
            # Convert image to numpy array
            img_array = np.array(self.image)
            
            # Compute embeddings
            self.predictor.set_image(img_array)
            
            self.show_encoding_state(False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute image embedding: {str(e)}")
            self.show_encoding_state(False)

    def show_encoding_state(self, encoding):
        self.is_encoding = encoding
        if encoding:
            self.loading_label.configure(text="Encoding image...")
            self.loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            self.canvas.config(state=tk.DISABLED)
        else:
            self.loading_label.place_forget()
            self.canvas.config(state=tk.NORMAL)

    def on_click(self, event, label):
        if not self.image or self.is_encoding:
            return

        # Convert canvas coordinates to image coordinates
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width, image_height = self.image.size
        scaled_width = int(image_width * self.image_scale)
        scaled_height = int(image_height * self.image_scale)
        
        x_offset = (canvas_width - scaled_width) // 2
        y_offset = (canvas_height - scaled_height) // 2

        image_x = (event.x - x_offset) / self.image_scale
        image_y = (event.y - y_offset) / self.image_scale

        # Check if click is within image bounds
        if (0 <= image_x < image_width and 0 <= image_y < image_height):
            point = ClickPoint(image_x, image_y, label)
            self.points.append(point)
            self.compute_mask()
            self.update_canvas()

    def compute_mask(self):
        if not self.points:
            return

        try:
            # Prepare input points
            input_points = np.array([[p.x, p.y] for p in self.points])
            input_labels = np.array([p.label for p in self.points])

            # Show loading indicator
            self.loading_label.configure(text="Computing mask...")
            self.loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            self.root.update()

            # Compute mask
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )

            # Take the mask with highest score
            mask_idx = np.argmax(scores)
            self.mask = masks[mask_idx]

            # Hide loading indicator
            self.loading_label.place_forget()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute mask: {str(e)}")
            self.loading_label.place_forget()

    def draw_points(self):
        for point in self.points:
            x = point.x * self.image_scale + (self.canvas.winfo_width() - self.image.size[0] * self.image_scale) // 2
            y = point.y * self.image_scale + (self.canvas.winfo_height() - self.image.size[1] * self.image_scale) // 2
            
            color = "green" if point.label == 1 else "red"
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=color, outline=color)

    def draw_mask_overlay(self):
        if self.mask is None:
            return

        # Create overlay image
        overlay = Image.new('RGBA', self.image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Convert mask to RGBA
        mask_rgba = np.zeros((*self.mask.shape, 4), dtype=np.uint8)
        mask_rgba[self.mask] = [135, 206, 235, 128]  # Light blue with 50% opacity

        # Apply mask to overlay
        overlay.paste(Image.fromarray(mask_rgba), (0, 0))

        # Resize overlay
        new_size = (int(self.image.size[0] * self.image_scale), int(self.image.size[1] * self.image_scale))
        overlay = overlay.resize(new_size, Image.Resampling.LANCZOS)
        self.mask_overlay = ImageTk.PhotoImage(overlay)

        # Draw overlay
        x_offset = (self.canvas.winfo_width() - new_size[0]) // 2
        y_offset = (self.canvas.winfo_height() - new_size[1]) // 2
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.mask_overlay)

    def clear_prompts(self):
        self.points = []
        self.mask = None
        self.mask_overlay = None
        self.update_canvas()

    def save_mask(self):
        if self.mask is None:
            messagebox.showerror("Error", "No mask available to save!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=f"mask_{os.path.basename(self.image_filename)}.png" if self.image_filename else None
        )

        if filename:
            try:
                # Save the binary mask
                mask_image = Image.fromarray(self.mask.astype(np.uint8) * 255)
                mask_image.save(filename)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save mask: {str(e)}")

def main():
    root = tk.Tk()
    app = SAMApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()