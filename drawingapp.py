import tkinter as tk
from network import load_trained_network
import numpy as np
import math
from PIL import Image, ImageFilter

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition")

        self.rows = 28
        self.cols = 28
        self.cell_size = 10  #each cell is 10×10 px
        self.canvas_width = self.cols * self.cell_size
        self.canvas_height = self.rows * self.cell_size
        self.brush_radius = self.cell_size * 1.5  #15px brush

        #setup canvas and cells
        self.canvas = tk.Canvas(
            root,
            width=self.canvas_width,
            height=self.canvas_height,
            highlightthickness=0
        )

        self.canvas.grid(row=0, column=0, rowspan=3)

        #create cell rectangles
        self.cell_ids = np.zeros((self.rows, self.cols), dtype=int)
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill="white",
                    outline="#ddd"
                )

                self.cell_ids[r, c] = rect

        #Buttons
        self.clear_btn = tk.Button(root, text="Clear", command=self.clear)
        self.clear_btn.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        #Results text
        self.results_text = tk.Text(root, width=20, height=12, font=("Consolas", 12))
        self.results_text.grid(row=1, column=1, rowspan=2, padx=5, pady=5)
        self.results_text.configure(state="disabled")

        #Pixel array (float in [0,1])
        self.pixels = np.zeros((self.rows, self.cols), dtype=np.float32)

        self.drawing = False

        #Bind events
        self.canvas.bind("<Button-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)

        #Load network
        self.net = load_trained_network()

        #Initial display
        self._update_results()

    def _start_draw(self, event):
        self.drawing = True
        self._antialiased_paint(event.x, event.y)

    def _draw(self, event):
        if self.drawing:
            self._antialiased_paint(event.x, event.y)

    def _stop_draw(self, event):
        self.drawing = False

    def _antialiased_paint(self, x, y):
        #Determine affected cell range
        r_min = max(0, int((y - self.brush_radius) // self.cell_size))
        r_max = min(self.rows - 1, int((y + self.brush_radius) // self.cell_size))
        c_min = max(0, int((x - self.brush_radius) // self.cell_size))
        c_max = min(self.cols - 1, int((x + self.brush_radius) // self.cell_size))

        updated = False
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                cx = (c + 0.5) * self.cell_size
                cy = (r + 0.5) * self.cell_size

                dist = math.hypot(cx - x, cy - y)

                if dist <= self.brush_radius:
                    intensity = 1.0 - (dist / self.brush_radius)
                    intensity = max(self.pixels[r, c], intensity)

                    if intensity > self.pixels[r, c]:
                        self.pixels[r, c] = intensity

                        gray = int(255 * (1.0 - intensity))
                        color = f"#{gray:02x}{gray:02x}{gray:02x}"

                        self.canvas.itemconfig(self.cell_ids[r, c], fill=color)
                        updated = True
        if updated:
            self._update_results()

    def _update_results(self):
        #If no drawing at all, show zeros
        if not np.any(self.pixels):
            self.results_text.configure(state="normal")
            self.results_text.delete("1.0", tk.END)

            for d in range(10):
                self.results_text.insert(tk.END, f"{d}:    0.00%\n")
            
            self.results_text.configure(state="disabled")

            return

        #---Preprocess to match MNIST---
        img = (self.pixels * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)

        #Center of mass shift
        coords = np.column_stack(np.nonzero(img))

        if coords.size:
            weights = img[img>0]

            cy = np.average(coords[:,0], weights=weights)
            cx = np.average(coords[:,1], weights=weights)

            shift_y = int(self.rows//2 - cy)
            shift_x = int(self.cols//2 - cx)

            pil_img = pil_img.transform(
                pil_img.size, Image.AFFINE,
                (1, 0, shift_x, 0, 1, shift_y), fillcolor=0
            )
        
        #Gaussian blur
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))

        #Contrast stretch to [0,1]
        np_img = np.array(pil_img).astype(np.float32)
        mn, mx = np_img.min(), np_img.max()

        if mx > mn:
            np_img = (np_img - mn) / (mx - mn)

        #---Feedforward and display probabilities---
        vect = np_img.reshape((784, 1))
        output = self.net.feedforward(vect)
        probs = np.ravel(output)
        ranking = np.argsort(probs)[::-1]

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", tk.END)

        for d in ranking:
            pct = probs[d] * 100
            self.results_text.insert(tk.END, f"{d}: {pct:6.2f}%\n")
        
        self.results_text.configure(state="disabled")

    def clear(self):
        self.pixels.fill(0)

        for r in range(self.rows):
            for c in range(self.cols):
                self.canvas.itemconfig(self.cell_ids[r, c], fill="white")
        
        self._update_results()

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
