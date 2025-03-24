import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

# Image history
history = []
redo_history = []
img = None
img_result = None


def save_to_history(image):
    if image is not None:
        history.append(image.copy())
        redo_history.clear()


def undo():
    if len(history) > 1:
        redo_history.append(history.pop())
        display_image(history[-1])


def redo():
    if redo_history:
        history.append(redo_history.pop())
        display_image(history[-1])


def load_image():
    global img
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif")])
    if file_path:
        try:
            pil_image = Image.open(file_path).convert("L")  # Convert to grayscale
            img = np.array(pil_image)
            save_to_history(img)
            display_image(img)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")


def save_image():
    if img_result is None:
        messagebox.showerror("Error", "No image to save")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if file_path:
        cv2.imwrite(file_path, img_result)


def display_image(image):
    global img_result
    img_result = image
    img = Image.fromarray(image)
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img


def otsu_threshold(image):
    """Calculate optimal threshold using Otsu's method"""
    # Calculate histogram
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    hist = hist.astype(float)

    # Calculate probabilities
    prob = hist / hist.sum()

    # Initialize variables
    max_var = 0
    optimal_thresh = 0

    for t in range(1, 256):
        # Class probabilities
        w0 = prob[:t].sum()
        w1 = prob[t:].sum()

        if w0 == 0 or w1 == 0:
            continue

        # Class means
        mu0 = (np.arange(t) * prob[:t]).sum() / w0
        mu1 = (np.arange(t, 256) * prob[t:]).sum() / w1

        # Between-class variance
        var = w0 * w1 * (mu0 - mu1) ** 2

        if var > max_var:
            max_var = var
            optimal_thresh = t

    return optimal_thresh


def recursive_otsu(image, levels, current_level=1):
    """Recursive implementation of multi-level Otsu thresholding"""
    if current_level >= levels or len(np.unique(image)) < 2:
        return []

    threshold = otsu_threshold(image)

    lower_part = image[image < threshold]
    upper_part = image[image >= threshold]

    lower_thresholds = recursive_otsu(lower_part, levels, current_level + 1)
    upper_thresholds = recursive_otsu(upper_part, levels, current_level + 1)

    return lower_thresholds + [threshold] + upper_thresholds


def sauvola_threshold(image, window_size=15, k=0.2, r=128):
    """Implement Sauvola's thresholding from scratch"""
    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd

    pad = window_size // 2
    padded = np.pad(image, pad, mode='reflect')
    threshold_map = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i + window_size, j:j + window_size]
            mean = np.mean(window)
            std = np.std(window)

            # Sauvola's formula
            threshold_map[i, j] = mean * (1 + k * (std / r - 1))

    binary = (image > threshold_map).astype(np.uint8) * 255
    return binary


def apply_recursive_otsu():
    global img
    if img is None:
        messagebox.showerror("Error", "No image loaded")
        return
    try:
        num_classes = int(otsu_levels_var.get())
        if num_classes < 2 or num_classes > 4:
            raise ValueError("Number of classes must be between 2 and 4")

        # Get thresholds using our recursive implementation
        thresholds = sorted(recursive_otsu(img, num_classes))

        # If we didn't get enough thresholds, use standard Otsu
        while len(thresholds) < num_classes - 1:
            thresholds.append(otsu_threshold(img))
            thresholds = sorted(thresholds)

        # Segment the image
        segmented = np.digitize(img, bins=thresholds) * (255 // num_classes)
        segmented = segmented.astype(np.uint8)

        save_to_history(segmented)
        display_image(segmented)
    except ValueError as e:
        messagebox.showerror("Error", str(e))


def apply_sauvola():
    global img
    if img is None:
        messagebox.showerror("Error", "No image loaded")
        return
    try:
        window_size = int(window_size_var.get())
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("Window size must be an odd number ≥ 3")

        # Get parameters from GUI
        k_value = float(k_value_var.get())
        r_value = float(r_value_var.get())

        # Apply our Sauvola implementation
        binary_sauvola = sauvola_threshold(img, window_size=window_size, k=k_value, r=r_value)

        save_to_history(binary_sauvola)
        display_image(binary_sauvola)
    except ValueError as e:
        messagebox.showerror("Error", str(e))


def toggle_parameter_input(event):
    if method_var.get() == "Sauvola":
        window_size_label.grid(row=0, column=2, padx=5)
        window_size_entry.grid(row=0, column=3, padx=5)
        k_value_label.grid(row=1, column=0, padx=5)
        k_value_entry.grid(row=1, column=1, padx=5)
        r_value_label.grid(row=1, column=2, padx=5)
        r_value_entry.grid(row=1, column=3, padx=5)
        otsu_levels_label.grid_remove()
        otsu_levels_entry.grid_remove()
    else:
        window_size_label.grid_remove()
        window_size_entry.grid_remove()
        k_value_label.grid_remove()
        k_value_entry.grid_remove()
        r_value_label.grid_remove()
        r_value_entry.grid_remove()
        otsu_levels_label.grid(row=0, column=2, padx=5)
        otsu_levels_entry.grid(row=0, column=3, padx=5)


def validate_otsu_input(new_value):
    if new_value == "":
        return True
    if new_value.isdigit():
        num = int(new_value)
        return 2 <= num <= 4
    return False


def validate_window_size(new_value):
    if new_value == "":
        return True
    if new_value.isdigit():
        num = int(new_value)
        return num >= 3 and num % 2 == 1
    return False


def step_process():
    global img_result, img
    image_to_analyze = img_result if img_result is not None else img
    if image_to_analyze is None:
        messagebox.showerror("Error", "No image loaded")
        return
    plt.figure()
    plt.hist(image_to_analyze.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show(block=False)


root = tk.Tk()
root.title("Image Thresholding")
root.geometry("400x600")

frame_top = tk.Frame(root)
frame_top.pack(pady=10)

panel = tk.Label(frame_top)
panel.pack()

frame_controls = tk.Frame(root)
frame_controls.pack(pady=5)

tk.Button(frame_controls, text="Load Image", command=load_image).grid(row=0, column=0, padx=5, pady=5)
tk.Button(frame_controls, text="Save Image", command=save_image).grid(row=0, column=1, padx=5, pady=5)

frame_method = tk.Frame(root)
frame_method.pack(pady=5)

tk.Label(frame_method, text="Thresholding Method:").grid(row=0, column=0, padx=5)
method_var = tk.StringVar(value="Sauvola")
method_menu = ttk.Combobox(frame_method, textvariable=method_var, values=["Sauvola", "Recursive Otsu"])
method_menu.grid(row=0, column=1, padx=5)
method_menu.bind("<<ComboboxSelected>>", toggle_parameter_input)

# Sauvola parameters
window_size_label = tk.Label(frame_method, text="Window Size:")
window_size_var = tk.StringVar(value="15")
window_size_entry = tk.Entry(frame_method, textvariable=window_size_var, width=5)
wcmd = root.register(validate_window_size)
window_size_entry.configure(validate="key", validatecommand=(wcmd, "%P"))

k_value_label = tk.Label(frame_method, text="k (Sensitivity to Local Contrast):")
k_value_var = tk.StringVar(value="0.2")
k_value_entry = tk.Entry(frame_method, textvariable=k_value_var, width=5)

r_value_label = tk.Label(frame_method, text="R (Dynamic Range Normalization):")
r_value_var = tk.StringVar(value="128")
r_value_entry = tk.Entry(frame_method, textvariable=r_value_var, width=5)

# Otsu parameters
otsu_levels_label = tk.Label(frame_method, text="Otsu Levels:")
otsu_levels_var = tk.StringVar(value="3")
otsu_levels_entry = tk.Entry(frame_method, textvariable=otsu_levels_var, width=5)
vcmd = root.register(validate_otsu_input)
otsu_levels_entry.configure(validate="key", validatecommand=(vcmd, "%P"))

tk.Button(root, text="Apply Threshold",
          command=lambda: apply_sauvola() if method_var.get() == "Sauvola" else apply_recursive_otsu()).pack(pady=5)
tk.Button(root, text="Step Process (Histogram)", command=step_process).pack(pady=5)

frame_undo_redo = tk.Frame(root)
frame_undo_redo.pack(pady=5)

tk.Button(frame_undo_redo, text="Undo", command=undo).grid(row=0, column=0, padx=5)
tk.Button(frame_undo_redo, text="Redo", command=redo).grid(row=0, column=1, padx=5)

toggle_parameter_input(None)

root.mainloop()