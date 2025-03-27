import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib
from skimage.morphology import remove_small_objects

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
            pil_image = Image.open(file_path).convert("L")
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

    # Calculate maximum display size (80% of screen height)
    screen_height = root.winfo_screenheight()
    max_height = int(screen_height * 0.8)

    # Resize if image is too large
    if img.height > max_height:
        ratio = max_height / float(img.height)
        new_width = int(float(img.width) * ratio)
        img = img.resize((new_width, max_height), Image.LANCZOS)

    img = ImageTk.PhotoImage(img)
    canvas.config(scrollregion=(0, 0, img.width(), img.height()))
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.image = img


def otsu_threshold(image):
    """Calculate optimal threshold using Otsu's method"""
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    hist = hist.astype(float)
    prob = hist / hist.sum()

    max_var = 0
    optimal_thresh = 0

    for t in range(1, 256):
        w0 = prob[:t].sum()
        w1 = prob[t:].sum()

        if w0 == 0 or w1 == 0:
            continue

        mu0 = (np.arange(t) * prob[:t]).sum() / w0
        mu1 = (np.arange(t, 256) * prob[t:]).sum() / w1

        var = w0 * w1 * (mu0 - mu1) ** 2

        if var > max_var:
            max_var = var
            optimal_thresh = t

    return optimal_thresh


def recursive_otsu(image, levels=3, d1=2, d2=26):
    """Improved recursive Otsu with stopping criteria from PDF"""
    thresholds = []
    prev_thresh = 0

    for _ in range(levels):
        if len(thresholds) == 0:
            current_image = image
        else:
            current_image = image[image >= prev_thresh]
            if len(current_image) == 0:
                break

        thresh = otsu_threshold(current_image)

        # Apply stopping criteria from PDF
        if len(thresholds) > 0:
            delta = thresh - prev_thresh
            if delta < d1 or delta > d2:
                break

        thresholds.append(thresh)
        prev_thresh = thresh

    return sorted(thresholds)


def sauvola_threshold(image, window_size=15, k=0.2, r=128):
    """Implement Sauvola's thresholding from scratch"""
    if window_size % 2 == 0:
        window_size += 1

    pad = window_size // 2
    padded = np.pad(image, pad, mode='reflect')
    threshold_map = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i + window_size, j:j + window_size]
            mean = np.mean(window)
            std = np.std(window)
            threshold_map[i, j] = mean * (1 + k * (std / r - 1))

    binary = (image > threshold_map).astype(np.uint8) * 255
    return binary


def estimate_background(image, window_size=21):
    """Background estimation using median filter"""
    return cv2.medianBlur(image, window_size)


def remove_background(image, background):
    """Background removal through subtraction"""
    return np.clip(background - image, 0, 255).astype(np.uint8)


def bilateral_filter(image, sigma_color=2, sigma_space=10):
    """Noise reduction using bilateral filter"""
    return cv2.bilateralFilter(image, -1, sigma_color, sigma_space)


def selective_bilateral_filter(image, mask, fg_params=(2, 2), bg_params=(10, 3)):
    """Selective filtering based on initial segmentation"""
    foreground = cv2.bilateralFilter(image, -1, fg_params[0], fg_params[1])
    background = cv2.bilateralFilter(image, -1, bg_params[0], bg_params[1])
    return np.where(mask > 0, foreground, background)


def despeckle(binary_image, min_size=5):
    """Remove small connected components"""
    return remove_small_objects(binary_image.astype(bool), min_size=min_size).astype(np.uint8) * 255


def apply_recursive_otsu_advanced():
    """Full pipeline from the PDF paper"""
    global img
    if img is None:
        messagebox.showerror("Error", "No image loaded")
        return

    try:
        # 1. Background estimation
        background = estimate_background(img)

        # 2. Background removal
        no_bg = remove_background(img, background)

        # 3. Initial bilateral filtering
        filtered = bilateral_filter(no_bg)

        # 4. Recursive Otsu thresholding
        thresholds = recursive_otsu(filtered)
        if not thresholds:
            thresholds = [otsu_threshold(filtered)]

        # Create initial binary mask
        binary = np.zeros_like(filtered)
        for thresh in thresholds:
            binary = np.maximum(binary, (filtered < thresh).astype(np.uint8) * 255)

        # 5. Selective bilateral filtering
        selective_filtered = selective_bilateral_filter(filtered, binary)

        # 6. Final recursive Otsu with hysteresis
        final_thresholds = recursive_otsu(selective_filtered)
        final_binary = np.zeros_like(selective_filtered)
        for thresh in final_thresholds:
            final_binary = np.maximum(final_binary, (selective_filtered < thresh).astype(np.uint8) * 255)

        # 7. Despeckling
        final_result = despeckle(final_binary)

        save_to_history(final_result)
        display_image(final_result)

    except Exception as e:
        messagebox.showerror("Error", str(e))


def apply_recursive_otsu_simple():
    """Original simple recursive Otsu implementation"""
    global img
    if img is None:
        messagebox.showerror("Error", "No image loaded")
        return
    try:
        num_classes = int(otsu_levels_var.get())
        thresholds = sorted(recursive_otsu(img, num_classes))

        while len(thresholds) < num_classes - 1:
            thresholds.append(otsu_threshold(img))
            thresholds = sorted(thresholds)

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

        k_value = float(k_value_var.get())
        r_value = float(r_value_var.get())

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
    elif method_var.get() == "Advanced Recursive Otsu":
        window_size_label.grid_remove()
        window_size_entry.grid_remove()
        k_value_label.grid_remove()
        k_value_entry.grid_remove()
        r_value_label.grid_remove()
        r_value_entry.grid_remove()
        otsu_levels_label.grid_remove()
        otsu_levels_entry.grid_remove()
    else:  # Simple Recursive Otsu
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
        return num > 1
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


# Main window setup
root = tk.Tk()
root.title("Advanced Document Binarization")
root.geometry("1000x800")

# Create main frames
frame_top = tk.Frame(root)
frame_top.pack(fill=tk.BOTH, expand=True)

# Create canvas with scrollbars
canvas_frame = tk.Frame(frame_top)
canvas_frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(canvas_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add scrollbars
v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
h_scroll = tk.Scrollbar(frame_top, orient=tk.HORIZONTAL, command=canvas.xview)
h_scroll.pack(fill=tk.X)

canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

# Control buttons frame
frame_controls = tk.Frame(root)
frame_controls.pack(fill=tk.X, pady=5)

tk.Button(frame_controls, text="Load Image", command=load_image).grid(row=0, column=0, padx=5, pady=5)
tk.Button(frame_controls, text="Save Image", command=save_image).grid(row=0, column=1, padx=5, pady=5)

# Method selection frame
frame_method = tk.Frame(root)
frame_method.pack(fill=tk.X, pady=5)

tk.Label(frame_method, text="Thresholding Method:").grid(row=0, column=0, padx=5)
method_var = tk.StringVar(value="Sauvola")
method_menu = ttk.Combobox(frame_method, textvariable=method_var,
                           values=["Sauvola", "Simple Recursive Otsu", "Advanced Recursive Otsu"])
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

# Action buttons frame
frame_actions = tk.Frame(root)
frame_actions.pack(fill=tk.X, pady=5)


def apply_threshold():
    method = method_var.get()
    if method == "Sauvola":
        apply_sauvola()
    elif method == "Simple Recursive Otsu":
        apply_recursive_otsu_simple()
    elif method == "Advanced Recursive Otsu":
        apply_recursive_otsu_advanced()


tk.Button(frame_actions, text="Apply Threshold", command=apply_threshold).pack(pady=5)
tk.Button(frame_actions, text="Step Process (Histogram)", command=step_process).pack(pady=5)

# Undo/Redo frame
frame_undo_redo = tk.Frame(root)
frame_undo_redo.pack(fill=tk.X, pady=5)

tk.Button(frame_undo_redo, text="Undo", command=undo).grid(row=0, column=0, padx=5)
tk.Button(frame_undo_redo, text="Redo", command=redo).grid(row=0, column=1, padx=5)

# Initialize parameter inputs
toggle_parameter_input(None)

root.mainloop()