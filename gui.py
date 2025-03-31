import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib
from thresholding import (
    sauvola_threshold,
    apply_recursive_otsu_advanced,
    apply_recursive_otsu_simple,
    invert_image
)

matplotlib.use("TkAgg")

# Globální proměnné
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
            # Načtení a převedení na odstíny šedi
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
    img_disp = Image.fromarray(image)

    # Přizpůsobení maximální výšce (80 % výšky obrazovky)
    screen_height = root.winfo_screenheight()
    max_height = int(screen_height * 0.8)
    if img_disp.height > max_height:
        ratio = max_height / float(img_disp.height)
        new_width = int(float(img_disp.width) * ratio)
        img_disp = img_disp.resize((new_width, max_height), Image.LANCZOS)

    img_disp = ImageTk.PhotoImage(img_disp)
    canvas.config(scrollregion=(0, 0, img_disp.width(), img_disp.height()))
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=img_disp)
    canvas.image = img_disp

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

# Funkce pro aplikaci prahování dle zvolené metody
def apply_threshold():
    global img
    if img is None:
        messagebox.showerror("Error", "No image loaded")
        return
    method = method_var.get()
    try:
        if method == "Sauvola":
            window_size = int(window_size_var.get())
            k_value = float(k_value_var.get())
            r_value = float(r_value_var.get())
            result = sauvola_threshold(img, window_size=window_size, k=k_value, r=r_value)
        elif method == "Simple Recursive Otsu":
            num_classes = int(otsu_levels_var.get())
            result = apply_recursive_otsu_simple(img, num_classes=num_classes)
            # Invertovat lze případně podle potřeby
            result = invert_image(result)
        elif method == "Advanced Recursive Otsu":
            result = apply_recursive_otsu_advanced(img)
        else:
            messagebox.showerror("Error", "Neznámá metoda")
            return

        save_to_history(result)
        display_image(result)
    except Exception as e:
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
        return int(new_value) > 1
    return False

def validate_window_size(new_value):
    if new_value == "":
        return True
    if new_value.isdigit():
        num = int(new_value)
        return num >= 3 and num % 2 == 1
    return False

# Hlavní okno GUI
root = tk.Tk()
root.title("Advanced Document Binarization")
root.geometry("1000x800")

# Rámce pro rozložení
frame_top = tk.Frame(root)
frame_top.pack(fill=tk.BOTH, expand=True)

canvas_frame = tk.Frame(frame_top)
canvas_frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(canvas_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
h_scroll = tk.Scrollbar(frame_top, orient=tk.HORIZONTAL, command=canvas.xview)
h_scroll.pack(fill=tk.X)
canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

frame_controls = tk.Frame(root)
frame_controls.pack(fill=tk.X, pady=5)

tk.Button(frame_controls, text="Load Image", command=load_image).grid(row=0, column=0, padx=5, pady=5)
tk.Button(frame_controls, text="Save Image", command=save_image).grid(row=0, column=1, padx=5, pady=5)

frame_method = tk.Frame(root)
frame_method.pack(fill=tk.X, pady=5)

tk.Label(frame_method, text="Thresholding Method:").grid(row=0, column=0, padx=5)
method_var = tk.StringVar(value="Sauvola")
method_menu = ttk.Combobox(frame_method, textvariable=method_var,
                           values=["Sauvola", "Simple Recursive Otsu", "Advanced Recursive Otsu"])
method_menu.grid(row=0, column=1, padx=5)
method_menu.bind("<<ComboboxSelected>>", toggle_parameter_input)

# Parametry pro Sauvola
window_size_label = tk.Label(frame_method, text="Window Size:")
window_size_var = tk.StringVar(value="15")
window_size_entry = tk.Entry(frame_method, textvariable=window_size_var, width=5)
wcmd = root.register(validate_window_size)
window_size_entry.configure(validate="key", validatecommand=(wcmd, "%P"))

k_value_label = tk.Label(frame_method, text="k (Sensitivity):")
k_value_var = tk.StringVar(value="0.2")
k_value_entry = tk.Entry(frame_method, textvariable=k_value_var, width=5)

r_value_label = tk.Label(frame_method, text="R (Dynamic Range):")
r_value_var = tk.StringVar(value="128")
r_value_entry = tk.Entry(frame_method, textvariable=r_value_var, width=5)

# Parametry pro Simple Recursive Otsu
otsu_levels_label = tk.Label(frame_method, text="Otsu Levels:")
otsu_levels_var = tk.StringVar(value="3")
otsu_levels_entry = tk.Entry(frame_method, textvariable=otsu_levels_var, width=5)
vcmd = root.register(validate_otsu_input)
otsu_levels_entry.configure(validate="key", validatecommand=(vcmd, "%P"))

frame_actions = tk.Frame(root)
frame_actions.pack(fill=tk.X, pady=5)

tk.Button(frame_actions, text="Apply Threshold", command=apply_threshold).pack(pady=5)
tk.Button(frame_actions, text="Step Process (Histogram)", command=step_process).pack(pady=5)

frame_undo_redo = tk.Frame(root)
frame_undo_redo.pack(fill=tk.X, pady=5)

tk.Button(frame_undo_redo, text="Undo", command=undo).grid(row=0, column=0, padx=5)
tk.Button(frame_undo_redo, text="Redo", command=redo).grid(row=0, column=1, padx=5)

toggle_parameter_input(None)

root.mainloop()
