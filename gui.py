import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib
from thresholding import (
    sauvola_threshold,
    otsu_recursive_otsu_gui,
)

# Nastavení backendu pro Matplotlib, aby fungoval s Tkinterem
matplotlib.use("TkAgg")

# Globální proměnné pro historii úprav a aktuální obrázky
history = []
redo_history = []
img = None  # Původní načtený obrázek
img_result = None  # Aktuálně zobrazený obrázek (po zpracování)


# Uložení obrázku do historie a vymazání redo historie
def save_to_history(image):
    if image is not None:
        history.append(image.copy())
        redo_history.clear()


# Funkce pro krok zpět (undo)
def undo():
    if len(history) > 1:
        redo_history.append(history.pop())
        display_image(history[-1])


# Funkce pro krok vpřed (redo)
def redo():
    if redo_history:
        history.append(redo_history.pop())
        display_image(history[-1])


# Načtení obrázku z disku
def load_image():
    global img
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif")])
    if file_path:
        try:
            pil_image = Image.open(file_path).convert("L")  # Načtení a převod na grayscale
            img = np.array(pil_image)
            save_to_history(img)
            display_image(img)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")


# Uložení výsledného obrázku na disk
def save_image():
    if img_result is None:
        messagebox.showerror("Error", "No image to save")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if file_path:
        cv2.imwrite(file_path, img_result)


# Zobrazení obrázku na plátně
def display_image(image):
    global img_result
    img_result = image
    img_disp = Image.fromarray(image)

    # Přizpůsobení velikosti oknu
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
    canvas.image = img_disp  # Uložení reference


# Zobrazení histogramu aktuálního obrázku
def step_process():
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


# Aplikace zvolené prahovací metody
def apply_threshold():
    global img
    if img is None:
        messagebox.showerror("Error", "No image loaded")
        return
    method = method_var.get()
    try:
        if method == "Sauvola":
            result = sauvola_threshold(img,
                                       window_size=int(window_size_var.get()),
                                       k=float(k_value_var.get()),
                                       r=float(r_value_var.get())
                                       )
        elif method == "Recursive Otsu":
            result = otsu_recursive_otsu_gui(
                img,
                bg_est_window=int(bg_window_var.get()),
                bilateral_r=int(bilateral_r_var.get()),
                bilateral_s=int(bilateral_s_var.get()),
                d1=int(d1_var.get()),
                d2=int(d2_var.get()),
                bg_bilateral_r=int(bg_r_var.get()),
                bg_bilateral_s=int(bg_s_var.get()),
                text_bilateral_r=int(text_r_var.get()),
                text_bilateral_s=int(text_s_var.get())
            )
        else:
            messagebox.showerror("Error", "Neznámá metoda")
            return

        save_to_history(result)
        display_image(result)
    except Exception as e:
        messagebox.showerror("Error", str(e))


# Zobrazení správných vstupů podle zvolené metody
def toggle_parameter_input(event):
    for widget in frame_method.grid_slaves():
        if int(widget.grid_info()["row"]) > 0:
            widget.grid_remove()

    # Zobrazení vstupů pro Sauvola
    if method_var.get() == "Sauvola":
        window_size_label.grid(row=1, column=0, padx=5)
        window_size_entry.grid(row=1, column=1, padx=5)
        k_value_label.grid(row=1, column=2, padx=5)
        k_value_entry.grid(row=1, column=3, padx=5)
        r_value_label.grid(row=2, column=0, padx=5)
        r_value_entry.grid(row=2, column=1, padx=5)

    # Zobrazení vstupů pro Recursive Otsu
    elif method_var.get() == "Recursive Otsu":
        bg_window_label.grid(row=1, column=0, padx=5)
        bg_window_entry.grid(row=1, column=1, padx=5)
        bilateral_r_label.grid(row=1, column=2, padx=5)
        bilateral_r_entry.grid(row=1, column=3, padx=5)
        bilateral_s_label.grid(row=2, column=0, padx=5)
        bilateral_s_entry.grid(row=2, column=1, padx=5)
        d1_label.grid(row=2, column=2, padx=5)
        d1_entry.grid(row=2, column=3, padx=5)
        d2_label.grid(row=3, column=0, padx=5)
        d2_entry.grid(row=3, column=1, padx=5)
        bg_r_label.grid(row=3, column=2, padx=5)
        bg_r_entry.grid(row=3, column=3, padx=5)
        bg_s_label.grid(row=4, column=0, padx=5)
        bg_s_entry.grid(row=4, column=1, padx=5)
        text_r_label.grid(row=4, column=2, padx=5)
        text_r_entry.grid(row=4, column=3, padx=5)
        text_s_label.grid(row=5, column=0, padx=5)
        text_s_entry.grid(row=5, column=1, padx=5)


# Validace vstupu pro velikost okna (liché číslo >= 3)
def validate_window_size(value):
    return value == "" or (value.isdigit() and int(value) >= 3 and int(value) % 2 == 1)


# Validace pro vstupy v Recursive Otsu
def validate_otsu_input(value):
    return value == "" or (value.isdigit() and int(value) > 1)


# === GRAFICKÉ ROZHRANÍ (GUI) === #

root = tk.Tk()
root.title("Advanced Document Binarization")
root.geometry("1000x800")

# Horní rám s plátnem pro obrázek
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

# Tlačítka pro nahrání a uložení
frame_controls = tk.Frame(root)
frame_controls.pack(fill=tk.X, pady=5)

tk.Button(frame_controls, text="Load Image", command=load_image).grid(row=0, column=0, padx=5, pady=5)
tk.Button(frame_controls, text="Save Image", command=save_image).grid(row=0, column=1, padx=5, pady=5)

# Výběr metody a vstupní pole
frame_method = tk.Frame(root)
frame_method.pack(fill=tk.X, pady=5)

tk.Label(frame_method, text="Thresholding Method:").grid(row=0, column=0, padx=5)
method_var = tk.StringVar(value="Sauvola")
method_menu = ttk.Combobox(frame_method, textvariable=method_var,
                           values=["Sauvola", "Recursive Otsu"])
method_menu.grid(row=0, column=1, padx=5)
method_menu.bind("<<ComboboxSelected>>", toggle_parameter_input)

# Vstupní pole – Sauvola
window_size_label = tk.Label(frame_method, text="Window Size:")
window_size_var = tk.StringVar(value="15")
window_size_entry = tk.Entry(frame_method, textvariable=window_size_var, width=5)
window_size_entry.configure(validate="key", validatecommand=(root.register(validate_window_size), "%P"))

k_value_label = tk.Label(frame_method, text="k (Sensitivity):")
k_value_var = tk.StringVar(value="0.2")
k_value_entry = tk.Entry(frame_method, textvariable=k_value_var, width=5)

r_value_label = tk.Label(frame_method, text="R (Dynamic Range):")
r_value_var = tk.StringVar(value="128")
r_value_entry = tk.Entry(frame_method, textvariable=r_value_var, width=5)

# Vstupní pole – Recursive Otsu (rozšířený)
bg_window_label = tk.Label(frame_method, text="BG Window:")
bg_window_var = tk.StringVar(value="21")
bg_window_entry = tk.Entry(frame_method, textvariable=bg_window_var, width=5)

bilateral_r_label = tk.Label(frame_method, text="Bilateral R:")
bilateral_r_var = tk.StringVar(value="2")
bilateral_r_entry = tk.Entry(frame_method, textvariable=bilateral_r_var, width=5)

bilateral_s_label = tk.Label(frame_method, text="Bilateral S:")
bilateral_s_var = tk.StringVar(value="10")
bilateral_s_entry = tk.Entry(frame_method, textvariable=bilateral_s_var, width=5)

d1_label = tk.Label(frame_method, text="Min Δ (d1):")
d1_var = tk.StringVar(value="2")
d1_entry = tk.Entry(frame_method, textvariable=d1_var, width=5)

d2_label = tk.Label(frame_method, text="Levels (d2):")
d2_var = tk.StringVar(value="26")
d2_entry = tk.Entry(frame_method, textvariable=d2_var, width=5)

bg_r_label = tk.Label(frame_method, text="BG R:")
bg_r_var = tk.StringVar(value="3")
bg_r_entry = tk.Entry(frame_method, textvariable=bg_r_var, width=5)

bg_s_label = tk.Label(frame_method, text="BG S:")
bg_s_var = tk.StringVar(value="10")
bg_s_entry = tk.Entry(frame_method, textvariable=bg_s_var, width=5)

text_r_label = tk.Label(frame_method, text="Text R:")
text_r_var = tk.StringVar(value="2")
text_r_entry = tk.Entry(frame_method, textvariable=text_r_var, width=5)

text_s_label = tk.Label(frame_method, text="Text S:")
text_s_var = tk.StringVar(value="2")
text_s_entry = tk.Entry(frame_method, textvariable=text_s_var, width=5)

# Tlačítka pro aplikaci prahování a zobrazení histogramu
frame_actions = tk.Frame(root)
frame_actions.pack(fill=tk.X, pady=5)

tk.Button(frame_actions, text="Apply Threshold", command=apply_threshold).pack(pady=5)
tk.Button(frame_actions, text="Step Process (Histogram)", command=step_process).pack(pady=5)

# Tlačítka pro undo/redo
frame_undo_redo = tk.Frame(root)
frame_undo_redo.pack(fill=tk.X, pady=5)

tk.Button(frame_undo_redo, text="Undo", command=undo).grid(row=0, column=0, padx=5)
tk.Button(frame_undo_redo, text="Redo", command=redo).grid(row=0, column=1, padx=5)

# Inicializace vstupů pro výchozí metodu
toggle_parameter_input(None)

# Spuštění GUI aplikace
root.mainloop()
