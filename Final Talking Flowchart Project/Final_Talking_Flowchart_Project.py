import cv2
import numpy as np
import pyttsx3
import math
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import scrolledtext
from PIL import Image, ImageTk
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
MIN_SHAPE_AREA = 2000
MAX_SHAPE_AREA = 50000

engine = pyttsx3.init()
engine.setProperty('rate', 140)

# ----------------- SHAPE GEOMETRY HELPERS -----------------

def angle(pt1, pt2, pt0):
    dx1, dy1 = pt1[0] - pt0[0], pt1[1] - pt0[1]
    dx2, dy2 = pt2[0] - pt0[0], pt2[1] - pt0[1]
    dot = dx1 * dx2 + dy1 * dy2
    mag1 = math.hypot(dx1, dy1)
    mag2 = math.hypot(dx2, dy2)
    if mag1 * mag2 == 0:
        return 0
    cosine = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cosine))

def is_oval(approx, w, h, area):
    sides = len(approx)
    if sides <= 6:
        return False
    ellipse_area = math.pi * (w / 2) * (h / 2)
    aspect_ratio = w / float(h)
    area_ratio = area / ellipse_area if ellipse_area > 0 else 0
    return (aspect_ratio < 0.7 or aspect_ratio > 1.3) and (0.7 < area_ratio < 1.3)

def is_rounded_terminal(cnt, approx, w, h, area):
    # Rough detection of rounded rectangle terminals
    if len(approx) < 4:
        return False
    rect_area = w * h
    if rect_area <= 0:
        return False
    extent = float(area) / rect_area
    if extent < 0.6:
        return False
    aspect_ratio = w / float(h)
    if aspect_ratio < 0.4 or aspect_ratio > 2.5:
        return False
    rect_perim = 2.0 * (w + h)
    if rect_perim <= 0:
        return False
    cnt_perim = cv2.arcLength(cnt, True)
    ratio = cnt_perim / rect_perim
    return ratio >= 1.03  # a bit larger than sharp rectangle

def is_diamond(approx):
    sides = len(approx)
    if sides != 4:
        return False
    pts = approx.reshape(4, 2)
    cx, cy = np.mean(pts, axis=0)
    directions = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        angle_deg = np.degrees(np.arctan2(dy, dx)) % 360
        directions.append(angle_deg)
    directions = np.sort(np.array(directions))
    angle_diffs = np.diff(np.concatenate([directions, directions[:1] + 360]))
    return np.all(np.abs(angle_diffs - 90) < 25)

def is_processing(approx, w, h):
    sides = len(approx)
    if sides != 4:
        return False
    pts = approx.reshape(4, 2)
    angles_list = [angle(pts[(j - 1) % 4], pts[(j + 1) % 4], pts[j]) for j in range(4)]
    aspect_ratio = w / float(h)
    return all(80 < a < 100 for a in angles_list) and not (0.9 < aspect_ratio < 1.1)

def is_input_output(approx):
    sides = len(approx)
    if sides != 4:
        return False
    pts = approx.reshape(4, 2)
    angles_list = [angle(pts[(j - 1) % 4], pts[(j + 1) % 4], pts[j]) for j in range(4)]
    return ((angles_list[0] < 80 and angles_list[2] < 80 and angles_list[1] > 100 and angles_list[3] > 100)
            or (angles_list[1] < 80 and angles_list[3] < 80 and angles_list[0] > 100 and angles_list[2] > 100))

def classify_shape(cnt, approx, w, h, area):
    # terminals: oval OR rounded-rectangle
    if is_oval(approx, w, h, area) or is_rounded_terminal(cnt, approx, w, h, area):
        return "Oval"
    if is_diamond(approx):
        return "Decision"
    if is_processing(approx, w, h):
        return "Processing"
    if is_input_output(approx):
        return "Input Output"
    return "Unidentified"

# ----------------- OCR FOR TEXT INSIDE SHAPES -----------------

def shape_text(gray, x, y, w, h, shape_type=None):
    # Padding per shape type
    if shape_type == "Processing":
        pad_x = int(0.08 * w)
        pad_y = int(0.18 * h)
    elif shape_type == "Decision":
        pad_x = int(0.10 * w)
        pad_y = int(0.12 * h)
    else:
        pad_x = int(0.15 * w)
        pad_y = int(0.22 * h)

    x1 = max(x + pad_x, 0)
    y1 = max(y + pad_y, 0)
    x2 = min(x + w - pad_x, gray.shape[1])
    y2 = min(y + h - pad_y, gray.shape[0])

    if x2 <= x1 or y2 <= y1:
        return ""

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return ""

    roi = cv2.resize(roi, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
    try:
        _, th = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except:
        th = roi_blur

    if np.sum(th == 0) > np.sum(th == 255):
        th = cv2.bitwise_not(th)

    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789=+-*/xX×÷().,^%<> "
    config7 = f"--oem 1 --psm 7 -c tessedit_char_whitelist={whitelist}"
    config6 = f"--oem 1 --psm 6 -c tessedit_char_whitelist={whitelist}"
    config11 = f"--oem 1 --psm 11 -c tessedit_char_whitelist={whitelist}"

    def clean(t: str) -> str:
        return re.sub(r"\s+", " ", t.replace("\n", " ")).strip()

    def postprocess(t: str) -> str:
        t = re.sub(r",(?=\S)", ", ", t)
        t = re.sub(r"([=+\-*/×÷%<>])", r" \1 ", t)
        t = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", t)
        t = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", t)
        t = re.sub(r"\bInput(?=[A-Z0-9])", "Input ", t)
        t = re.sub(r"\b(SI)([A-Z])", r"\1 \2", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    if shape_type in ("Decision", "Processing"):
        configs = [config6, config7, config11]
    else:
        configs = [config7, config6, config11]

    for cfg in configs:
        txt = clean(pytesseract.image_to_string(th, config=cfg))
        if len(txt) >= 2:
            return postprocess(txt)

    return ""

# ----------------- NARRATION HELPERS -----------------

def narration_for_shape(shape, text):
    if shape == "Start":
        return "- Flowchart starts with a Start box" + (f" containing {text}" if text else " no text")
    elif shape == "Processing":
        return "- Then a processing box containing " + (text if text else "no text")
    elif shape == "Input Output":
        return "- Then an input output box" + (f" containing {text}" if text else "no text")
    elif shape == "Decision":
        return "- Then a decision box" + (f" containing {text}" if text else "no text")
    elif shape == "Stop":
        return "- Flowchart ends with a Stop box" + (f" containing {text}" if text else " no text")
    else:
        return f", then a {shape} box" + (f" containing {text}" if text else "no text")

# ----------------- TKINTER UI -----------------

root = tk.Tk()
root.title("Talking Flowchart")
root.configure(bg="#eceff1")
root.state('zoomed')

root.grid_columnconfigure(0, minsize=750, weight=7)
root.grid_columnconfigure(1, weight=3)
root.grid_rowconfigure(1, weight=1)

frame_img = tk.Frame(root, bg="#cfd8dc", bd=2, relief=tk.FLAT)
frame_img.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=(10, 5))
frame_img.grid_columnconfigure(0, weight=1)
frame_img.grid_rowconfigure(0, weight=1)
canvas_img = tk.Canvas(frame_img, bg="white", highlightbackground="#78909c")
canvas_img.grid(row=0, column=0, sticky="nsew")

frame_right = tk.Frame(root, bg="#f6f7fb")
frame_right.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=(10, 5))
frame_right.grid_columnconfigure(0, weight=1)
frame_right.grid_rowconfigure(1, weight=1)
frame_right.grid_rowconfigure(3, weight=3)

lbl_shapes = tk.Label(frame_right, text="Flowchart Components (Shapes)", font=("Segoe UI", 16, "bold"), bg="#8ccbf4")
lbl_shapes.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 2))

shapes_txt = scrolledtext.ScrolledText(frame_right, font=("Consolas", 12), fg="#222", bg="#e3e6e8", bd=2, relief=tk.FLAT, height=17)
shapes_txt.grid(row=1, column=0, sticky="nsew", pady=(0, 4))

lbl_narrate = tk.Label(frame_right, text="Narrated Text", font=("Segoe UI", 16, "bold"), bg="#8ccbf4")
lbl_narrate.grid(row=2, column=0, sticky="ew", pady=(0, 2))

narrate_txt = scrolledtext.ScrolledText(frame_right, font=("Consolas", 12), fg="#1565c0", bg="#e3e6e8", bd=2, relief=tk.FLAT, height=17)
narrate_txt.grid(row=3, column=0, sticky="nsew", pady=(0, 4))

cv_img = None
gray_img = None
detected_shapes = []
narration = ""

def resize_image_for_canvas(img, canvas_widget):
    canvas_widget.update()
    canvas_w = canvas_widget.winfo_width()
    canvas_h = canvas_widget.winfo_height()
    h, w = img.shape[:2]
    scale = min(canvas_w / w, canvas_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def upload_image():
    global cv_img, gray_img
    filename = filedialog.askopenfilename(filetypes=[("Image Files", ".png;.jpg;.jpeg;.bmp")])
    if not filename:
        return
    img_raw = cv2.imread(filename)
    if img_raw is None:
        messagebox.showerror("Error", "Image not found or cannot be opened.")
        return
    cv_img = img_raw
    gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    show_img = resize_image_for_canvas(cv_img, canvas_img)
    show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
    show_img = Image.fromarray(show_img)
    show_img = ImageTk.PhotoImage(show_img)
    canvas_img.delete("all")
    canvas_img.create_image(0, 0, image=show_img, anchor=tk.NW)
    canvas_img.image = show_img
    shapes_txt.delete(1.0, tk.END)
    narrate_txt.delete(1.0, tk.END)

def detect_shapes():
    global cv_img, gray_img, detected_shapes
    if cv_img is None:
        messagebox.showwarning("Warning", "Please upload an image first!")
        return

    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected_shapes.clear()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_SHAPE_AREA or area > MAX_SHAPE_AREA:
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        x_mid, y_mid = x + w // 2, y + h // 2

        shape = classify_shape(cnt, approx, w, h, area)
        if shape != "Unidentified":
            shape_item = (shape, (x_mid, y_mid), (x, y, w, h))
            if shape_item not in detected_shapes:
                detected_shapes.append(shape_item)

    detected_shapes.sort(key=lambda item: (item[1][1], item[1][0]))

    # label first and last as Start/Stop for narration (type only)
    if len(detected_shapes) >= 1:
        _, first_center, first_box = detected_shapes[0]
        detected_shapes[0] = ("Start", first_center, first_box)
    if len(detected_shapes) >= 2:
        _, last_center, last_box = detected_shapes[-1]
        detected_shapes[-1] = ("Stop", last_center, last_box)

    shapes_txt.delete(1.0, tk.END)
    shapes_txt.insert(tk.END, "Detected Components:\n================================\n")
    for idx, (shape, (x_mid, y_mid), (x, y, w, h)) in enumerate(detected_shapes, start=1):
        shapes_txt.insert(tk.END, f"{idx}. {shape} at ({x_mid}, {y_mid})\n")

    img_disp = resize_image_for_canvas(cv_img, canvas_img)
    img_annot = img_disp.copy()

    h_orig, w_orig = cv_img.shape[:2]
    h_disp, w_disp = img_disp.shape[:2]
    scale_x = w_disp / float(w_orig)
    scale_y = h_disp / float(h_orig)

    for idx, (shape, (x_mid, y_mid), (x, y, w, h)) in enumerate(detected_shapes, start=1):
        draw_x = int(x_mid * scale_x)
        draw_y = int(y_mid * scale_y)
        cv2.putText(img_annot, str(idx), (draw_x, draw_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)

    img_annot = Image.fromarray(cv2.cvtColor(img_annot, cv2.COLOR_BGR2RGB))
    img_annot = ImageTk.PhotoImage(img_annot)
    canvas_img.delete("all")
    canvas_img.create_image(0, 0, image=img_annot, anchor=tk.NW)
    canvas_img.image = img_annot

def narrate_flowchart():
    global narration
    if not detected_shapes:
        narrate_txt.delete(1.0, tk.END)
        narrate_txt.insert(tk.END, "No shapes detected!\n")
        return

    # ---------- VALIDITY BASED ON TEXT "START" / "STOP" ----------
    # Get OCR text for first and last shapes
    first_shape, (_, _), (fx, fy, fw, fh) = detected_shapes[0]
    last_shape, (_, _), (lx, ly, lw, lh) = detected_shapes[-1]

    first_text = shape_text(gray_img, fx, fy, fw, fh, first_shape) or ""
    last_text = shape_text(gray_img, lx, ly, lw, lh, last_shape) or ""

    first_low = first_text.lower()
    last_low = last_text.lower()

    has_start = "start" in first_low
    has_stop = ("stop" in last_low) or ("end" in last_low)

    if not (has_start and has_stop):
        warning = "- Warning: Invalid flowchart structure. A valid flowchart must include both Start and Stop terminal shapes."
        narrate_txt.delete(1.0, tk.END)
        narrate_txt.insert(tk.END, warning + "\n")
        engine.say(warning)
        engine.runAndWait()
        return
    # -------------------------------------------------------------

    narration_lines = []

    for shape, (x_mid, y_mid), (x, y, w, h) in detected_shapes:
        text = shape_text(gray_img, x, y, w, h, shape)

        if shape == "Start" and not text:
            text = "Start"
        elif shape == "Stop" and not text:
            text = "Stop"

        narration_lines.append(narration_for_shape(shape, text))

    narrate_txt.delete(1.0, tk.END)
    for line in narration_lines:
        narrate_txt.insert(tk.END, line + "\n")

    engine.say(' '.join(narration_lines))
    engine.runAndWait()

def exit_app():
    root.destroy()

btnframe = tk.Frame(root, bg="#eceff1")
btnframe.grid(row=2, column=0, columnspan=2, sticky="ew", padx=30, pady=(4, 18))
for i in range(4):
    btnframe.grid_columnconfigure(i, weight=1)

btn_upload = tk.Button(btnframe, text="Upload Flowchart Image", command=upload_image,
                       font=("Segoe UI", 15), bg="#3949ab", fg="white", bd=0, relief=tk.RAISED)
btn_upload.grid(row=0, column=0, padx=8, pady=8, sticky="ew")

btn_detect = tk.Button(btnframe, text="Detect Shapes", command=detect_shapes,
                       font=("Segoe UI", 15), bg="#1976d2", fg="white", bd=0, relief=tk.RAISED)
btn_detect.grid(row=0, column=1, padx=8, pady=8, sticky="ew")

btn_narrate = tk.Button(btnframe, text="Narrate Flowchart", command=narrate_flowchart,
                        font=("Segoe UI", 15), bg="#00897b", fg="white", bd=0, relief=tk.RAISED)
btn_narrate.grid(row=0, column=2, padx=8, pady=8, sticky="ew")

btn_exit = tk.Button(btnframe, text="Exit", command=exit_app,
                     font=("Segoe UI", 15), bg="#c62828", fg="white", bd=0, relief=tk.RAISED)
btn_exit.grid(row=0, column=3, padx=8, pady=8, sticky="ew")

root.mainloop()
