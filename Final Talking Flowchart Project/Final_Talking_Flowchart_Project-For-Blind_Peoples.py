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
import fitz
import os

import threading
import time   

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
MIN_SHAPE_AREA = 1000
MAX_SHAPE_AREA = 50000

engine = pyttsx3.init()
engine.setProperty('rate', 200)
stop_flag = False
run_id = 0  

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

    return (aspect_ratio < 0.85 or aspect_ratio > 1.15) and (0.7 < area_ratio < 1.3)


def is_rounded_terminal(cnt, approx, w, h, area):
    if len(approx) < 4:
        return False

    rect_area = w * h
    if rect_area <= 0:
        return False

    extent = float(area) / rect_area
    if extent < 0.5:
        return False

    aspect_ratio = w / float(h)
    if aspect_ratio < 0.4 or aspect_ratio > 2.5:
        return False

    rect_perim = 2.0 * (w + h)
    if rect_perim <= 0:
        return False

    cnt_perim = cv2.arcLength(cnt, True)
    ratio = cnt_perim / rect_perim

    return ratio >= 1.0

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
    if len(approx) != 4:
        return False

    pts = approx.reshape(4, 2)
    angles_list = [angle(pts[(j - 1) % 4], pts[(j + 1) % 4], pts[j]) for j in range(4)]

    aspect_ratio = w / float(h)

    return all(80 < a < 100 for a in angles_list) and not (0.9 < aspect_ratio < 1.1)


def is_input_output(approx):
    if len(approx) != 4:
        return False

    pts = approx.reshape(4, 2)
    angles_list = [angle(pts[(j - 1) % 4], pts[(j + 1) % 4], pts[j]) for j in range(4)]

    return (
        (angles_list[0] < 80 and angles_list[2] < 80 and angles_list[1] > 100 and angles_list[3] > 100)
        or
        (angles_list[1] < 80 and angles_list[3] < 80 and angles_list[0] > 100 and angles_list[2] > 100)
    )

def classify_shape(cnt, approx, w, h, area):

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

    if shape_type == "Processing":
        pad_x = int(0.08 * w)
        pad_y = int(0.18 * h)

    elif shape_type == "Decision":
        pad_x = int(0.25 * w)
        pad_y = int(0.25 * h)

    elif shape_type == "Oval":
        pad_x = int(0.18 * w)
        pad_y = int(0.25 * h)

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

    # preprocessing
    roi = cv2.resize(roi, None, fx=5.0, fy=5.0, interpolation=cv2.INTER_LANCZOS4)

    roi = cv2.GaussianBlur(roi, (5, 5), 0)

    _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.sum(th == 0) > np.sum(th == 255):
        th = cv2.bitwise_not(th)

    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789=+-*/xX×÷().,^%<>?≥≤:_[]&,!;√"

    config7 = f"--oem 3 --psm 7 -c tessedit_char_whitelist={whitelist}"
    config6 = f"--oem 3 --psm 6 -c tessedit_char_whitelist={whitelist}"
    config11 = f"--oem 3 --psm 11 -c tessedit_char_whitelist={whitelist}"

    def clean(t: str) -> str:
        t = t.replace("\n", " ")
        t = re.sub(r"\s+", " ", t)
        return t.strip()


    def postprocess(t: str) -> str:

        t = re.sub(r",(?=\S)", ", ", t)
        t = re.sub(r"([=+\-*/×÷%<>])", r" \1 ", t)
        t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
        t = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", t)
        t = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", t)

        t = re.sub(r"!+", "!", t)
        t = re.sub(r"\bN!+\b", "N!", t)

        # 🔥 √ fix
        t = re.sub(r"to\s*/\s*N", "to √N", t)

        t = re.sub(r"\s+", " ", t)

        return t.strip()


    # ================= OCR LOGIC =================

    if shape_type in ("Decision", "Processing"):
        configs = [config6, config7, config11]
    else:
        configs = [config7, config6, config11]

    best_text = ""

    for cfg in configs:
        txt = pytesseract.image_to_string(th, config=cfg)
        txt = clean(txt)

        if len(txt) >= 2:
            best_text = postprocess(txt)
            break

    best_text = best_text.strip()

    # 🔥 Keep factorial symbol
    best_text = re.sub(r"[^A-Za-z0-9=+\-*/×÷().,%<>! ]+", "", best_text)

    return best_text
# ----------------- NARRATION HELPERS -----------------

def narration_for_shape(shape, text):

    if not text:
        text = ""

    if shape == "Start":
        return f"The flowchart starts with {text if text else 'a Start step'}."

    elif shape == "Processing":
        return f"Then a processing step: {text}." if text else "Then a processing step."

    elif shape == "Input Output":
        return f"Then an input or output operation: {text}." if text else "Then an input or output operation."

    elif shape == "Decision":
        return f"Then a decision is made: {text}." if text else "Then a decision is made."

    elif shape == "Oval":
        return f"Then a terminal step: {text}." if text else "Then a terminal step."

    elif shape == "Stop":
        return "- Flowchart ends with a Stop box" + (f" containing {text}" if text else " no text")
    else:
        return f", then a {shape} box" + (f" containing {text}" if text else "no text")

def ask_pdf_mode(total_pages):
    popup = tk.Toplevel()
    popup.title("Select PDF Pages")
    popup.geometry("320x280")  
    popup.resizable(False, False)

    tk.Label(popup, text=f"Total pages : {total_pages}",
             font=("Segoe UI", 11, "bold")).pack(pady=10)

    mode = tk.StringVar(value="single")
    page_input = tk.StringVar(value="1")

    def toggle_entry():
        if mode.get() == "all":
            entry.config(state="disabled")
        else:
            entry.config(state="normal")

    # ---------- OPTIONS ----------
    tk.Radiobutton(popup, text="Single Page",
                   variable=mode, value="single",
                   command=toggle_entry).pack(anchor="w", padx=20)

    tk.Radiobutton(popup, text="Page Range (e.g. 2-5)",
                   variable=mode, value="range",
                   command=toggle_entry).pack(anchor="w", padx=20)

    tk.Radiobutton(popup, text="All Pages",
                   variable=mode, value="all",
                   command=toggle_entry).pack(anchor="w", padx=20)

    # ---------- INPUT ----------
    tk.Label(popup, text="Page / Range:").pack(pady=(10, 0))

    entry = tk.Entry(popup, textvariable=page_input, justify="center")
    entry.pack(pady=5)
    entry.focus()

    result = {"mode": None, "page": 1, "start": 1, "end": total_pages}

    # ---------- VALIDATION ----------
    def submit():
        val = page_input.get().strip()

        if mode.get() == "single":
            if not val.isdigit():
                messagebox.showerror("Error", "Enter valid page number")
                return

            p = int(val)
            if p < 1 or p > total_pages:
                messagebox.showerror("Error", "Out of range")
                return

            result["page"] = p

        elif mode.get() == "range":
            if "-" not in val:
                messagebox.showerror("Error", "Use format 2-5")
                return

            try:
                s, e = map(int, val.split("-"))
            except:
                messagebox.showerror("Error", "Invalid range")
                return

            if s < 1 or e > total_pages or s > e:
                messagebox.showerror("Error", "Invalid range")
                return

            result["start"] = s
            result["end"] = e

        result["mode"] = mode.get()
        popup.destroy()

    def cancel():
        popup.destroy()

    # ---------- BUTTONS ----------
    btn_frame = tk.Frame(popup, bg=popup["bg"])
    btn_frame.pack(pady=15)

    tk.Button(
        btn_frame,
        text="OK",
        width=10,
        command=submit,
        bg="#4CAF50",    
        fg="white",
        activebackground="#45a049",
        activeforeground="white",
        bd=0,
        font=("Segoe UI", 10, "bold")
    ).grid(row=0, column=0, padx=5)

    tk.Button(
        btn_frame,
        text="Cancel",
        width=10,
        command=cancel,
        bg="#e74c3c",     
        fg="white",
        activebackground="#c0392b",
        activeforeground="white",
        bd=0,
        font=("Segoe UI", 10)
    ).grid(row=0, column=1, padx=5)

    popup.wait_window()
    return result


# ----------------- TKINTER UI -----------------

import tkinter as tk
from tkinter import scrolledtext

root = tk.Tk()
root.title("Talking Flowchart")
root.configure(bg="#eceff1")
root.state('zoomed')

# -------- GRID SETUP --------
root.grid_columnconfigure(0, minsize=750, weight=7)
root.grid_columnconfigure(1, weight=3)
root.grid_rowconfigure(1, weight=1)

from PIL import Image, ImageTk

root.grid_rowconfigure(1, weight=1)

top_bar = tk.Frame(root, bg="#d6c3a3", height=40, bd=3, relief=tk.RAISED)
top_bar.grid(row=0, column=0, columnspan=3, sticky="ew")

img = Image.open("logo.png")
img = img.resize((40, 36))
logo = ImageTk.PhotoImage(img)

title_frame = tk.Frame(top_bar, bg="#d6c3a3")
title_frame.pack(pady=5)

logo_label = tk.Label(title_frame, image=logo, bg="#d6c3a3", bd=0)
logo_label.pack(side="left", padx=6)
logo_label.image = logo  

lbl_title = tk.Label(
    title_frame,
    text="TALKING FLOWCHART",
    font=("Segoe UI", 14, "bold"),
    bg="#d6c3a3",
    fg="#3e2c1c"
)
lbl_title.pack(side="left")

# -------- LEFT SIDE --------
frame_img = tk.Frame(root, bg="#cfd8dc", bd=1, relief=tk.FLAT)

frame_img.grid(row=1, column=0, sticky="nsew", padx=(5, 10), pady=(6, 4))

frame_img.grid_columnconfigure(0, weight=1)
frame_img.grid_rowconfigure(0, weight=1)

canvas_img = tk.Canvas(frame_img, bg="white", highlightthickness=0)
canvas_img.grid(row=0, column=0, sticky="nsew")

canvas_img.create_text(
    375, 300,
    text="Upload a flowchart or PDF",
    font=("Segoe UI", 16),
    fill="#6b7280"
)


# -------- RIGHT SIDE --------
frame_right = tk.Frame(root, bg="#f6f7fb")

frame_right.grid(row=1, column=1, sticky="nsew", padx=(0, 5), pady=(6, 4))

frame_right.grid_columnconfigure(0, weight=1)
frame_right.grid_rowconfigure(1, weight=1)
frame_right.grid_rowconfigure(3, weight=4)

lbl_shapes = tk.Label(frame_right,
                      text="Flowchart Components (Shapes)",
                      font=("Segoe UI", 16, "bold"),
                      bg="#8ccbf4")
lbl_shapes.grid(row=0, column=0, sticky="ew")

shapes_txt = scrolledtext.ScrolledText(frame_right,
                                       font=("Segoe UI", 11),
                                       fg="black",
                                       bg="#e3e6e8",
                                       height=17)
shapes_txt.grid(row=1, column=0, sticky="nsew")

lbl_narrate = tk.Label(frame_right,
                       text="Narrated Text",
                       font=("Segoe UI", 16, "bold"),
                       bg="#8ccbf4")
lbl_narrate.grid(row=2, column=0, sticky="ew")

narrate_txt = scrolledtext.ScrolledText(frame_right,
                                        font=("Segoe UI", 11),
                                        fg="black",
                                        bg="#e3e6e8",
                                        height=17)
narrate_txt.grid(row=3, column=0, sticky="nsew")

narrate_txt.tag_configure(
    "heading",
    font=("Segoe UI", 12, "bold")
)

narrate_txt.tag_configure(
    "normal",
    font=("Segoe UI", 11)
)

narrate_txt.tag_configure(
    "conclusion",
    font=("Segoe UI", 11, "bold")
)

cv_img = None
gray_img = None
detected_shapes = []

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

    filename = filedialog.askopenfilename(
        filetypes=[
            ("Flowchart Files", "*.png *.jpg *.jpeg *.bmp *.pdf")
        ]
    )

    if not filename:
        return

    ext = os.path.splitext(filename)[1].lower()

    # ---------------- IMAGE FILE ----------------
    if ext in [".png", ".jpg", ".jpeg", ".bmp"]:

        img_raw = cv2.imread(filename)

        if img_raw is None:
            messagebox.showerror("Error", "Image cannot be opened.")
            return

        cv_img = img_raw
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # ---------------- PDF FILE ----------------
    elif ext == ".pdf":

        pdf = fitz.open(filename)
        total_pages = len(pdf)

        images = []

        for i in range(total_pages):
            page = pdf[i]

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            img_data = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img_data.reshape(pix.height, pix.width, pix.n)

            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            images.append(img)

# ===== SINGLE PAGE =====
        if total_pages == 1:

            cv_img = images[0]
            gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

# ===== MULTIPLE PAGES =====
        else:
            choice = ask_pdf_mode(total_pages)

            if not choice or choice.get("mode") is None:
                return
            global pdf_images, current_page

            if choice["mode"] == "single":

                page_index = max(0, min(choice["page"] - 1, total_pages - 1))

                cv_img = images[page_index]
                gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

                show_current_page()   # 🔥 also needed

            elif choice["mode"] == "range":

                narrate_txt.delete(1.0, tk.END)
                shapes_txt.delete(1.0, tk.END)

                start = choice["start"] - 1
                end = choice["end"]

                pdf_images = images[start:end]
                current_page = 0

                cv_img = pdf_images[0]
                gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

                show_current_page()
                detect_shapes()

                threading.Thread(target=process_all_pages, daemon=True).start()

                return

            elif choice["mode"] == "all":

                narrate_txt.delete(1.0, tk.END)
                shapes_txt.delete(1.0, tk.END)

                pdf_images = images
                current_page = 0

                cv_img = pdf_images[0]
                gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

                show_current_page()
                detect_shapes()

                threading.Thread(target=process_all_pages, daemon=True).start()

    # -------- SHOW IMAGE ON SCREEN --------
    show_img = resize_image_for_canvas(cv_img, canvas_img)
    show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
    show_img = Image.fromarray(show_img)
    show_img = ImageTk.PhotoImage(show_img)

    canvas_img.delete("all")
    canvas_img.create_image(0, 0, image=show_img, anchor=tk.NW)
    canvas_img.image = show_img

    shapes_txt.delete(1.0, tk.END)
    narrate_txt.delete(1.0, tk.END)

def show_current_page():
    global cv_img

    show_img = resize_image_for_canvas(cv_img, canvas_img)
    show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
    show_img = Image.fromarray(show_img)
    show_img = ImageTk.PhotoImage(show_img)

    canvas_img.delete("all")
    canvas_img.create_image(0, 0, image=show_img, anchor=tk.NW)
    canvas_img.image = show_img

def process_all_pages():
    global pdf_images, current_page, cv_img, gray_img, stop_flag

    stop_flag = False

    for i in range(len(pdf_images)):

        if stop_flag:
            return

        current_page = i

        cv_img = pdf_images[i]
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        show_current_page()
        detect_shapes()

        narrate_txt.insert(tk.END, f"\n===== PAGE {i+1} =====\n")

        root.update() 

        if i == 0:
            time.sleep(0.4)

        narrate_flowchart()

        time.sleep(0.5)

def detect_shapes():
    global cv_img, gray_img, detected_shapes

    if cv_img is None:
        messagebox.showwarning("Warning", "Please upload an image first!")
        return

    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 3
    )

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

    if not detected_shapes:
        shapes_txt.delete(1.0, tk.END)
        shapes_txt.insert(tk.END, "⚠️ No shapes detected! Try another image.\n")

        messagebox.showwarning("Warning", "No shapes detected!")
        return

    detected_shapes.sort(key=lambda item: (item[1][1], item[1][0]))

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

        cv2.putText(
            img_annot,
            str(idx),
            (draw_x, draw_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (0, 0, 255),
            1
        )

    img_annot = Image.fromarray(cv2.cvtColor(img_annot, cv2.COLOR_BGR2RGB))
    img_annot = ImageTk.PhotoImage(img_annot)

    canvas_img.delete("all")
    canvas_img.create_image(0, 0, image=img_annot, anchor=tk.NW)
    canvas_img.image = img_annot

def read_full_page_text():
    global gray_img, detected_shapes

    if gray_img is None:
        return "", ""

    if not detected_shapes:
        text = pytesseract.image_to_string(gray_img)
        text = re.sub(r'\s+', ' ', text).strip()
        return text, ""

    top_y = min(box[1] for _, _, box in detected_shapes)
    bottom_y = max(box[1] + box[3] for _, _, box in detected_shapes)

    top_region = gray_img[0:top_y, :]
    bottom_region = gray_img[bottom_y:, :]

    top_text = pytesseract.image_to_string(top_region)
    bottom_text = pytesseract.image_to_string(bottom_region)

    top_text = re.sub(r'\s+', ' ', top_text).strip()
    bottom_text = re.sub(r'\s+', ' ', bottom_text).strip()

    return top_text, bottom_text

def narrate_flowchart():
    global stop_flag, run_id

    stop_flag = False 

    run_id += 1
    my_id = run_id

    if not detected_shapes:
        narrate_txt.delete(1.0, tk.END)
        narrate_txt.insert(tk.END, "No shapes detected!\n")
        return
    
    top_text, bottom_text = read_full_page_text()

    narrate_txt.delete(1.0, tk.END)

    if stop_flag:
        return

    # -------- TOP TEXT --------
    if top_text:
        narrate_txt.insert(tk.END, top_text + "\n\n")

        text = top_text
        text = text.replace("Topic:", "Topic:||")
        text = text.replace("Introduction", "||Introduction||")
        text = text.replace("Conclusion", "||Conclusion||")

        parts = text.split("||")

        for part in parts:
            if stop_flag:
                return

            part = part.strip()

            if part:
                engine.say(part)
                engine.runAndWait()

                time.sleep(0.05)

    time.sleep(0.1)
    
    if stop_flag:
        return

    # -------- INTRO --------
    intro = "Now i will explain the flowchart for you"

    narrate_txt.insert(tk.END, intro + "\n\n")

    engine.say("")
    engine.runAndWait() 

    if stop_flag:
        return

    engine.say(intro)
    engine.runAndWait()

    time.sleep(0.2) 

    # ---------- VALIDITY BASED ON TEXT "START" / "STOP" ----------

    first_shape, (_, _), (fx, fy, fw, fh) = detected_shapes[0]
    last_shape,  (_, _), (lx, ly, lw, lh) = detected_shapes[-1]

    first_text = shape_text(gray_img, fx, fy, fw, fh, first_shape) or ""
    last_text  = shape_text(gray_img, lx, ly, lw, lh, last_shape) or ""

    first_low = first_text.lower()
    last_low  = last_text.lower()

    start_words = ["start", "begin", "init"]
    stop_words  = ["stop", "end", "finish", "terminate", "exit", "done"]

    has_start = any(word in first_low for word in start_words)
    has_stop  = any(word in last_low  for word in stop_words)

    if not (has_start and has_stop):

        warning = "- Warning: Invalid flowchart structure. A valid flowchart must include both Start and Stop terminal shapes."
        if stop_flag:
            return
        narrate_txt.insert(tk.END, warning + "\n")

        engine.say(warning)
        engine.runAndWait()

        return
    narration_lines = []
    visited = set()

    def get_next_shape(current, direction=None):
        _, (cx, cy), _ = current

        candidates = []
        for s in detected_shapes:
            _, (sx, sy), _ = s

            if sy > cy:
                dist = (sx - cx)**2 + (sy - cy)**2
                candidates.append((dist, s))

        return min(candidates, key=lambda x: x[0])[1] if candidates else None

    def get_branches(decision):
        _, (cx, cy), _ = decision

        left_candidates = []
        right_candidates = []

        for s in detected_shapes:
            _, (sx, sy), _ = s

            if sy > cy:
                dx = sx - cx
                dy = sy - cy
                dist = dx*dx + dy*dy

                if dx < -30:
                    left_candidates.append((dist, s))
                elif dx > 30:
                    right_candidates.append((dist, s))

        left = min(left_candidates, key=lambda x: x[0])[1] if left_candidates else None
        right = min(right_candidates, key=lambda x: x[0])[1] if right_candidates else None

        return left, right

    current = detected_shapes[0]

    while current and current not in visited:
        visited.add(current)

        shape, (_, _), (x, y, w, h) = current
        text = shape_text(gray_img, x, y, w, h, shape)

        # skip YES/NO labels
        if text.lower().strip() in ["yes", "no", "y", "n"]:
            current = get_next_shape(current)
            continue

        narration_lines.append(narration_for_shape(shape, text))

        # ================= DECISION HANDLING (FIXED + NESTED) =================
        if shape == "Decision":
            left, right = get_branches(current)

            branches = []

            if left:
                branches.append(("YES", left))
            if right:
                branches.append(("NO", right))

            if len(branches) < 2:
                _, (cx, cy), _ = current

                extra_candidates = []
                for s in detected_shapes:
                    if s == current:
                        continue

                    _, (sx, sy), _ = s
                    if sy > cy:
                        dist = (sx - cx)**2 + (sy - cy)**2
                        extra_candidates.append((dist, s))

                extra_candidates.sort()

                for _, s in extra_candidates:
                    if s != left and s != right:
                        if len(branches) == 0:
                            branches.append(("YES", s))
                        else:
                            branches.append(("NO", s))
                        break

            for label, branch_node in branches:
                narration_lines.append(f"If the condition is {label}:")

                temp = branch_node
                local_visited = set()

                while temp and temp not in local_visited:
                    local_visited.add(temp)

                    s, (_, _), (tx, ty, tw, th) = temp
                    t = shape_text(gray_img, tx, ty, tw, th, s)

                    if s == "Decision":
                        narration_lines.append("    " + narration_for_shape(s, t))

                        sub_left, sub_right = get_branches(temp)

                        if sub_left:
                            narration_lines.append("    If the condition is YES:")
                            sub_temp = sub_left
                            sub_visited = set()

                            while sub_temp and sub_temp not in sub_visited:
                                sub_visited.add(sub_temp)

                                ss, (_, _), (sx, sy, sw, sh) = sub_temp
                                tt = shape_text(gray_img, sx, sy, sw, sh, ss)

                                if tt.lower().strip() not in ["yes", "no", "y", "n"]:
                                    narration_lines.append("        " + narration_for_shape(ss, tt))

                                if ss == "Stop":
                                    break

                                sub_temp = get_next_shape(sub_temp)

                        if sub_right:
                            narration_lines.append("    If the condition is NO:")
                            sub_temp = sub_right
                            sub_visited = set()

                            while sub_temp and sub_temp not in sub_visited:
                                sub_visited.add(sub_temp)

                                ss, (_, _), (sx, sy, sw, sh) = sub_temp
                                tt = shape_text(gray_img, sx, sy, sw, sh, ss)

                                if tt.lower().strip() not in ["yes", "no", "y", "n"]:
                                    narration_lines.append("        " + narration_for_shape(ss, tt))

                                if ss == "Stop":
                                    break

                                sub_temp = get_next_shape(sub_temp)

                        break

                    elif t.lower().strip() not in ["yes", "no", "y", "n"]:
                        narration_lines.append("    " + narration_for_shape(s, t))

                    if s == "Stop":
                        break

                    next_node = get_next_shape(temp)

                    if not next_node or next_node == temp:
                        break

                    temp = next_node

        if shape != "Decision":
            current = get_next_shape(current)
        else:
            current = None

            
          # OUTPUT
    for line in narration_lines:
        if stop_flag or my_id != run_id:
            return

        narrate_txt.insert(tk.END, line + "\n")
        narrate_txt.see(tk.END)
        root.update_idletasks()

        if stop_flag or my_id != run_id:
            return

        engine.say(line)
        engine.runAndWait()

        if stop_flag or my_id != run_id:
            return

        time.sleep(0.2)  


# -------- SMART CONCLUSION --------

    steps = []

    for shape, (_, _), (x, y, w, h) in detected_shapes:
        text = shape_text(gray_img, x, y, w, h, shape).lower()
        if text:
            steps.append((shape, text))

    inputs = []
    outputs = []
    operations = []
    decisions = []

    for shape, text in steps:

        if shape == "Input Output":
            if "enter" in text or "input" in text:
                inputs.append(text)
            else:
                outputs.append(text)

        elif shape == "Processing":
            operations.append(text)

        elif shape == "Decision":
            decisions.append(text)


# -------- FLOW TYPE DETECTION --------
        all_text = " ".join(inputs + operations + outputs).lower()

        flow_type = "step-by-step process"

        if any(k in all_text for k in ["celsius", "fahrenheit", "kelvin"]):
            flow_type = "temperature conversion"

        elif any(k in all_text for k in ["login", "password", "username", "otp", "verify"]):
            flow_type = "authentication"

        elif any(k in all_text for k in ["even", "odd", "prime"]):
            flow_type = "number checking"

        elif any(k in all_text for k in ["area", "perimeter", "circle", "rectangle", "triangle"]):
            flow_type = "geometry calculation"

        elif any(k in all_text for k in ["meter", "km", "cm", "kg", "gram", "convert"]):
            flow_type = "unit conversion"

        elif any(k in all_text for k in ["marks", "grade", "percentage", "result"]):
            flow_type = "grading system"

        elif any(k in all_text for k in ["salary", "tax", "income", "bonus"]):
            flow_type = "financial calculation"

        elif any(k in all_text for k in ["balance", "withdraw", "deposit", "account"]):
            flow_type = "banking process"

        elif any(k in all_text for k in ["loop", "repeat", "until", "while"]):
            flow_type = "loop-based process"

        elif any(k in all_text for k in ["factorial", "series", "sum"]):
            flow_type = "mathematical series"

        elif any(op in all_text for op in ["+", "-", "*", "/", "%"]):
            flow_type = "calculation"

        elif decisions:
            flow_type = "decision-based process"

# -------- BUILD FINAL SENTENCE --------
    if inputs and outputs:
        conclusion = f"This is a {flow_type} flowchart. It takes {', '.join(inputs)} and produces {', '.join(outputs)}."

    elif inputs and operations:
        conclusion = f"This is a {flow_type} flowchart. It processes {', '.join(inputs)} using {', '.join(operations)}."

    elif operations and outputs:
        conclusion = f"This is a {flow_type} flowchart. It performs {', '.join(operations)} to produce {', '.join(outputs)}."

    elif operations:
        conclusion = f"This is a {flow_type} flowchart performing {', '.join(operations)}."

    else:
        conclusion = f"This flowchart shows a {flow_type}."


# -------- OUTPUT --------
    if stop_flag:
        return

    narrate_txt.insert(tk.END, "\n--- Conclusion ---\n")
    narrate_txt.insert(tk.END, conclusion + "\n")

    if stop_flag:
        return

    engine.stop()  
    engine.say("Conclusion. " + conclusion)
    engine.runAndWait()

# -------- READ REMAINING PAGE TEXT --------
    if bottom_text and bottom_text.strip():

        if stop_flag:
            return

        msg = "Next part says"

        narrate_txt.insert(tk.END, "\n" + msg + "\n\n")
        narrate_txt.insert(tk.END, bottom_text + "\n")

        if stop_flag:
            return

        engine.stop()

        engine.say(msg)

        sentences = bottom_text.replace('\n', ' ').split('.')

        for sentence in sentences:
            if stop_flag:
                return

            sentence = sentence.strip()
            if sentence:
                engine.say(sentence)

        engine.runAndWait()

def reset_all():
    global cv_img, gray_img, detected_shapes, stop_flag, run_id, engine

    stop_flag = True
    run_id += 1 

    try:
        engine.stop()
    except:
        pass

    engine = pyttsx3.init()
    engine.setProperty('rate', 110)

    narrate_txt.delete(1.0, tk.END)
    shapes_txt.delete(1.0, tk.END)

    detected_shapes.clear()
    cv_img = None
    gray_img = None

    canvas_img.delete("all")
    canvas_img.create_text(
        300, 200,
        text="Upload a flowchart or PDF",
        font=("Segoe UI", 16),
        fill="gray"
    )

def repeat_narration():
    global stop_flag, run_id

    stop_flag = True
    run_id += 1  

    try:
        engine.stop()
    except:
        pass

    time.sleep(0.2)

    stop_flag = False
    threading.Thread(target=narrate_flowchart).start()

btnframe = tk.Frame(root, bg="#eceff1")
btnframe.grid(row=2, column=0, columnspan=2, sticky="ew", padx=30, pady=(4, 18))

for i in range(5):
    btnframe.grid_columnconfigure(i, weight=1)


# -------- BUTTONS --------
btn_upload = tk.Button(
    btnframe,
    text="Upload Flowchart",
    command=upload_image,
    font=("Segoe UI", 15, "bold"),
    bg="#3949ab",
    fg="white",
    bd=0,
    relief=tk.RAISED
)
btn_upload.grid(row=0, column=0, padx=8, pady=8, sticky="ew")


btn_detect = tk.Button(
    btnframe,
    text="Detect Shapes",
    command=detect_shapes,
    font=("Segoe UI", 15, "bold"),
    bg="#1976d2",
    fg="white",
    bd=0,
    relief=tk.RAISED
)
btn_detect.grid(row=0, column=1, padx=8, pady=8, sticky="ew")


btn_narrate = tk.Button(
    btnframe,
    text="Narrate Flowchart",
    command=lambda: threading.Thread(target=narrate_flowchart).start(),
    font=("Segoe UI", 15, "bold"),
    bg="#00897b",
    fg="white",
    bd=0,
    relief=tk.RAISED
)
btn_narrate.grid(row=0, column=2, padx=8, pady=8, sticky="ew")


btn_repeat = tk.Button(
    btnframe,
    text="Repeat",
    command=repeat_narration,  
    font=("Segoe UI", 15, "bold"),
    bg="#6a1b9a",
    fg="white",
    bd=0,
    relief=tk.RAISED
)
btn_repeat.grid(row=0, column=3, padx=8, pady=8, sticky="ew")

btn_stop = tk.Button(
    btnframe,
    text="Reset",
    command=reset_all,
    font=("Segoe UI", 15, "bold"),
    bg="#c62828",
    fg="white",
    bd=0,
    relief=tk.RAISED
)
btn_stop.grid(row=0, column=4, padx=8, pady=8, sticky="ew")

root.mainloop()