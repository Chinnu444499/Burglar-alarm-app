# app.py
# Burglar Alarm — Final: camera preview auto-resize (fill/crop) option A
# Requirements:
#   pip install customtkinter opencv-python pillow numpy
# Place landing_bg.jpg and alarm.wav in same folder as app.py

import os
import sys
import time
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps
import winsound
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

# ---------------- Resource helper ----------------
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ---------------- Config ----------------
ALARM_FILE = resource_path("alarm.wav")
LANDING_BG = resource_path("landing_bg.jpg")   # put this file in project folder
MOTION_THRESHOLD = 4000
COOLDOWN = 4
ZONE_TIME = 2.0
MIN_CONTOUR_AREA = 350
OVERLAY_ALPHA = 0.45
WINDOW_SIZE = (1150, 680)

# Landing card size (you can change these two)
CARD_WIDTH = 760
CARD_HEIGHT = 440

# fonts / sizes for card
CARD_TITLE_SIZE = 24
CARD_SUBSIZE = 12
OPEN_BTN_WIDTH = 360
OPEN_BTN_HEIGHT = 56

# ------------------------------------------

# Globals
cap = None
prev_gray = None
curr_gray = None
monitoring = False
last_alarm_time = 0
start_time = None

# UI globals
root = None
bg_label = None
_bg_image_orig = None
_bg_bind_id = None

threshold_var = None
zone_var = None
status_var = None

video_label = None   # will display live camera
mask_label = None    # will display mask preview
video_area_widget = None  # container whose size we follow

# ---------------- Utilities ----------------
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray

def count_motion(prev, curr, roi=None, thresh_val=25):
    if roi is not None:
        x, y, w, h = roi
        prev_roi = prev[y:y+h, x:x+w]
        curr_roi = curr[y:y+h, x:x+w]
    else:
        prev_roi = prev
        curr_roi = curr
    diff = cv2.absdiff(prev_roi, curr_roi)
    _, th = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, None, iterations=2)
    cnt = cv2.countNonZero(th)
    return cnt, th

def get_zones(w, h):
    third = max(1, w // 3)
    left = (0, 0, third, h)
    center = (third, 0, third, h)
    right = (2 * third, 0, w - 2 * third, h)
    return {"Left": left, "Center": center, "Right": right}

def play_siren():
    try:
        if os.path.exists(ALARM_FILE):
            winsound.PlaySound(ALARM_FILE, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            winsound.Beep(1500, 700)
    except Exception as e:
        print("Siren playback error:", e)

# ---------------- Monitoring logic ----------------
def start_monitoring():
    global cap, prev_gray, curr_gray, monitoring, last_alarm_time, start_time
    if monitoring:
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Could not open camera. Close other apps using the camera and retry.")
        return
    ret1, f1 = cap.read()
    ret2, f2 = cap.read()
    if not (ret1 and ret2):
        messagebox.showerror("Camera Error", "Could not read frames from camera.")
        cap.release()
        cap = None
        return
    prev_gray = preprocess(f1)
    curr_gray = preprocess(f2)
    globals()["prev_gray"] = prev_gray
    globals()["curr_gray"] = curr_gray
    last_alarm_time = 0
    start_time = time.time()
    monitoring = True
    status_var.set("Monitoring")
    root.after(10, monitor_loop)

def stop_monitoring():
    global monitoring, cap
    monitoring = False
    status_var.set("Stopped")
    if cap:
        try:
            cap.release()
        except:
            pass
        cap = None

def monitor_loop():
    """
    This loop runs on the Tk main thread via root.after.
    It resizes the camera frame to cover the video area (Option A: fill & crop).
    """
    global prev_gray, curr_gray, cap, monitoring, last_alarm_time, start_time
    if not monitoring:
        return

    # read GUI controls
    try:
        threshold = int(threshold_var.get())
    except:
        threshold = MOTION_THRESHOLD
    try:
        zone_time = float(zone_var.get())
    except:
        zone_time = ZONE_TIME

    # get frame and sizes
    ret, frame = cap.read()
    if not ret:
        stop_monitoring()
        return

    frame_h, frame_w = frame.shape[:2]

    # compute zones using camera native size
    zones = get_zones(frame_w, frame_h)
    zone_names = ["Left", "Center", "Right"]
    elapsed = time.time() - start_time
    idx = int((elapsed // zone_time) % 3)
    active_zone_name = zone_names[idx]
    active_roi = zones[active_zone_name]

    # motion detection using prev & curr grayscale frames (frame differencing)
    try:
        motion_pixels, mask_roi = count_motion(prev_gray, curr_gray, roi=active_roi)
    except Exception as e:
        motion_pixels = 0
        mask_roi = np.zeros((active_roi[3], active_roi[2]), dtype=np.uint8)
        print("Motion count error:", e)

    # draw overlays on a copy of frame
    display = frame.copy()
    for name, r in zones.items():
        x, y, w, h = r
        rect_color = (0, 200, 150) if name == active_zone_name else (140, 140, 140)
        cv2.rectangle(display, (x, y), (x + w, y + h), rect_color, 2)
        cv2.putText(display, name, (x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rect_color, 2)

    cv2.putText(display, f"Zone: {active_zone_name}", (10, frame_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.putText(display, f"Motion pixels: {motion_pixels}", (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # build mask_full and draw motion areas (similar to earlier)
    mask_full = np.zeros((frame_h, frame_w), dtype=np.uint8)
    try:
        xR, yR, wR, hR = active_roi
        mask_full[yR:yR+hR, xR:xR+wR] = mask_roi
    except Exception:
        pass

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask_clean = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = display.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        cv2.drawContours(overlay, [cnt], -1, (0,0,255), -1)
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        cv2.rectangle(display, (x_c, y_c), (x_c+w_c, y_c+h_c), (0,255,0), 2)
        cv2.putText(display, f"{int(area)}", (x_c, y_c-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    display = cv2.addWeighted(overlay, OVERLAY_ALPHA, display, 1-OVERLAY_ALPHA, 0)

    # ------------------ Resize to video_area_widget size (Option A: cover/crop) ------------------
    global video_area_widget, video_label, mask_label
    # fallback size if widget not yet available
    target_w, target_h = 640, 480
    try:
        if video_area_widget is not None:
            # use inner available area of video_area (subtract some padding if present)
            video_area_widget.update_idletasks()
            vw = max(20, video_area_widget.winfo_width() - 16)
            vh = max(20, video_area_widget.winfo_height() - 80)  # leave room for label and mask
            # ensure positive
            if vw > 0 and vh > 0:
                target_w, target_h = vw, vh
    except Exception:
        pass

    # convert display BGR->RGB and create PIL image
    try:
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        # ImageOps.fit will resize and crop to fill the target area (cover)
        cover = ImageOps.fit(pil, (target_w, target_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(cover)
        video_label.configure(image=photo)
        video_label.image = photo
    except Exception as e:
        print("Video resize/display error:", e)

    # Mask preview: scale to a proportion of video width (e.g., 0.25 width)
    try:
        mask_nonzero = mask_clean
        # make RGB for display
        mask_color = cv2.cvtColor(mask_nonzero, cv2.COLOR_GRAY2BGR)
        pm = Image.fromarray(mask_color)
        # choose mask preview size: 25% of target_w, maintain ratio
        mw = max(120, int(target_w * 0.22))
        mh = max(90, int(mw * 0.75))
        mask_fit = ImageOps.fit(pm, (mw, mh), Image.NEAREST)
        mask_photo = ImageTk.PhotoImage(mask_fit)
        mask_label.configure(image=mask_photo)
        mask_label.image = mask_photo
    except Exception as e:
        print("Mask resize/display error:", e)

    # ------------------ Alarm logic ------------------
    now = time.time()
    if motion_pixels > threshold:
        if now - last_alarm_time > COOLDOWN:
            last_alarm_time = now
            status_var.set(f"ALARM! {active_zone_name}")
            try:
                os.makedirs("assets", exist_ok=True)
                fname = f"assets/alarm_{int(now)}_{active_zone_name}.jpg"
                cv2.imwrite(fname, display)
                with open("assets/alarm_log.txt", "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}, {active_zone_name}, {fname}\n")
                print("Saved snapshot:", fname)
            except Exception as e:
                print("Error saving snapshot:", e)
            threading.Thread(target=play_siren, daemon=True).start()
    else:
        status_var.set("Monitoring")

    # prepare next frames
    prev_gray = curr_gray
    curr_gray = preprocess(display)
    globals()["prev_gray"] = prev_gray
    globals()["curr_gray"] = curr_gray

    root.after(30, monitor_loop)

def on_exit():
    stop_monitoring()
    try:
        if cap is not None:
            cap.release()
    except:
        pass
    try:
        if _bg_bind_id is not None:
            root.unbind("<Configure>", _bg_bind_id)
    except:
        pass
    root.destroy()

# ---------------- Landing background helpers ----------------
def _load_bg_image():
    global _bg_image_orig
    try:
        img = Image.open(LANDING_BG).convert("RGB")
    except Exception as e:
        print("Could not load landing background:", e)
        _bg_image_orig = None
        return
    _bg_image_orig = img

def _place_bg_label():
    global bg_label, _bg_image_orig, _bg_bind_id
    if _bg_image_orig is None:
        return
    bg_label = tk.Label(root, bd=0)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    def update_bg(event=None):
        if _bg_image_orig is None:
            return
        w = root.winfo_width() or WINDOW_SIZE[0]
        h = root.winfo_height() or WINDOW_SIZE[1]
        resized = ImageOps.fit(_bg_image_orig, (max(2,w), max(2,h)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        bg_label.image = photo
        bg_label.configure(image=photo)
    _bg_bind_id = root.bind("<Configure>", lambda e: update_bg())
    root.after(50, update_bg)

# ---------------- UI builders ----------------
def build_landing_ui():
    _load_bg_image()
    _place_bg_label()
    card = tk.Frame(root, bg="#ffffff", bd=0, relief="flat")
    card.place(relx=0.5, rely=0.5, anchor="center", width=CARD_WIDTH, height=CARD_HEIGHT)
    title = tk.Label(card, text="Burglar Alarm", font=("Segoe UI Semibold", CARD_TITLE_SIZE), bg="#ffffff", fg="#0b1220")
    title.place(relx=0.5, rely=0.18, anchor="center")
    subtitle = tk.Label(card, text="Software burglar robot • Motion detection demo", font=("Segoe UI", CARD_SUBSIZE), bg="#ffffff", fg="#475569")
    subtitle.place(relx=0.5, rely=0.28, anchor="center")
    def on_open():
        try:
            root.unbind("<Configure>")
        except:
            pass
        for w in root.winfo_children():
            w.destroy()
        build_main_ui()
    open_btn = ctk.CTkButton(card, text="Open App", command=on_open, fg_color="#0ea5a4", hover_color="#12b8b4", width=OPEN_BTN_WIDTH, height=OPEN_BTN_HEIGHT)
    open_btn.place(relx=0.5, rely=0.55, anchor="center")
    info = tk.Label(card, text="Camera + Motion Detection — click Open to continue", bg="#ffffff", fg="#475569", font=("Segoe UI", CARD_SUBSIZE))
    info.place(relx=0.5, rely=0.72, anchor="center")
    footer = tk.Label(card, text="Built with OpenCV · CustomTkinter", bg="#ffffff", fg="#6b7280", font=("Segoe UI", 10))
    footer.place(relx=0.5, rely=0.88, anchor="center")

def build_main_ui():
    """
    Main dark monitoring UI. Left control fixed width; right video area expands.
    Also set video_area_widget global so monitor_loop can query its size.
    """
    global threshold_var, zone_var, status_var, video_label, mask_label, video_area_widget
    root.configure(bg="#0f1720")
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("dark-blue")

    main_frame = ctk.CTkFrame(root, corner_radius=12, fg_color="transparent")
    main_frame.pack(fill="both", expand=True, padx=24, pady=18)

    container = ctk.CTkFrame(main_frame, corner_radius=10, fg_color="#121212")
    container.pack(fill="both", expand=True, padx=16, pady=12)

    container.grid_columnconfigure(0, weight=0, minsize=360)
    container.grid_columnconfigure(1, weight=1)
    container.grid_rowconfigure(0, weight=1)

    # left controls
    control_card = ctk.CTkFrame(container, width=340, corner_radius=12)
    control_card.grid(row=0, column=0, padx=(30,16), pady=20, sticky="n")
    ctk.CTkLabel(control_card, text="Idle", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(12,8))
    start_btn = ctk.CTkButton(control_card, text="  ▶  Start Monitoring", command=start_monitoring, width=260, height=40, fg_color="#0b63b8")
    start_btn.pack(pady=(8,6))
    stop_btn = ctk.CTkButton(control_card, text="  ⏸  Stop Monitoring", command=stop_monitoring, width=260, height=40, fg_color="#0b63b8")
    stop_btn.pack(pady=(6,12))
    ctk.CTkLabel(control_card, text="Motion Threshold", anchor="w").pack(padx=16, pady=(6,4), fill="x")
    threshold_var = tk.DoubleVar(value=float(MOTION_THRESHOLD))
    thr_frame = ctk.CTkFrame(control_card, corner_radius=6)
    thr_frame.pack(padx=12, pady=(0,8), fill="x")
    threshold_slider = ctk.CTkSlider(thr_frame, from_=1000, to=30000, number_of_steps=100, variable=threshold_var)
    threshold_slider.pack(padx=8, pady=10, fill="x")
    row = ctk.CTkFrame(control_card, corner_radius=6)
    row.pack(padx=12, pady=(0,8), fill="x")
    ctk.CTkLabel(row, text="⚙ Threshold:").grid(row=0, column=0, padx=8, pady=8, sticky="w")
    curr_label = ctk.CTkLabel(row, textvariable=threshold_var)
    curr_label.grid(row=0, column=1, padx=8, pady=8, sticky="e")
    ctk.CTkLabel(control_card, text="Zone Time (s)", anchor="w").pack(padx=16, pady=(4,4), fill="x")
    zone_var = tk.DoubleVar(value=float(ZONE_TIME))
    zone_slider = ctk.CTkSlider(control_card, from_=0.5, to=5.0, number_of_steps=9, variable=zone_var)
    zone_slider.set(ZONE_TIME)
    zone_slider.pack(padx=12, pady=(0,8), fill="x")
    exit_btn = ctk.CTkButton(control_card, text="  ⛔ Exit", command=on_exit, width=200, fg_color="#ff5f5f")
    exit_btn.pack(pady=(6,14))
    status_var = tk.StringVar(value="Idle")
    ctk.CTkLabel(control_card, textvariable=status_var, fg_color=None).pack(pady=(4,10))

    # right video area
    video_area = ctk.CTkFrame(container, corner_radius=8)
    video_area.grid(row=0, column=1, padx=(8,30), pady=20, sticky="nsew")
    video_area.grid_rowconfigure(0, weight=0)
    video_area.grid_rowconfigure(1, weight=1)  # center video expands
    video_area.grid_columnconfigure(0, weight=1)

    # keep global reference for size queries
    video_area_widget = video_area

    ctk.CTkLabel(video_area, text="Live Camera", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, pady=(8,6))

    video_label = ctk.CTkLabel(video_area, text="")   # image set by monitor_loop()
    video_label.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")

    mask_label = ctk.CTkLabel(video_area, text="")
    mask_label.grid(row=2, column=0, padx=8, pady=(6,12), sticky="n")

    footer = ctk.CTkLabel(root, text="Built with OpenCV • CustomTkinter • Simulated rotating camera", anchor="w")
    footer.pack(side="bottom", fill="x", padx=18, pady=12)

# ---------------- Main ----------------
if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("dark-blue")
    root = ctk.CTk()
    root.title("Burglar Alarm")
    root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
    root.resizable(True, True)
    build_landing_ui()
    root.protocol("WM_DELETE_WINDOW", on_exit)
    root.mainloop()
