# --- Import Required Libraries ---

import tkinter as tk
from tkinter import messagebox, filedialog, ttk, Toplevel, Label, Spinbox, StringVar, IntVar, BooleanVar, Checkbutton, Entry, Scale, HORIZONTAL, Frame, Text, Scrollbar, Canvas, WORD, BOTH, LEFT, RIGHT, Y, VERTICAL
from PIL import Image, ImageTk
import threading
import os
import subprocess
import cv2
import csv
import json
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------- SETTINGS ----------

settings_file = "settings.json"

default_settings = {
    "mode": "Accurate",
    "device_id": 0,
    "confidence_threshold": 70,
    "camera_flip": False,
    "show_probabilities": False,
    "prediction_interval": 1,
    "save_screenshots": False
}

def load_settings():
    if os.path.exists(settings_file):
        with open(settings_file, "r") as f:
            return json.load(f)
    else:
        return default_settings

def save_settings(data):
    with open(settings_file, "w") as f:
        json.dump(data, f, indent=4)

def open_settings():
    settings = load_settings()

    win = Toplevel(root)
    win.title("Settings")
    win.geometry("550x600")
    win.configure(bg="#eef6fb")
    win.resizable(False, False)

    Label(win, text="⚙️ Detection Settings", font=("Helvetica", 20, "bold"), bg="#eef6fb", fg="#003366").pack(pady=10)

    section_font = ("Helvetica", 14, "bold")
    label_font = ("Helvetica", 11)

    def add_section(title):
        Label(win, text=title, font=section_font, bg="#eef6fb", fg="#0a2940").pack(pady=(20, 10))

    def add_tip(text):
        Label(win, text=text, font=("Helvetica", 9), bg="#eef6fb", fg="#666666").pack()

    # --- Detection Settings ---
    add_section("Detection Settings")

    Label(win, text="Detection Mode:", font=label_font, bg="#eef6fb").pack()
    mode_var = StringVar(value=settings.get("mode", "Accurate"))
    mode_combo = ttk.Combobox(win, textvariable=mode_var, state="readonly", width=30, font=("Helvetica", 10))
    mode_combo['values'] = ['Fast', 'Accurate']
    mode_combo.pack(pady=5)
    add_tip("Fast = quicker but less accurate. Accurate = slower but better results.")

    Label(win, text="Webcam Device ID:", font=label_font, bg="#eef6fb").pack(pady=(10, 0))
    device_var = IntVar(value=settings.get("device_id", 0))
    device_spin = Spinbox(win, from_=0, to=10, textvariable=device_var, width=5, font=("Helvetica", 10))
    device_spin.pack()
    add_tip("Usually 0. Change if multiple webcams are connected.")

    Label(win, text="Confidence Threshold (%):", font=label_font, bg="#eef6fb").pack(pady=(10, 0))
    confidence_var = IntVar(value=settings.get("confidence_threshold", 70))
    confidence_slider = Scale(win, from_=50, to=100, orient=HORIZONTAL, variable=confidence_var,
                               length=250, font=("Helvetica", 10), bg="#eef6fb", troughcolor="#d9e4f5")
    confidence_slider.pack()
    add_tip("Minimum confidence to accept a prediction.")

    # --- Save Settings Button ---
    def save():
        updated_settings = {
            "mode": mode_var.get(),
            "device_id": device_var.get(),
            "confidence_threshold": confidence_var.get(),
            **{k: settings.get(k, v) for k, v in default_settings.items() if k not in ["mode", "device_id", "confidence_threshold"]}
        }
        save_settings(updated_settings)
        messagebox.showinfo("Settings", "Settings saved successfully!")
        win.destroy()

    tk.Button(win, text="Save Settings", command=save,
              font=("Helvetica", 12, "bold"), bg="#00ADB5", fg="white",
              activebackground="#393E46", padx=20, pady=10).pack(pady=20)

# ---------- FEEDBACK ----------

def open_feedback():
    feedback_win = Toplevel(root)
    feedback_win.title("Feedback")
    feedback_win.geometry("450x400")
    feedback_win.configure(bg="#eef6fb")
    feedback_win.resizable(False, False)

    Label(feedback_win, text="📝 Feedback Form", font=("Helvetica", 18, "bold"), bg="#eef6fb", fg="#003366").pack(pady=15)

    Label(feedback_win, text="Enter your feedback below:", font=("Helvetica", 12), bg="#eef6fb").pack()

    text_frame = Frame(feedback_win, bg="#eef6fb")
    text_frame.pack(pady=10, padx=20, fill=BOTH, expand=True)

    feedback_text = Text(text_frame, wrap=WORD, font=("Helvetica", 10), height=10)
    feedback_text.pack(side=LEFT, fill=BOTH, expand=True)

    scrollbar = Scrollbar(text_frame, command=feedback_text.yview)
    scrollbar.pack(side=RIGHT, fill=Y)
    feedback_text.config(yscrollcommand=scrollbar.set)

    def submit_feedback():
        feedback = feedback_text.get("1.0", "end").strip()
        if feedback:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("feedback.txt", "a", encoding="utf-8") as f:
                f.write(f"Time: {current_time}\n")
                f.write(f"Feedback:\n{feedback}\n")
                f.write("-" * 50 + "\n")
            messagebox.showinfo("Thank You!", "✅ Thank you for your feedback!")
            feedback_win.destroy()
        else:
            messagebox.showwarning("Empty", "⚠️ Please enter your feedback before submitting.")

    ttk.Button(feedback_win, text="Submit Feedback", command=submit_feedback).pack(pady=20)


# ---------- CHARTS ----------
def show_charts(title, folder):
    win = Toplevel(root)
    win.title(f"{title} Charts")
    win.geometry("900x650")
    win.configure(bg="#f9fbfc")

    Label(win, text=f"{title} Charts", font=("Helvetica", 18, "bold"), bg="#f9fbfc", fg="#003366").pack(pady=15)

    canvas = Canvas(win, bg="#f9fbfc")
    scrollbar = Scrollbar(win, orient=VERTICAL, command=canvas.yview)
    scrollable = Frame(canvas, bg="#f9fbfc")

    scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=LEFT, fill=BOTH, expand=True)
    scrollbar.pack(side=RIGHT, fill=Y)

    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, file)
            img = Image.open(img_path).resize((700, 400))
            photo = ImageTk.PhotoImage(img)
            lbl = Label(scrollable, image=photo, bg="#f9fbfc")
            lbl.image = photo
            lbl.pack(pady=10)

def open_chart_window():
    win = Toplevel(root)
    win.title("Chart Selection")
    win.geometry("400x300")
    win.configure(bg="#eef6fb")
    win.resizable(False, False)

    Label(win, text="Select Chart", font=("Helvetica", 16, "bold"), bg="#eef6fb", fg="#003366").pack(pady=25)

    charts = {
        "ASL": "E:/Sign_Language_Detection/ASL/chart",
        "BSL": "E:/Sign_Language_Detection/BSL/chart",
        "Facial Expression": "E:/Sign_Language_Detection/Facial_expression/chart"
    }

    for name, path in charts.items():
        ttk.Button(win, text=f"{name} Chart", command=lambda p=path, n=name: show_charts(n, p)).pack(pady=10)

# --- Utility Function to Load and Resize Images for Buttons ---
def load_image(path, size=(100, 100)):
    img = Image.open(path)
    img = img.resize(size, Image.LANCZOS)
    return ImageTk.PhotoImage(img)

# --- Function to Launch External Detection Scripts ---
def launch_detection(title, script_name, working_directory):
    def run():
        os.chdir(working_directory)
        subprocess.run(["python", script_name])
    threading.Thread(target=run).start()

# --- Threaded Functions ---
def threaded_function(func, description):
    def run():
        progress.start()
        func()
        progress.stop()
        messagebox.showinfo("Success", f"{description} Completed Successfully!")
    threading.Thread(target=run).start()

# --- Setup Main Tkinter Window ---
root = tk.Tk()
root.title("Sign Language Recognition Pro")
root.geometry("1300x750")
root.configure(bg="#121212")

header = tk.Label(root, text="Sign Language Recognition System", font=("Arial Rounded MT Bold", 30),
                  fg="#00ADB5", bg="#121212")
header.pack(pady=20)

progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=600, mode='indeterminate')
progress.pack(pady=10)

# Load All Button Icons
icons = {
    "ASL": load_image("E:/Sign_Language_Recognition_System/ASL/ASL.png"),
    "Facial": load_image("E:/Sign_Language_Recognition_System/Facial_expression/Facial.png"),
    "ASL + Facial": load_image("E:/Sign_Language_Recognition_System/merged/images/ASLCombined.png"),
    "BSL": load_image("E:/Sign_Language_Recognition_System/BSL/BSL.png"),
    "BSL + Facial": load_image("E:/Sign_Language_Recognition_System/merged/images/BSLCombined.png"),
    "Settings": load_image("E:/Sign_Language_Recognition_System/merged/images/translation.png"),
    "Charts": load_image("E:/Sign_Language_Recognition_System/merged/images/charts.png"),
    "Feedback": load_image("E:/Sign_Language_Recognition_System/merged/images/feedback.png")
}

# Define Buttons
buttons = [
    ("ASL Detection", lambda: launch_detection("Sign_Language_Recognition_System - ASL Detection", "cam_test.py", "E:/Sign_Language_Recognition_System/ASL/src"), "ASL"),
    ("BSL Detection", lambda: launch_detection("Sign_Language_Recognition_System - BSL Detection", "cam_test.py", "E:/Sign_Language_Recognition_System/BSL/src"), "BSL"),
    ("Facial Emotion", lambda: launch_detection("Sign_Language_Recognition_System - Facial Emotion", "realtimedetection.py", "E:/Sign_Language_Recognition_System/Facial_expression/src"), "Facial"),
    ("ASL + Facial", lambda: launch_detection("Sign_Language_Recognition_System - Combined ASL + Facial", "ASLcombined.py", "E:/Sign_Language_Recognition_System/merged"), "ASL + Facial"),
    ("BSL + Facial", lambda: launch_detection("Sign_Language_Recognition_System - BSL + Facial", "BSLcombined.py", "E:/Sign_Language_Recognition_System/merged"), "BSL + Facial"),
    ("Settings", lambda: open_settings(), "Settings"),
    ("Charts", open_chart_window, "Charts"),
    ("Feedback", lambda: open_feedback(), "Feedback"),
]

frame = tk.Frame(root, bg="#121212")
frame.pack()

# Hover Effects
def on_enter(e):
    e.widget['background'] = '#00ADB5'

def on_leave(e):
    e.widget['background'] = '#222831'

for idx, (text, command, icon_key) in enumerate(buttons):
    btn = tk.Button(
        frame, text=text, image=icons[icon_key], compound="top",
        command=command, font=("Arial", 12, "bold"), bg="#222831", fg="white",
        activebackground="#00ADB5", bd=0, relief="ridge", padx=20, pady=10
    )
    btn.grid(row=idx//4, column=idx%4, padx=30, pady=30)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

# Bottom Frame for Preprocessing, CSV, Training
bottom_frame = tk.Frame(root, bg="#121212")
bottom_frame.pack(pady=30)

# --- Start Tkinter Main Event Loop ---
root.mainloop()
