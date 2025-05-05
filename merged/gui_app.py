import tkinter as tk
from tkinter import messagebox, filedialog, ttk, Toplevel, Label, Spinbox, StringVar, IntVar, Text, Scrollbar, Canvas, WORD, BOTH, LEFT, RIGHT, Y, VERTICAL, HORIZONTAL, Scale
from PIL import Image, ImageTk
import threading
import os
import subprocess
import json
from datetime import datetime

# --- SETTINGS ---
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
    return json.load(open(settings_file)) if os.path.exists(settings_file) else default_settings

def save_settings(data):
    with open(settings_file, "w") as f:
        json.dump(data, f, indent=4)

def open_settings():
    global settings_win
    if settings_win and settings_win.winfo_exists():
        settings_win.lift()
        return

    settings = load_settings()
    settings_win = Toplevel(root)
    settings_win.title("Settings")
    settings_win.geometry("400x300")
    settings_win.configure(bg="#eef6fb")
    settings_win.protocol("WM_DELETE_WINDOW", lambda: settings_win.destroy())

    Label(settings_win, text="Detection Settings", font=("Helvetica", 16, "bold"), bg="#eef6fb").pack(pady=10)

    mode_var = StringVar(value=settings.get("mode", "Accurate"))
    ttk.Label(settings_win, text="Detection Mode:", background="#eef6fb").pack()
    ttk.Combobox(settings_win, textvariable=mode_var, values=["Fast", "Accurate"], state="readonly").pack(pady=5)

    device_var = IntVar(value=settings.get("device_id", 0))
    ttk.Label(settings_win, text="Device ID:", background="#eef6fb").pack()
    Spinbox(settings_win, from_=0, to=10, textvariable=device_var).pack(pady=5)

    confidence_var = IntVar(value=settings.get("confidence_threshold", 70))
    ttk.Label(settings_win, text="Confidence Threshold (%):", background="#eef6fb").pack()
    Scale(settings_win, from_=50, to=100, orient=HORIZONTAL, variable=confidence_var).pack(pady=5)

    def save():
        updated = {
            "mode": mode_var.get(),
            "device_id": device_var.get(),
            "confidence_threshold": confidence_var.get(),
            **{k: settings.get(k, v) for k, v in default_settings.items() if k not in ["mode", "device_id", "confidence_threshold"]}
        }
        save_settings(updated)
        messagebox.showinfo("Saved", "Settings saved.")
        settings_win.destroy()

    tk.Button(settings_win, text="Save Settings", command=save, bg="#00ADB5", fg="white", padx=10, pady=5).pack(pady=15)


# --- CHART VIEWER ---
def show_charts(title, folder):
    global open_chart_windows
    if title in open_chart_windows and open_chart_windows[title].winfo_exists():
        open_chart_windows[title].lift()
        return

    win = Toplevel(root)
    win.title(f"{title} Charts")
    win.geometry("900x650")
    win.configure(bg="#f9fbfc")
    open_chart_windows[title] = win
    win.protocol("WM_DELETE_WINDOW", lambda: win.destroy())

    canvas = Canvas(win, bg="#f9fbfc")
    scrollbar = Scrollbar(win, orient=VERTICAL, command=canvas.yview)
    scrollable = tk.Frame(canvas, bg="#f9fbfc")
    scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)
    scrollbar.pack(side=RIGHT, fill=Y)

    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img_path = os.path.join(folder, file)
                img = Image.open(img_path).resize((800, 400), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                lbl = Label(scrollable, image=photo, bg="#f9fbfc")
                lbl.image = photo
                lbl.pack(pady=10)
            except Exception as e:
                print(f"Error loading image {file}: {e}")

def open_chart_window():
    global chart_selector_win
    if chart_selector_win and chart_selector_win.winfo_exists():
        chart_selector_win.lift()
        return

    chart_selector_win = Toplevel(root)
    chart_selector_win.title("Select Chart")
    chart_selector_win.geometry("400x300")
    chart_selector_win.configure(bg="#eef6fb")
    chart_selector_win.protocol("WM_DELETE_WINDOW", lambda: chart_selector_win.destroy())

    Label(chart_selector_win, text="Choose Chart", font=("Helvetica", 16, "bold"), bg="#eef6fb").pack(pady=20)
    charts = {
        "ASL": "E:/Sign_Language_Detection/ASL/chart",
        "BSL": "E:/Sign_Language_Detection/BSL/chart",
        "Facial": "E:/Sign_Language_Detection/Facial_expression/chart"
    }

    for name, path in charts.items():
        ttk.Button(chart_selector_win, text=f"{name} Chart", command=lambda p=path, n=name: show_charts(n, p)).pack(pady=10)

# --- FEEDBACK FUNCTION ---
def open_feedback():
    global feedback_win
    if feedback_win and feedback_win.winfo_exists():
        feedback_win.lift()
        return

    feedback_win = Toplevel(root)
    feedback_win.title("Feedback")
    feedback_win.geometry("500x400")
    feedback_win.configure(bg="#eef6fb")
    feedback_win.protocol("WM_DELETE_WINDOW", lambda: feedback_win.destroy())

    Label(feedback_win, text="📝 We Value Your Feedback", font=("Helvetica", 16, "bold"), bg="#eef6fb").pack(pady=10)
    feedback_box = Text(feedback_win, wrap=WORD, font=("Helvetica", 11), height=15, width=55)
    feedback_box.pack(pady=10, padx=20)

    def submit_feedback():
        text = feedback_box.get("1.0", tk.END).strip()
        if text:
            filename = f"feedback_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
            messagebox.showinfo("Thank You!", "Feedback saved successfully.")
            feedback_win.destroy()
        else:
            messagebox.showwarning("Empty", "Please enter feedback.")

    tk.Button(feedback_win, text="Submit Feedback", command=submit_feedback, bg="#00ADB5", fg="white", padx=10, pady=5).pack(pady=10)

# --- IMAGE LOADER ---
def load_image(path, size=(100, 100)):
    img = Image.open(path)
    return ImageTk.PhotoImage(img.resize(size, Image.LANCZOS))

# --- DETECTION SCRIPT LAUNCHER ---
def launch_detection(script_name, working_directory):
    def run(): 
        os.chdir(working_directory)
        subprocess.run(["python", script_name])
    threading.Thread(target=run).start()

# --- MAIN WINDOW ---
root = tk.Tk()
root.title("Sign Language Recognition Pro")
root.geometry("1300x750")
root.configure(bg="#121212")

tk.Label(root, text="Sign Language Recognition System", font=("Arial Rounded MT Bold", 30),
         fg="#00ADB5", bg="#121212").pack(pady=20)

progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=600, mode='indeterminate')
progress.pack(pady=10)

# Track open windows to prevent duplicates
settings_win = None
feedback_win = None
chart_selector_win = None
open_chart_windows = {}

# --- ICON PATHS: UPDATE THESE TO MATCH YOUR PROJECT ---
icons = {
    "ASL": load_image("E:/Sign_Language_Recognition_System/merged/images/ASL.png"),
    "BSL": load_image("E:/Sign_Language_Recognition_System/merged/images/BSL.png"),
    "Facial": load_image("E:/Sign_Language_Recognition_System/Facial_expression/Facial.png"),
    "ASL + Facial": load_image("E:/Sign_Language_Recognition_System/merged/images/ASLCombined.png"),
    "BSL + Facial": load_image("E:/Sign_Language_Recognition_System/merged/images/BSLCombined.png"),
    "Settings": load_image("E:/Sign_Language_Recognition_System/merged/images/translation.png"),
    "Charts": load_image("E:/Sign_Language_Recognition_System/merged/images/charts.png"),
    "Feedback": load_image("E:/Sign_Language_Recognition_System/merged/images/feedback.png")
}

# --- BUTTONS ---
buttons = [
    ("ASL Detection", lambda: launch_detection("cam_test.py", "E:/Sign_Language_Recognition_System/ASL/src"), "ASL"),
    ("BSL Detection", lambda: launch_detection("cam_test.py", "E:/Sign_Language_Recognition_System/BSL/src"), "BSL"),
    ("Facial Emotion", lambda: launch_detection("realtimedetection.py", "E:/Sign_Language_Recognition_System/Facial_expression/src"), "Facial"),
    ("ASL + Facial", lambda: launch_detection("ASLcombined.py", "E:/Sign_Language_Recognition_System/merged"), "ASL + Facial"),
    ("BSL + Facial", lambda: launch_detection("BSLcombined.py", "E:/Sign_Language_Recognition_System/merged"), "BSL + Facial"),
    ("Settings", open_settings, "Settings"),
    ("Charts", open_chart_window, "Charts"),
    ("Feedback", open_feedback, "Feedback"),
]

frame = tk.Frame(root, bg="#121212")
frame.pack()

def on_enter(e): e.widget['background'] = '#00ADB5'
def on_leave(e): e.widget['background'] = '#222831'

for idx, (text, command, icon) in enumerate(buttons):
    btn = tk.Button(frame, text=text, image=icons[icon], compound="top", command=command,
                    font=("Arial", 12, "bold"), bg="#222831", fg="white",
                    activebackground="#00ADB5", bd=0, padx=20, pady=10)
    btn.grid(row=idx//4, column=idx%4, padx=30, pady=30)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

root.mainloop()
