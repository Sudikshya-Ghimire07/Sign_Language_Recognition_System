import tkinter as tk
from tkinter import messagebox, filedialog, ttk, Toplevel, Label, Spinbox
from tkinter import StringVar, IntVar, Text, Scrollbar, Canvas, WORD, BOTH, LEFT, RIGHT
from tkinter import Y, VERTICAL, HORIZONTAL, Scale
from PIL import Image, ImageTk
import threading
import os
import subprocess
import json
import sys
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


#----------- ASLTest ---------------
def show_asltest_window():
    global open_chart_windows
    title = "ASLTest"
    if title in open_chart_windows and open_chart_windows[title].winfo_exists():
        open_chart_windows[title].lift()
        return

    win = tk.Toplevel(root)
    win.title("ASL Image Test")
    win.geometry("400x250")
    win.configure(bg="#f0f0f0")
    open_chart_windows[title] = win
    win.protocol("WM_DELETE_WINDOW", lambda: win.destroy())

    tk.Label(win, text="Upload an ASL Image", font=("Helvetica", 14), bg="#f0f0f0").pack(pady=20)

    def run_asl_test():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return

        script_path = os.path.join("E:/Sign_Language_Recognition_System/ASL/src", "test.py")

        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"Script not found:\n{script_path}")
            return

            
        # Create a pop-up progress window
        progress_win = tk.Toplevel(win)
        progress_win.title("Processing...")
        progress_win.geometry("300x100")
        progress_win.resizable(False, False)
        progress_win.grab_set()  # Make it modal
        tk.Label(progress_win, text="Running prediction...\nPlease wait.", font=("Arial", 12)).pack(pady=10)

        progress = ttk.Progressbar(progress_win, mode='indeterminate', length=250)
        progress.pack(pady=5)
        progress.start()

        def run_prediction():
            try:
                result = subprocess.run(
                    [sys.executable, script_path, file_path],
                    cwd="E:/Sign_Language_Recognition_System/ASL/src",
                    capture_output=True,
                    text=True
                )
                output = result.stdout.strip()
                if result.returncode != 0:
                    output = result.stderr.strip() or output
                    win.after(0, lambda: messagebox.showerror("Prediction Failed", output))
                else:
                    win.after(0, lambda: [messagebox.showinfo("Prediction Result", output), win.destroy()])

            except Exception as e:
                win.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                win.after(0, progress_win.destroy)

        threading.Thread(target=run_prediction).start()

    tk.Button(win, text="Upload & Predict", command=run_asl_test, bg="#2196F3", fg="white", font=("Arial", 12)).pack(pady=10)



#----------- BSLTest ---------------
def show_bsltest_window():
    global open_chart_windows
    title = "BSLTest"
    if title in open_chart_windows and open_chart_windows[title].winfo_exists():
        open_chart_windows[title].lift()
        return

    win = tk.Toplevel(root)
    win.title("BSL Image Test")
    win.geometry("400x250")
    win.configure(bg="#f0f0f0")
    open_chart_windows[title] = win
    win.protocol("WM_DELETE_WINDOW", lambda: win.destroy())

    tk.Label(win, text="Upload an BSL Image", font=("Helvetica", 14), bg="#f0f0f0").pack(pady=20)

    def run_asl_test():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return

        script_path = os.path.join("E:/Sign_Language_Recognition_System/BSL/src", "test.py")

        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"Script not found:\n{script_path}")
            return

        # Create a pop-up progress window
        progress_win = tk.Toplevel(win)
        progress_win.title("Processing...")
        progress_win.geometry("300x100")
        progress_win.resizable(False, False)
        progress_win.grab_set()  # Make it modal
        tk.Label(progress_win, text="Running prediction...\nPlease wait.", font=("Arial", 12)).pack(pady=10)

        progress = ttk.Progressbar(progress_win, mode='indeterminate', length=250)
        progress.pack(pady=5)
        progress.start()

        def run_prediction():
            try:
                result = subprocess.run(
                    [sys.executable, script_path, file_path],
                    cwd="E:/Sign_Language_Recognition_System/BSL/src",
                    capture_output=True,
                    text=True
                )
                output = result.stdout.strip()
                if result.returncode != 0:
                    output = result.stderr.strip() or output
                    win.after(0, lambda: messagebox.showerror("Prediction Failed", output))
                else:
                   win.after(0, lambda: [messagebox.showinfo("Prediction Result", output), win.destroy()])

            except Exception as e:
                win.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                win.after(0, progress_win.destroy)

        threading.Thread(target=run_prediction).start()

    tk.Button(win, text="Upload & Predict", command=run_asl_test, bg="#2196F3", fg="white", font=("Arial", 12)).pack(pady=10)



#-----------FacialTest ---------------
def show_facialtest_window():
    global open_chart_windows
    title = "FacialTest"
    if title in open_chart_windows and open_chart_windows[title].winfo_exists():
        open_chart_windows[title].lift()
        return

    win = tk.Toplevel(root)
    win.title("Facial Image Test")
    win.geometry("400x250")
    win.configure(bg="#f0f0f0")
    open_chart_windows[title] = win
    win.protocol("WM_DELETE_WINDOW", lambda: win.destroy())

    tk.Label(win, text="Upload an FSL Image", font=("Helvetica", 14), bg="#f0f0f0").pack(pady=20)

    def run_asl_test():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return

        script_path = os.path.join("E:/Sign_Language_Recognition_System/Facial_expression/src", "test.py")
        working_dir = "E:/Sign_Language_Recognition_System/Facial_expression/src"

        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"Script not found:\n{script_path}")
            return

        # Create a pop-up progress window
        progress_win = tk.Toplevel(win)
        progress_win.title("Processing...")
        progress_win.geometry("300x100")
        progress_win.resizable(False, False)
        progress_win.grab_set()
        tk.Label(progress_win, text="Running prediction...\nPlease wait.", font=("Arial", 12)).pack(pady=10)

        progress = ttk.Progressbar(progress_win, mode='indeterminate', length=250)
        progress.pack(pady=5)
        progress.start()

        def run_prediction():
            try:
                result = subprocess.run(
                    [sys.executable, script_path, file_path],
                    cwd=working_dir,
                    capture_output=True,
                    text=True
                )
                output = result.stdout.strip()
                if result.returncode != 0:
                    error_msg = result.stderr.strip() or output
                    win.after(0, lambda: messagebox.showerror("Prediction Failed", error_msg))
                else:
                    win.after(0, lambda: [messagebox.showinfo("Prediction Result", output), win.destroy()])

            except Exception as e:
                win.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                win.after(0, progress_win.destroy)

        threading.Thread(target=run_prediction).start()

    tk.Button(win, text="Upload & Predict", command=run_asl_test, bg="#2196F3", fg="white", font=("Arial", 12)).pack(pady=10)




# --- FEEDBACK FUNCTION ---
def open_feedback():
    global feedback_win
    if feedback_win and feedback_win.winfo_exists():
        feedback_win.lift()
        return

    feedback_win = Toplevel(root)
    feedback_win.title("📝 User Feedback")
    feedback_win.geometry("600x480")
    feedback_win.configure(bg="#eef6fb")
    feedback_win.resizable(False, False)
    feedback_win.protocol("WM_DELETE_WINDOW", feedback_win.destroy)

    Label(feedback_win, text="💬 We Value Your Feedback", font=("Helvetica", 18, "bold"), bg="#eef6fb", fg="#222831").pack(pady=15)
    Label(feedback_win, text="Let us know what you think about the system below:", font=("Helvetica", 11), bg="#eef6fb", fg="#393E46").pack()

    # Frame for text + scrollbar
    text_frame = tk.Frame(feedback_win, bg="#eef6fb")
    text_frame.pack(pady=15, padx=20)

    feedback_box = Text(text_frame, wrap="word", font=("Helvetica", 11), height=12, width=60, bd=2, relief="groove")
    feedback_box.pack(side="left", fill="both", expand=True)

    # Scrollbar
    scrollbar = Scrollbar(text_frame, command=feedback_box.yview)
    scrollbar.pack(side="right", fill="y")
    feedback_box.config(yscrollcommand=scrollbar.set)

    # Auto focus on textbox
    feedback_box.focus_set()

    # Character count label
    char_count = tk.StringVar(value="0 / 1000 characters")
    count_label = Label(feedback_win, textvariable=char_count, font=("Helvetica", 9), bg="#eef6fb", fg="#555")
    count_label.pack()

    def update_char_count(event=None):
        content = feedback_box.get("1.0", "end-1c")
        length = len(content)
        if length > 1000:
            feedback_box.delete("1.0", "end")
            feedback_box.insert("1.0", content[:1000])
            length = 1000
        char_count.set(f"{length} / 1000 characters")

    feedback_box.bind("<KeyRelease>", update_char_count)

    def submit_feedback(event=None):
        text = feedback_box.get("1.0", tk.END).strip()
        if text:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("feedback.txt", "a", encoding="utf-8") as f:
                f.write(f"\n--- Feedback received at {timestamp} ---\n{text}\n")
                f.write("-" * 60 + "\n")
            messagebox.showinfo("Thank You!", "✅ Feedback submitted successfully!")
            feedback_win.destroy()
        else:
            messagebox.showwarning("Empty", "⚠️ Please enter some feedback before submitting.")

    # Submit Button with hover
    submit_btn = tk.Button(
        feedback_win, text="📨 Submit Feedback", command=submit_feedback,
        bg="#00ADB5", fg="white", font=("Helvetica", 12, "bold"), padx=15, pady=8,
        activebackground="#007a85", bd=0, cursor="hand2"
    )
    submit_btn.pack(pady=10)

    def on_hover(e):
        submit_btn.config(bg="#007a85")

    def on_leave(e):
        submit_btn.config(bg="#00ADB5")

    submit_btn.bind("<Enter>", on_hover)
    submit_btn.bind("<Leave>", on_leave)

    # Submit with Ctrl+Enter
    feedback_win.bind('<Control-Return>', submit_feedback)
# --- IMAGE LOADER ---
def load_image(path, size=(100, 100)):
    img = Image.open(path)
    return ImageTk.PhotoImage(img.resize(size, Image.LANCZOS))

# --- DETECTION SCRIPT LAUNCHER ---
def launch_detection(script_name, working_directory):
    def run():
        try:
            root.after(0, lambda: [progress.start(), disable_buttons()])
            os.chdir(working_directory)
            subprocess.run(["python", script_name])
        finally:
            root.after(0, lambda: [progress.stop(), enable_buttons()])
    threading.Thread(target=run).start()

def disable_buttons():
    for widget in pages[current_page].winfo_children():
        if isinstance(widget, tk.Button):
            widget.config(state="disabled")

def enable_buttons():
    for widget in pages[current_page].winfo_children():
        if isinstance(widget, tk.Button):
            widget.config(state="normal")


# --- MAIN WINDOW ---
root = tk.Tk()
root.title("Sign Language Recognition Pro")
root.geometry("1300x750")
root.configure(bg="#121212")

# --- PAGE SETUP ---
pages = [tk.Frame(root, bg="#121212") for _ in range(3)]
for page in pages:
    page.place(relx=0.5, rely=0.5, anchor="center")

current_page = 0

def show_page(index):
    global current_page
    for i, page in enumerate(pages):
        page.place_forget()
    pages[index].place(relx=0.5, rely=0.5, anchor="center")
    current_page = index


tk.Label(root, text="Sign Language Recognition System", font=("Arial Rounded MT Bold", 30),
         fg="#00ADB5", bg="#121212").pack(pady=20)

progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=600, mode='indeterminate')
progress.pack(pady=10)

## Window ##
def on_enter(e):
    e.widget['background'] = '#00ADB5'

def on_leave(e):
    e.widget['background'] = '#222831'


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
     "ASLTest": load_image("E:/Sign_Language_Recognition_System/merged/images/ASLTest.png"),
      "BSLTest": load_image("E:/Sign_Language_Recognition_System/merged/images/BSLTest.png"),
      "FacialTest": load_image("E:/Sign_Language_Recognition_System/merged/images/Facial.png"),
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
     ("ASLTest", show_asltest_window, "ASLTest"),
     ("BSLTest", show_bsltest_window, "BSLTest"),
     ("FacialTest", show_facialtest_window, "FacialTest"),
    ("ASL + Facial", lambda: launch_detection("ASLcombined.py", "E:/Sign_Language_Recognition_System/merged"), "ASL + Facial"),
    ("BSL + Facial", lambda: launch_detection("BSLcombined.py", "E:/Sign_Language_Recognition_System/merged"), "BSL + Facial"),
    ("Settings", open_settings, "Settings"),
    ("Charts", open_chart_window, "Charts"),
    ("Feedback", open_feedback, "Feedback"),
]
# --- PAGE 1 BUTTONS ---  
page1_buttons = [
    ("ASL Detection", lambda: launch_detection("cam_test.py", "E:/Sign_Language_Recognition_System/ASL/src"), "ASL"),
    ("BSL Detection", lambda: launch_detection("cam_test.py", "E:/Sign_Language_Recognition_System/BSL/src"), "BSL"),
    ("Facial Emotion", lambda: launch_detection("realtimedetection.py", "E:/Sign_Language_Recognition_System/Facial_expression/src"), "Facial"),
    ("ASL + Facial", lambda: launch_detection("ASLcombined.py", "E:/Sign_Language_Recognition_System/merged"), "ASL + Facial"),
    ("BSL + Facial", lambda: launch_detection("BSLcombined.py", "E:/Sign_Language_Recognition_System/merged"), "BSL + Facial"),
    ("Next ▶", lambda: show_page(1), "Next")
    
]

# --- PAGE 2 BUTTONS ---  
page2_buttons = [
    ("ASLTest", show_asltest_window, "ASLTest"),
    ("BSLTest", show_bsltest_window, "BSLTest"),
    ("FacialTest", show_facialtest_window, "FacialTest"),
    ("🔁 Back", lambda: show_page(0), "Back"),  
    ("Next ▶", lambda: show_page(2), "Next")
]

# --- PAGE 3 BUTTONS ---  
page3_buttons = [
    ("Charts", open_chart_window, "Charts"),
    ("Feedback", open_feedback, "Feedback"),
    ("Settings", open_settings, "Settings"),
    ("🔁 Back", lambda: show_page(1), "Back")
]
# --- PAGE 1 LABEL (Subheading) ---  
page1_label = tk.Label(pages[0], text="Detections", font=("Arial", 16, "bold"), bg="#222831", fg="white")
page1_label.grid(row=0, column=0, columnspan=3, padx=20, pady=10)

# --- PAGE 2 LABEL (Subheading) ---  
page2_label = tk.Label(pages[1], text="Tests", font=("Arial", 16, "bold"), bg="#222831", fg="white")
page2_label.grid(row=0, column=0, columnspan=3, padx=20, pady=10)

# --- PAGE 3 LABEL (Subheading) ---  
page3_label = tk.Label(pages[2], text="Extras", font=("Arial", 16, "bold"), bg="#222831", fg="white")
page3_label.grid(row=0, column=0, columnspan=3, padx=20, pady=10)

all_pages = [page1_buttons, page2_buttons, page3_buttons]

# Button setup
for page_index, button_list in enumerate(all_pages):
    for idx, (text, command, icon) in enumerate(button_list):
        row = (idx // 3) + 1  # Start placing buttons after the header (row 1 onwards)
        col = idx % 3

        # Text-only buttons for navigation
        if text in ["Next ▶", "🔁 Back"]:
            btn = tk.Button(pages[page_index], text=text, command=command,
                            font=("Arial", 12, "bold"), bg="#00ADB5", fg="white",
                            activebackground="#393E46", bd=0, padx=20, pady=10)
        else:
            btn = tk.Button(pages[page_index], text=text, image=icons[icon], compound="top", command=command,
                            font=("Arial", 12, "bold"), bg="#222831", fg="white",
                            activebackground="#00ADB5", bd=0, padx=20, pady=10)

        # Grid placement for the button
        btn.grid(row=row, column=col, padx=30, pady=30)
        
        # Hover effect bindings
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

# Only call show_page once to display the first page
show_page(0)

root.mainloop()
