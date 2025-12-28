"""GUI application for Mood Detection using CustomTkinter (if available).

This file provides a MoodApp class that wraps the GUI logic and uses
src.inference.MoodPredictor for predictions. It keeps the UI testable and
separable from Tkinter mainloop.
"""
from __future__ import annotations

import os
import cv2
from tkinter.filedialog import askopenfilename
from typing import Optional

try:
    import customtkinter as ctk
except Exception as e:
    raise ImportError(
        "CustomTkinter is required to run this GUI. Install it with: `pip install customtkinter`"
    ) from e

from PIL import Image, ImageTk
try:
    from src.inference import MoodPredictor
except ModuleNotFoundError:
    # Support running this file directly (e.g. `python src/gui/app.py`) by
    # adding the project root to sys.path so absolute `src` imports work.
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.inference import MoodPredictor


class MoodApp:
    def __init__(self, model_path: str = "models/mood.h5"):
        self.predictor = MoodPredictor(model_path)
        self.predictor.load()

        self.root = ctk.CTk()
        self.root.geometry("710x370")
        self.root.title("Mood Detector")
        try:
            # Use package-relative assets path (works when running as module or from repo root)
            icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'icon.ico')
            self.root.iconbitmap(icon_path)
        except Exception:
            pass

        # CustomTkinter settings
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Main UI elements (CustomTkinter widgets only)
        self.select_label = ctk.CTkLabel(self.root, text="\n\n\n\n\n\tSelect Input Way ", font=('calibri', 11, 'bold'), text_color='#aaaaaa')
        self.select_label.place(x=40, y=20)

        self.data_label = ctk.CTkLabel(self.root, font=('Impact', 25), text=" ")
        self.data_label.place(x=420, y=120)

        # Buttons
        self.s_file = ctk.CTkButton(self.root, text='Browse', command=self.from_file, corner_radius=6)
        self.s_file.place(x=75, y=230)

        self.camera_btn = ctk.CTkButton(self.root, text='Camera', command=self.switch_camera, corner_radius=6, width=110)
        self.camera_btn.place(x=75, y=265)

        self.capture_btn = ctk.CTkButton(self.root, text='Capture', command=self.capture, width=40, corner_radius=6)
        self.capture_btn.place(x=185, y=265)

        # Theme toggle (uses CTkButton to toggle appearance mode)
        self.dark_theme = True
        self.theme_btn = ctk.CTkButton(self.root, text='Theme', command=self.theme_switch, corner_radius=6)
        self.theme_btn.place(x=665, y=330)

        # Camera
        self.vid = cv2.VideoCapture(0)
        self.opencv_image: Optional[cv2.Mat] = None
        self.cam_running = False

    def theme_switch(self):
        if self.dark_theme:
            ctk.set_appearance_mode("light")
            self.dark_theme = False
        else:
            ctk.set_appearance_mode("dark")
            self.dark_theme = True

    def open_camera(self):
        if not self.vid or not self.vid.isOpened():
            self.vid = cv2.VideoCapture(0)
        ret, frame = self.vid.read()
        if not ret:
            return
        self.opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        captured_image = Image.fromarray(self.opencv_image)
        photo_image = ImageTk.PhotoImage(image=captured_image)
        # Keep a reference to avoid garbage collection
        self.select_label.photo_image = photo_image
        self.select_label.configure(image=photo_image, text="")
        if self.cam_running:
            self.root.after(30, self.open_camera)

    def switch_camera(self):
        self.cam_running = not self.cam_running
        if self.cam_running:
            self.open_camera()
        else:
            if self.vid:
                self.vid.release()
            # Reset to placeholder text and remove image
            self.select_label.configure(text="\n\n\n\n\n\tSelect Input Way ")
            self.select_label.photo_image = None


    def capture(self):
        if self.opencv_image is None:
            # No frame yet
            return
        res = self.predictor.predict_from_array(self.opencv_image)
        self.data_label.configure(text=f"Mood: {res['mood']} ({res['probability']:.2%})")

    def from_file(self):
        link = askopenfilename()
        if not link:
            return
        my_img = Image.open(link)
        resized_img = my_img.resize((200, 200))
        new_img = ImageTk.PhotoImage(resized_img)
        # Keep a reference on the label to prevent GC
        self.select_label.photo_image = new_img
        self.select_label.configure(image=new_img, text="")
        res = self.predictor.predict_from_file(link)
        self.data_label.configure(text=f"Mood: {res['mood']} ({res['probability']:.2%})")

    def run(self):
        self.root.mainloop()


def main():
    app = MoodApp()
    app.run()


if __name__ == '__main__':
    main()