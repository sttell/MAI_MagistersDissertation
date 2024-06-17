from camera import Camera

from tkinter import Label, Tk, Button, ttk
from PIL import ImageTk, Image

import numpy as np
import cv2

import os


# Window parameters
WINDOW_SCALE = 0.7
IMAGE_SCALE = 1.5
WINDOW_WIDTH = int(1920 * WINDOW_SCALE)
WINDOW_HEIGHT = int(1280 * WINDOW_SCALE)
IMAGE_WIDTH = int(640 * IMAGE_SCALE)
IMAGE_HEIGHT = int(480 * IMAGE_SCALE)
PANEL_Y = WINDOW_HEIGHT / 2 + IMAGE_HEIGHT / 2 + 10

# Save path for dataset directory
BASE_SAVE_PATH = "./images"

# Settings for camera
CAMERA_SETTINGS = {
    "RES": {"W": 640, "H": 480},
    "FPS": 30,
    "HOLES_FILL": 3
}


def get_last_free_frame_idx():
    """
    Get last free image id from directory.
    :return: last free image id from directory.
    """
    indices = [int(i.split('c_')[-1].replace('.png', '')) for i in os.listdir(BASE_SAVE_PATH) if i.endswith('.png')]
    return np.max(indices) + 1


# Create camera object and start streaming
camera = Camera(CAMERA_SETTINGS)
camera.start()

# Save last free frame
last_frame_idx = get_last_free_frame_idx()
last_color_frame = 0

# Create UI elements
win = Tk()
win.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")


def to_display(img, label, x, y, w, h):
    """
    Display image.
    :param img: image object
    :param label: label object from Tkinter
    :param x: X-coordinate of place
    :param y: Y-coordinate of place
    :param w: Image width
    :param h: Image height
    """
    image = Image.fromarray(img)
    image = image.resize((w, h))
    pic = ImageTk.PhotoImage(image)
    label.configure(image=pic)
    label.image = pic
    label.place(x=x, y=y)


def print_frames():
    """
    Loop event for frames painting.
    Get camera frames and place it on the UI window.
    """
    global last_color_frame

    # Create window label
    label = Label(win, bg="black")

    # Wait for frames from camera stream
    frames = camera.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame or not color_frame:
        return

    # Update last color frame and display it
    last_color_frame = np.asanyarray(color_frame.get_data())
    last_color_frame = cv2.cvtColor(last_color_frame, cv2.COLOR_BGR2RGB)
    to_display(last_color_frame, label, WINDOW_WIDTH / 2 - IMAGE_WIDTH / 2, WINDOW_HEIGHT / 2 - IMAGE_HEIGHT / 2, IMAGE_WIDTH, IMAGE_HEIGHT)

    # Wait for next frame
    win.after(1000.0 / float(CAMERA_SETTINGS["FPS"]), print_frames)


def save_last_frames(*args):
    """
    Save last frame to directory.
    :param args: *For compatibility*.
    """
    global last_color_frame
    global last_frame_idx
    cv2.imwrite(os.path.join(BASE_SAVE_PATH, f"c_{last_frame_idx}.png"), last_color_frame)
    last_frame_idx += 1


# Add save frame button event
label_true = Button(win, text="Save Frames", bg="green", fg="black", command=save_last_frames)
label_true.place(x=WINDOW_WIDTH / 2 + 15, y=PANEL_Y, width=30)
win.bind("t", save_last_frames)

# Main loop
try:
    print_frames()
    win.mainloop()
finally:
    camera.stop()
