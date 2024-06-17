import cv2
import argparse
from tkinter import Label, Tk, Button, ttk
from PIL import ImageTk, Image
import json
import os
import re

# Window parameters
WINDOW_SCALE = 0.7
IMAGE_SCALE = 1.5
WINDOW_WIDTH = int(1920 * WINDOW_SCALE)
WINDOW_HEIGHT = int(1280 * WINDOW_SCALE)
IMAGE_WIDTH = int(640 * IMAGE_SCALE)
IMAGE_HEIGHT = int(480 * IMAGE_SCALE)

# Initialize arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="./saved", type=str, help="Path to directory with images")
parser.add_argument("--output", default="./saved.json", type=str, help="Path to output labels.json file")
args = parser.parse_args()

# Parse required arguments
input_directory = os.path.abspath(args.input)
output_path = os.path.abspath(args.output)

# Create UI window
win = Tk()
win.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")


def sort_human(l):
    """
    Human sort for filenames
    :param l: list of items
    :return: Sorted list
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l


# Sort images paths
paths = sort_human([
    os.path.join(input_directory, path)
    for path in os.listdir(input_directory)
    if 'c_' in path and (path.endswith('.jpg') or path.endswith('.png'))
])

# Create counters and tmp variables
images_count = len(paths)
labels = {}
count = 0


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


def switch(i):
    """
    Switch images.
    :param i: image
    """
    label = Label(win, bg="black")
    img = cv2.imread(paths[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    to_display(img, label, WINDOW_WIDTH / 2 - IMAGE_WIDTH / 2, WINDOW_HEIGHT / 2 - IMAGE_HEIGHT / 2, IMAGE_WIDTH, IMAGE_HEIGHT)


def counup(*args):
    """
    Increment counter with overflow check.
    :param args: *For compatibility*
    """
    global count
    count += 1
    if count + 1 > images_count:
        count = 0
    switch(count)


def coundown(*args):
    """
    Decrement counter with overflow check.
    :param args: *For compatibility*
    """
    global count
    count -= 1
    if count < 0:
        count = images_count - 1
    switch(count)


def store_ok(*args):
    """
    Store true label.
    :param args: *For compatibility*
    """
    global count
    labels[paths[count]] = 1


def store_false(*args):
    """
    Store false label.
    :param args: *For compatibility*
    """
    global count
    labels[paths[count]] = 0


def dump_results(*args):
    """
    Dump labeling results into output file.
    :param args: *For compatibility*
    """
    global count
    file = open(output_path, 'w')
    json.dump(labels, file, indent=4)
    file.close()


# Initialize the UI objects
swicth_right_button = Button(win, text="▶", bg="gray", fg="white", command=counup)
switch_left_button  = Button(win, text="◀", bg="gray", fg="white", command=coundown)
label_true = Button(win, text="OK", bg="green", fg="black", command=store_ok)
label_false = Button(win, text="FALSE", bg="red", fg="black", command=store_false)
dump_labels = Button(win, text="Dump Results", bg="green", command=dump_results)

# Place UI objects
panel_y = WINDOW_HEIGHT / 2 + IMAGE_HEIGHT / 2 + 10
swicth_right_button.place(x=WINDOW_WIDTH / 2 + 25, y=panel_y, width=40)
switch_left_button.place(x=WINDOW_WIDTH / 2 - 25, y=panel_y, width=40)
label_true.place(x=WINDOW_WIDTH / 2 - 100 + 35, y=panel_y, width=30)
label_false.place(x=WINDOW_WIDTH / 2 - 160 + 35, y=panel_y, width=60)
dump_labels.place(x=WINDOW_WIDTH / 2 + 75, y=panel_y, width=100)

# Bind callbacks
win.bind("<Left>", coundown)
win.bind("<Right>", counup)
win.bind("t", store_ok)
win.bind("f", store_false)
win.bind("q", dump_results)

# Start mainloop
win.mainloop()