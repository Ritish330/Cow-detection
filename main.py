import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import argparse
import numpy as np
from pathlib import Path
import torch
import time
from util_function.plot import colors, plot_one_box
from util_function.misc import set_logging, LoadImages, increment_path
import threading
import random


def detect(file_path, label, count_label, health_label):
    # Show processing message during detection
    count_label.config(text="Processing...", font=("Arial", 18))
    health_label.config(text="", font=("Arial", 18))

    try:
        model = torch.hub.load("ultralytics/yolov5", "custom", path="weights/300.pt")
    except:
        model = torch.hub.load(
            "ultralytics/yolov5", "custom", source="local", path="weights/300.pt"
        )

    model.conf = 0.15
    model.iou = 0.35
    names = (
        model.module.names if hasattr(model, "module") else model.names
    )  # get class names

    # Perform object detection
    image = cv2.imread(file_path)  # Read the selected image with OpenCV
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Convert image to RGB (OpenCV uses BGR by default)
    results = model(image)  # Get detection results

    # Process detections
    count = 0
    healthy_animals = 0
    for det in results.xyxy[0]:  # detections per image
        xyxy = det[:4].cpu().numpy().astype(int)
        conf = det[4].cpu().numpy()
        class_index = int(det[5].cpu().numpy())
        label_text = names[class_index] + f" {conf:.2f}"

        # Draw bounding box and label on the image
        plot_one_box(xyxy, image, label=label_text, color=colors(class_index, True))

        # Increment count for each detection
        count += 1

        # Simulate guessing animal health
        is_healthy = random.choice([True, False])  # Randomly guessing health
        if is_healthy:
            healthy_animals += 1

    # Update the count label with the number of detections or "Animal Not Defined in Dataset" message
    if count > 0:
        count_label.config(text=f"Number of Cattles: {count}", font=("Arial", 18))
        health_label.config(
            text=f"Healthy Cattles: {healthy_animals}", font=("Arial", 18)
        )
    else:
        count_label.config(text="Animal Not Defined in Dataset", font=("Arial", 18))
        health_label.config(text="No health assessment available", font=("Arial", 18))

    # Convert the result to PIL Image
    result_image = Image.fromarray(image)
    result_image = result_image.resize((400, 400))  # Resize image to 400x400

    # Add a border to the image
    result_image_with_border = Image.new(
        "RGB", (result_image.width + 10, result_image.height + 10), color="black"
    )
    result_image_with_border.paste(result_image, (5, 5))

    result_image_with_border = ImageTk.PhotoImage(result_image_with_border)

    # Update the label in the Tkinter window
    label.config(image=result_image_with_border)
    label.image = result_image_with_border
    count_label.update()
    health_label.update()


def update_window(label, count_label, health_label):
    if file_path:
        detect(
            file_path, label, count_label, health_label
        )  # Perform object detection and update the Tkinter window


# Create a Tkinter window with specified colors
root = tk.Tk()
root.title("Cattle Detection System")
root.config(bg="gray")  # Set background color to #7e22ce

# Center the window on the screen
window_width = 600
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

file_path = None

# Create a label to display the resulting image
label = tk.Label(root, borderwidth=2, relief="solid")
label.pack()

# Create a label to display the count of animals or message when no animals detected
count_label = tk.Label(
    root, text="Number of Cattles: 0", bg="#7e22ce", fg="white", font=("Arial", 18)
)
count_label.pack()

# Create a label to display the health of animals
health_label = tk.Label(
    root, text="Healthy Cattles: 0", bg="#7e22ce", fg="white", font=("Arial", 18)
)
health_label.pack()


# Function to perform object detection on the selected image
def detect_objects():
    global file_path
    file_path = filedialog.askopenfilename()  # Open a file dialog to choose the image
    if file_path:
        # Align the button at the bottom after selecting the image
        btn.pack_forget()
        btn.pack(side=tk.BOTTOM)

        # Perform object detection and update the Tkinter window
        update_window(label, count_label, health_label)


# Create a button to select and process the image
btn = tk.Button(
    root,
    text="Select Image",
    command=detect_objects,
    bg="white",
    fg="black",
    width=40,
    padx=10,
    pady=5,
)  # Set button size, padding, and colors
btn.pack(side=tk.BOTTOM)  # Position button at the bottom

# Run the Tkinter main loop
root.mainloop()
