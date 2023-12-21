import os
import cv2
from jetson.inference import imageNet, detectNet

def detect_people(image_path, detection_model):
    # Load image
    img = cv2.imread(image_path)

    # Convert image to RGBA (required by detectNet)
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    # Load the image into the detectNet
    detection_model.LoadImage(img_rgba, img.shape[1], img.shape[0])

    # Detect objects in the image
    detections = detection_model.Detect()

    # Filter detections for people
    people_detections = [d for d in detections if d.ClassID == 1]  # Assuming class ID 1 corresponds to people

    return people_detections

def process_folder(folder_path, detection_model):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter out non-MJPEG files
    mjpeg_files = [f for f in files if f.lower().endswith(".mjpeg")]

    # Process each MJPEG file
    for mjpeg_file in mjpeg_files:
        file_path = os.path.join(folder_path, mjpeg_file)

        # Detect people in the MJPEG file
        people_detections = detect_people(file_path, detection_model)

        # Print the results
        print(f"File: {mjpeg_file}")
        print(f"People Detections: {len(people_detections)}")
        for detection in people_detections:
            print(f"  Confidence: {detection.Confidence}")
            print(f"  Location: {detection.Left} {detection.Top} {detection.Right} {detection.Bottom}")
        print("\n")

if __name__ == "__main__":
    # Specify the path to the folder containing MJPEG files
    folder_path = "/path/to/mjpeg/files"

    # Specify the path to the detection model
    detection_model_path = "ssd-mobilenet-v2"

    # Create detectNet object
    detection_model = detectNet.DetectNet(detection_model_path, 0.5)

    # Process the folder
    process_folder(folder_path, detection_model)
