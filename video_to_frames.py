import cv2
import os

def video_to_frames(input_video, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(input_video)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open video file")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Loop through each frame in the video
    frame_count = 0
    while True:
        ret, frame = cap.read()

        # Break the loop if the video is finished
        if not ret:
            break

        # Save the frame as an image
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Frames saved to {output_folder}")

if __name__ == "__main__":
    # Specify the input video file and output folder
    input_video = "path/to/your/video.mp4"
    output_folder = "path/to/your/output/folder"

    # Convert video to frames
    video_to_frames(input_video, output_folder)
