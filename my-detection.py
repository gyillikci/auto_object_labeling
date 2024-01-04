from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput

file_name = 'example.txt'

net = detectNet("peoplenet", threshold=0.5)
net.SetTrackingEnabled(True)
net.SetTrackingParams(minFrames=1, dropFrames=15, overlapThreshold=0.5)
camera = videoSource("pedestrians.mp4")      # '/dev/video0' for V4L2
display = videoOutput("display://0") # 'my_video.mp4' for file


with open(file_name, 'w') as file:
    while display.IsStreaming():
        img = camera.Capture()

        if img is None: # capture timeout
            continue

        detections = net.Detect(img)

        for detection in detections:
            if detection.TrackStatus >= 0:  # actively tracking
                print(f"object {detection.TrackID} at ({detection.Left}, {detection.Top}) has been tracked for {detection.TrackFrames} frames {detection.ClassID} Class")
                line = f"object {detection.TrackID} at ({detection.Left}, {detection.Top}) has been tracked for {detection.TrackFrames} frames {detection.ClassID} Class\n"
        
                # Write the line to the file
                file.write(line)
            else:  # if tracking was lost, this object will be dropped the next frame
                print(f"object {detection.TrackID} has lost tracking") 
        
        display.Render(img)
        display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))