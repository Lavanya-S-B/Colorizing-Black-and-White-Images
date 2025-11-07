import cv2
import numpy as np
import os
from tkinter import Tk, filedialog

# Hide Tkinter root
Tk().withdraw()

# Select input images
file_paths = filedialog.askopenfilenames(title="Select Input Images", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

# Create output folder
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# Load pre-trained colorization model (from OpenCV)
proto_file = "colorization_deploy_v2.prototxt"
model_file = "colorization_release_v2.caffemodel"
pts_file = "pts_in_hull.npy"

net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
pts = np.load(pts_file)

# Add cluster centers as 1x1 convolution kernel
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype(np.float32)]
net.getLayer(conv8).blobs = [np.ones([1, 313], dtype=np.float32)]

for file_path in file_paths:
    img = cv2.imread(file_path)
    if img is None:
        print(f"Failed to read {file_path}")
        continue

    img_rgb = img.astype(np.float32) / 255.0
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
    l_channel = img_lab[:, :, 0]

    # Resize L channel to network input size
    l_rs = cv2.resize(l_channel, (224, 224))
    l_rs -= 50  # mean-centering

    net.setInput(cv2.dnn.blobFromImage(l_rs))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    lab_out = np.concatenate((l_channel[:, :, np.newaxis], ab), axis=2)
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    bgr_out = np.clip(bgr_out, 0, 1)

    # Convert to 8-bit
    output_img = (255 * bgr_out).astype(np.uint8)

    # Display input and colorized output
    cv2.imshow("Input Image", img)
    cv2.imshow("Colorized Output", output_img)

    # Save output
    base_name = os.path.basename(file_path)
    output_path = os.path.join(output_folder, f"{os.path.splitext(base_name)[0]}_colorized.png")
    cv2.imwrite(output_path, output_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Colorization done!")
