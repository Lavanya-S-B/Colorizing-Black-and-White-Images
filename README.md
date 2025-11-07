Colorizing Black and White Images using Deep Learning (OpenCV + Caffe Model)

This project automatically colorizes black & white (grayscale) images using a pre-trained deep learning model proposed by Richard Zhang et al. (ECCV 2016). It uses the Caffe colorization model with OpenCV DNN to restore realistic colors without manual editing.

Model Used

File | Purpose
colorization_release_v2.caffemodel | Pre-trained deep learning weights

colorization_deploy_v2.prototxt | Model architecture

pts_in_hull.npy | Cluster centers for ab color space

Note: The .caffemodel file is large and is not stored in GitHub.

Download Model File (Upload your file and replace this link):

MODEL LINK: https://drive.google.com/your-download-link-here

Project Folder Structure

Colorizing-Black-and-White-Images/

│ colorize.py

│ colorization_deploy_v2.prototxt

│ colorization_release_v2.caffemodel (Download separately)

│ pts_in_hull.npy


├── input_images/ (Place grayscale images here)

├── output_images/ (Colorized results are saved here)

└── README.md

Installation

Step 1: Install Dependencies

pip install opencv-python numpy

Step 2: Place Your Input Images

Place grayscale images inside the folder: input_images/

Step 3: Run the Script

python colorize.py

Step 4: View Results

Colorized images are automatically saved inside: output_images/

A preview window will display Original (Left) and Colorized (Right).

Sample Output (Replace images later)

Before (B/W) | After (Colorized)
[Original Image] | [Colorized Image]

Features

• Fully automated colorization

• Works offline

• Preserves high image quality

• Supports multiple images at once

• Deep learning model based on real-world color distribution

Author

Lavanya S. B.

Acknowledgement

This project is based on research work by:

Richard Zhang, Phillip Isola, and Alexei A. Efros (ECCV 2016).

Paper: “Colorful Image Colorization”
