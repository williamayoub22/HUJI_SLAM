# 🚀 Vision-Aided Navigation and SLAM Final Project
**Hebrew University of Jerusalem - Final Project**  
**Group**: Iris Kaplan (`iris.kaplan1@mail.huji.ac.il`) & William Ayoub (`william.ayoub@mail.huji.ac.il`)

📖 **Read the Full Project Report:** [docs/VAN_ex_Final_Project.pdf](docs/VAN_ex_Final_Project.pdf)

---

## 🗺️ Abstract
We developed and implemented a complete Vision-Aided Navigation and Simultaneous Localization and Mapping (SLAM) library. The primary goal of this project was to estimate the trajectory of a vehicle from a video captured with an onboard stereo camera (KITTI odometry dataset), while simultaneously constructing a 3D map of visual landmarks.

Vision-aided navigation is highly important in robotics and autonomous driving because it provides reliable egomotion estimation and environmental mapping using relatively inexpensive and passive sensors.

The system consists of several robust stages:
1. **Feature Detection and Tracking:** Extracting AKAZE features from stereo pairs, matching them, triangulating them to 3D, and tracking them temporally.
2. **Visual Odometry:** Calculating initial relative poses between frames using Perspective-n-Point (PnP) and RANSAC.
3. **Local Bundle Adjustment:** Refining the camera poses and 3D map locally over keyframe windows to minimize reprojection error and extract reliable covariance estimates.
4. **Loop Closure and Global Optimization:** Using the bundle-adjusted trajectory to build a pose graph, detecting loop closures via Mahalanobis-distance filtering and geometric consensus, and optimizing the global pose graph to eliminate accumulated drift.

<img width="1949" height="705" alt="image" src="https://github.com/user-attachments/assets/78375c83-f1ec-4c84-9a35-8c0514492c06" />


---

## 📦 Dataset Download & Setup
Due to its large size, the dataset of images (extracted video frames) used in this course is published in the repository's **Releases**.

Please extract the ZIP file into a `dataset` folder located next to the `src` folder. After extraction, the directory structure should look like this:

```text
project/
├── dataset/
│   ├── poses/
│   └── sequences/
└── src/
```

Make sure that the `poses` and `sequences` folders are directly inside the `dataset` folder.
