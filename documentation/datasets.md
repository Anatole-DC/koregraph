## Dataset informations

### Coco annotations

```python
[
  "nose", 
  "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder","right_shoulder", 
  "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", 
  "left_knee", "right_knee", "left_ankle", "right_ankle"
]
```

### 3D&2D Keypoints Annotation

We provide COCO-format keypoints annotation in both 2D and 3D. Each keypoints sequence is stored in a .pkl file with the following attributes:

- **keypoints3d :** Sequences of 3D keypoints reconstructed frame-by-frame. Array shape is (N, 17, 3).
- **keypoints3d_optim :** Sequences of 3D keypoints reconstructed with temporal smoothness and constrains.
- **keypoints2d :** Multi-view frame-by-frame 2D keypoints detection results. Array shape is (9, N, 17, 3). The last dimension is (x, y, confidence).
