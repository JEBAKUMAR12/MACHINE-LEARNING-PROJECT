🚗 Self-Driving Car Steering Simulation (Machine Learning Project)


📌 About


This project demonstrates a conceptual self-driving car steering simulation using machine learning.
It shows how a neural network can predict steering angles based on camera input (images).


👉 The focus is on the pipeline:
Capturing image sequences
Preprocessing data
Feeding to a model
Predicting steering angles
⚠ Note: This is a simulated environment for learning, not for use in real cars.

📊 Dataset
Since this is a simulation, dummy image data is generated within the code:
Black images (66x200 pixels)
A green vertical line at the center (to simulate a lane marker)
👉 In real-world self-driving applications:
You would collect data from a front-facing camera mounted on the car.
The dataset would contain sequences of images and corresponding steering angles.

Example datasets:
Udacity Self-Driving Car Dataset
Comma.ai OpenPilot dataset

🧠 Model
The model architecture used in this project:
TimeDistributed CNN layers: Extract spatial features from each frame.
LSTM layer: Learn temporal dependencies between frames (sequence learning).
Dense layers: Final decision layers to predict the steering angle.

Example architecture:

mathematica
Copy
Edit
Input: Sequence of 5 frames (66x200x3)
 ↓ TimeDistributed Conv2D
 ↓ TimeDistributed Conv2D
 ↓ TimeDistributed Conv2D
 ↓ TimeDistributed Flatten
 ↓ LSTM (100 units)
 ↓ Dense (50 units)
 ↓ Dense (10 units)
 ↓ Dense (1 output -> steering angle)
✅ Trained model expected at models/steering_model.h5.
✅ Model uses MSE loss and Adam optimizer.

⚙ Description


🚀 Pipeline:
1️⃣ Capture or simulate image frames
2️⃣ Preprocess frames (resize, normalize)
3️⃣ Maintain a buffer of recent frames (sequence)
4️⃣ Predict steering angle using the ML model
5️⃣ (Simulate) Apply control based on prediction

🖼 Visualization:
The OpenCV window displays each frame for easy tracking.

📚 Machine Learning Concepts
Supervised Learning:
The model is trained on labeled data (image sequences + steering angle).

Convolutional Neural Networks (CNN):
Automatically extract spatial features like lane lines or edges.

Recurrent Neural Networks (RNN - LSTM):
Learn patterns over time (e.g., curve of the road).

Loss function:
MSE (Mean Squared Error) — measures difference between predicted and actual angles.

Optimization:
Adam optimizer — adjusts weights during training for faster convergence.

💡 Lessons Learned



✅ Combining CNNs and LSTMs helps the model learn both what the road looks like and how it changes over time.
✅ Preprocessing (resizing, normalizing images) is crucial for accurate predictions.
✅ Sequence modeling captures dynamics better than single-frame models in driving tasks.
✅ Simulation helps in safe, low-cost development before deploying on hardware.
✅ Real-time prediction requires efficient code and hardware acceleration (GPU).
