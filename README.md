# Skin Cancer Detection System (Melanoma vs. Benign) 🎗️

This project implements an end-to-end Deep Learning solution for detecting Melanoma skin cancer. It utilizes **EfficientNet-B0** for high-accuracy image classification and provides a user-friendly Web Interface powered by **FastAPI** (Backend) and **HTML/CSS** (Frontend).

---

## 👥 Our Team & Roles

| Team Member | Role | Key Responsibilities |
| :--- | :--- | :--- |
| **Yasmin Ehab** | Team Leader & Lead Frontend | UI/UX Refactoring, System Integration, Project Management. |
| **Mariam Sameh** | AI Engineer | Data Pipeline, Model Training & Optimization. |
| **Yousef Atef** | AI Engineer | Model Architecture, Transfer Learning. |
| **Ali Mohammed** | Data Scientist | Data Analysis, Quality Assurance, Model Improvement. |
| **Youssef Haitham** | Frontend Developer | Initial Web Layout, Prototyping. |
| **Osama Ghonim** | Backend Developer | API Logic, Server Deployment. |

---

## 🚀 Project Pipeline Details

### 1. Problem Definition and Data Collection
* **Objective:** To build a binary classification model capable of distinguishing between **Benign** skin lesions and **Malignant (Melanoma)** skin cancer.
* **Dataset:** The entire team collaborated to gather a diverse labeled dataset of skin lesion images, ensuring both high quality for training and visual clarity for the user interface display.

### 2. Data Cleaning and Analysis
* **Verification:** Verified image formats and directory structures.
* **Normalization:** Applied statistical normalization using ImageNet mean and standard deviation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) to ensure faster convergence during training.
* **Visual Quality Check:** The Frontend team (Yasmin & Youssef H.) assisted in filtering out low-quality images that would not display well in the application.

### 3. Feature Engineering (Data Augmentation)
To improve model generalization and prevent overfitting, we applied various data augmentation techniques using `torchvision.transforms`:
* **Resizing:** All images resized to `224x224`.
* **Random Horizontal Flip:** To handle orientation invariance.
* **Random Rotation:** Rotated images by up to 10 degrees.
* **Color Jitter:** Slight adjustments to brightness and contrast to handle lighting variations.

### 4. Model Design
We employed **Transfer Learning** using the **EfficientNet-B0** architecture.
* **Base Model:** EfficientNet-B0 (Pre-trained on ImageNet).
* **Integration Strategy:** The design phase involved collaboration between AI and Frontend leads (Yasmin) to ensure the model's output format matches the UI requirements.
* **Custom Head:** We replaced the default classifier with a custom Sequential layer:
    * `Dropout (0.5)` for regularization.
    * `Linear` layer (1280 -> 512).
    * `BatchNorm1d` for stable training.
    * `GELU` activation function.
    * Final `Linear` output layer (512 -> 2 classes).

### 5. Model Training
The model was trained using the following configuration:
* **Optimizer:** `AdamW` (Weight Decay: 1e-4).
* **Loss Function:** `CrossEntropyLoss` with label smoothing (0.05).
* **Scheduler:** `OneCycleLR` for dynamic learning rate adjustment.
* **Epochs:** 15.
* **Hardware:** Trained on NVIDIA CUDA GPU.
* **Best Validation Accuracy:** Saved the model weights with the highest validation accuracy during training.

### 6. Model Testing and Inference
We evaluated the model on a separate Test set containing 2000 images.
* **Final Test Accuracy:** **96.40%**
* **System Testing:** While AI engineers focused on accuracy, the Frontend and Backend developers tested the inference speed and system integration to ensure real-time performance.
* **Inference Logic:** The model outputs logits, which are converted to probabilities using `Softmax`.

### 7. GUI Implementation & Application Running
**(Bonus Achieved)**: We implemented a full-stack web application.
* **Backend:** Built with **FastAPI**. It handles image uploads, preprocesses them using the same transforms as training, runs inference, and returns JSON responses.
* **Frontend:** Built with HTML/CSS.
    * *Refactoring:* Migrated from inline CSS to external stylesheets for better maintainability.
    * *Responsive Design:* Optimized layout for different screen sizes.
* **Server:** Uses `Uvicorn` to serve the application.

---

## 🛠️ Technical Stack

* **Deep Learning:** PyTorch, Torchvision
* **Backend:** FastAPI, Uvicorn
* **Frontend:** HTML, CSS
* **Data Processing:** NumPy, Pillow (PIL)
* **Visualization:** Matplotlib, Tqdm

---

## 💻 How to Run the Project

1.  **Install Dependencies:**
    ```bash
    pip install torch torchvision fastapi uvicorn pillow
    ```

2.  **Directory Structure:**
    Ensure the `final_best_melanoma_model.pth` file is in the same directory as `Backend.py`.

3.  **Start the Server:**
    Run the backend script to start the FastAPI server.
    ```bash
    python Backend.py
    ```

4.  **Access the Application:**
    Open your browser and navigate to: `http://127.0.0.1:8000/`

---

## 📸 Screenshots

Here is a glimpse of our application in action:

### 1. Home Page
![Application Home Page](images/home_page.png)

### 2. Prediction Result (Malignant Case)
![Prediction Result](images/result_example.png)

### 3. Model Info
![Model Information](images/model_info.png)

### 4. Team Page
![Our Team](images/team_page.png)



