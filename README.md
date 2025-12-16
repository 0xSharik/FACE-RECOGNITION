# Face Recognition System

A high-accuracy, real-time face recognition system using DeepFace (ArcFace model), OpenCV, and Flask.

## Features
*   **Real-time Recognition**: Detects and identifies faces from the webcam stream.
*   **High Accuracy**: Uses the state-of-the-art **ArcFace** model.
*   **Web Interface**: View the live feed and recognition status in your browser.
*   **Smart Caching**: Caches face encodings for faster startup.

## Project Structure
*   `face.py`: The main application server.
*   `templates/index.html`: The frontend user interface.
*   `known_faces/`: Folder containing images of people to recognize.
*   `_archive/`: Folder containing old/unused files.

## Installation

1.  **Prerequisites**:
    *   Python 3.10+
    *   Webcam

2.  **Setup**:
    ```powershell
    # 1. Create virtual environment
    python -m venv venv
    
    # 2. Activate it
    .\venv\Scripts\Activate
    
    # 3. Install dependencies
    pip install -r requirements.txt
    pip install deepface tf-keras
    ```

## 🚀 Deployment on Railway

This project is optimized for Railway.app.

### 1. Environment Variables
Go to **Settings > Variables** and add:
- `PORT`: `5000` (Railway sets this automatically, but good to know)
- `UPLOAD_FOLDER`: `/data/known_faces` (CRITICAL: Must match Volume path)
- `ADMIN_TOKEN`: `your-secure-password` (Required for Admin Panel actions)

### 2. Add Persistent Storage (Volume)
**Crucial Step:** To prevent losing faces when the server restarts:
1.  Go to your Service **Settings > Volumes**.
2.  Click **Add Volume**.
3.  Mount Path: `/data`

### 3. Start Command
The project includes a `Procfile` that automatically runs:
`gunicorn face:app --workers 1 --threads 4 --timeout 120`

### 4. Deploying (Two Options)

#### Option A: GitHub (Recommended)
1. Push your code to a GitHub repository.
2. In Railway, click **New Project > Deploy from GitHub repo**.
3. Select your repo. Railway will auto-build.

#### Option B: Railway CLI (No GitHub)
If you don't want to push your code to GitHub:
1. **Install CLI**: `npm i -g @railway/cli`
2. **Login**: `railway login`
3. **Link Project**: `railway link` (Select your empty project)
4. **Deploy**: `railway up` 

*Note: This respects `.gitignore`, so `known_faces/` and `venv/` will NOT be uploaded.*

## Usage

1.  **Add Known Faces**:
    *   Put photos of people you want to recognize in the `known_faces/` folder.
    *   Name them like `John Doe.jpg` or `Jane.png`.

2.  **Run the Server**:
    ```powershell
    python face.py
    ```
    *   *Note: The first run may take a moment to download the ArcFace model weights.*

3.  **Open the Interface**:
    *   Go to: **http://localhost:5000**

## Troubleshooting
*   **Camera Inactive**: Ensure no other app is using the webcam. Check the terminal logs for "Failed to open camera" errors.
*   **Unknown Face**: Add high-quality, front-facing photos to `known_faces/` and restart the server.
