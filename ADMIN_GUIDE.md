# Admin Panel Guide

This guide explains how to use the Admin Panel to manage your Face Recognition system.

## Accessing the Admin Panel
1.  Open your browser and go to your application URL (e.g., `http://localhost:5000`).
2.  Click the **"Admin Panel"** button in the top right corner.
3.  Alternatively, go directly to `/admin` (e.g., `http://localhost:5000/admin`).

## Features & Controls

### 1. Service Control (Start/Stop)
At the top of the Admin Panel, you will see the **Service Status** indicator.
*   **Active (Green)**: The system is checking faces normally.
*   **Stopped (Red)**: The system is paused. No recognition calls will be processed.
*   **Toggle Button**: Click **"Stop Service"** to pause recognition (useful for maintenance or saving resources). Click **"Start Service"** to resume.

### 2. Adding a New Face
To register a new person:
1.  Click the **"+ Add New Face"** button.
2.  **Select Photo**: Choose a clear, front-facing photo of the person (JPG or PNG).
    *   *Tip: Ensure good lighting and that the face takes up most of the image.*
3.  **Enter Name**: Type the name of the person (e.g., "John Doe").
4.  Click **"Save Face"**.
5.  Expected result: The page will reload, and the new person will appear in the grid.

### 3. Deleting a Face
To remove a person:
1.  Find their card in the grid.
2.  Click the red **"Delete"** button.
3.  Confirm the action in the popup.
4.  Expected result: The person is removed from the system immediately.

## Troubleshooting
*   **"Service Paused" on Live View**: This means an admin has stopped the service. Go to the Admin Panel and click "Start Service".
*   **Upload Failed**: unique names are required. Ensure the image is less than 16MB.
*   **Stuck on "Processing..."**: If the button gets stuck, refresh the page.
