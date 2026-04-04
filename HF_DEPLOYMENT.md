# Deployment Guide: Hugging Face Spaces

This guide explains how to deploy your PD Measurement Backend to **Hugging Face Spaces** for faster, more reliable free hosting.

## 📋 Prerequisites
1.  A [Hugging Face](https://huggingface.co/) account.
2.  Your code pushed to the GitHub repository: `https://github.com/dodoankur/antigravity-pd-system`.

---

## 🛠️ Step-By-Step Deployment

### Step 1: Create a New Space
1.  Log in to [Hugging Face](https://huggingface.co/new-space).
2.  **Space Name**: `pd-api-backend` (or your preferred name).
3.  **SDK**: Select **Docker**.
4.  **Docker Template**: Choose **Blank**.
5.  **Visibility**: Public.
6.  Click **Create Space**.

### Step 2: Connect to GitHub
1.  Once the space is created, go to the **Settings** tab of your new Space.
2.  Scroll down to **Connected GitHub Repository**.
3.  Click **Connect a GitHub repository**.
4.  Authorize Hugging Face to access your GitHub account.
5.  Select your repository: `dodoankur/antigravity-pd-system`.
6.  **Branch**: `main`.

### Step 3: Configure Docker Path
> [!IMPORTANT]
> Since your project is a monorepo (it contains both frontend and backend), you MUST tell Hugging Face where the `Dockerfile` is.

1.  In the Space **Settings**, look for the **Docker context** or **Docker filepath**.
2.  Set the **Docker filepath** to: `backend/Dockerfile`.
3.  Set the **Docker Context** to: `backend/`.
4.  Click **Save Changes**.

### Step 4: Wait for Build
1.  Go to the **App** tab of your Space.
2.  Hugging Face will automatically start building your Docker image.
3.  Once the status turns to **Running**, your backend is live!
4.  Your API URL will be: `https://<your-username>-<space-name>.hf.space`. 
    *Example:* `https://dodoankur-pd-api-backend.hf.space`

---

## 🛠️ Update your Frontend (Vercel)

Once the Hugging Face Space is **Running**:

1.  Copy the public URL of your Space.
2.  Go to your **Vercel Dashboard**.
3.  Update the `VITE_API_BASE_URL` environment variable to your new Hugging Face URL.
4.  **Redeploy** the frontend in Vercel.

---

## ✅ Why is this better?
-   **No Cold Starts**: Hugging Face Spaces are persistent and don't "sleep" as aggressively as Render Free tier.
-   **More RAM/CPU**: Hugging Face provides more resources for shared CPU spaces, which is critical for OpenCV and MediaPipe processing.
-   **Faster Results**: Your PD measurements will process in 1-2 seconds instead of taking 10+ seconds on Render.
