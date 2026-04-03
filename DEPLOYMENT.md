# Project Deployment Guide: GitLab + Render + Vercel

Follow these steps to deploy your PD Measurement project for free.

## 1. Push Code to GitLab
If your code isn't on GitLab yet, follow these steps in your terminal:

```bash
# 1. Initialize git (if not already)
git init

# 2. Add all files
git add .

# 3. Commit your changes
git commit -m "Initial deployment commit"

# 4. Create a new project on GitLab.com, then copy the URL
# 5. Link your local repo (replace with your actual URL)
git remote add origin https://gitlab.com/YOUR_USERNAME/pd-measurement-project.git

# 6. Push the code
git push -u origin main
```

---

## 2. Deploy Backend (Render.com)
1.  Go to [Render Dashboard](https://dashboard.render.com).
2.  Click **New +** > **Web Service**.
3.  Select **GitLab** and search for your project.
4.  **Configuration**:
    - **Name**: `pd-api-backend`
    - **Region**: Choose one closest to you (e.g., Ohio or Singapore).
    - **Runtime**: Select **Docker** (Render will automatically find the `backend/Dockerfile`).
5.  **Environment Variables**: 
    - Render will automatically use the `PORT` from the Dockerfile.
6.  Click **Create Web Service**. 
    - *Note: The first build might take 5-7 minutes.*
7.  **Copy your Backend URL**: It will look like `https://pd-api-backend.onrender.com`.

---

## 3. Deploy Frontend (Vercel)
1.  Go to [Vercel Dashboard](https://vercel.com/dashboard).
2.  Click **Add New...** > **Project**.
3.  Select **GitLab** and import the same repository.
4.  **Configuration**:
    - **Framework Preset**: `Vite`
    - **Root Directory**: `frontend` (Click **Edit** and select the `frontend` folder).
    - **Build Command**: `npm run build`
    - **Output Directory**: `dist`
5.  **Environment Variables**:
    - Add a variable named `VITE_API_BASE_URL` and set its value to your **Render Backend URL** (e.g., `https://pd-api-backend.onrender.com`).
6.  Click **Deploy**.

---

## 4. Final Connection
Once both are deployed:
1.  Verify the backend by visiting `https://your-backend.onrender.com/api/health`. You should see `{"status":"healthy"}`.
2.  Your frontend (e.g., `https://your-app.vercel.app`) will now communicate with your live backend.

### 💡 Pro Tip: Customizing the Front End
If you want to change the API URL without redeploying, ensure your React code uses `import.meta.env.VITE_API_BASE_URL` as the default endpoint.

> [!NOTE]
> Render's **Free Tier** spins down after 15 minutes of inactivity. When you first open the app, it may take 60 seconds to "wake up" the backend.
