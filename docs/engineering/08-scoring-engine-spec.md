# 08. Scoring Engine Specification

## Algorithm Logic
1. **Input:** High-res image + coarse landmarks.
2. **Preprocessing:** Crop to eye regions with 20% padding.
3. **Canny Edge Detection / Hough Transform:** Identify the iris-sclera boundary.
4. **Circular Fit:** Use RANSAC to fit a circle to the iris boundary, ignoring eyelid occlusions.
5. **Distance Calculation:**
   - Pixel distance between circle centers ($P_d$).
   - Average iris radius in pixels ($I_r$).
   - $PD = (P_d / (2 * I_r)) * 11.7$ (assuming 11.7mm HVID).
6. **Refinement:** Use a lightweight CNN to predict pupil centers within the detected iris.
