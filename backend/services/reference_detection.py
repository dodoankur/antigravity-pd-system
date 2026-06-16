"""
Reference Object Detection Service
Detects reference objects (credit card, GBP coins, ruler) for scale calibration
"""
import logging
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ReferenceObjectType(Enum):
    CREDIT_CARD = "credit_card"
    COIN_GBP_1P = "coin_gbp_1p"
    COIN_GBP_2P = "coin_gbp_2p"
    COIN_GBP_5P = "coin_gbp_5p"
    COIN_GBP_10P = "coin_gbp_10p"
    COIN_GBP_20P = "coin_gbp_20p"
    COIN_GBP_50P = "coin_gbp_50p"
    COIN_GBP_1 = "coin_gbp_1"
    COIN_GBP_2 = "coin_gbp_2"
    RULER = "ruler"


@dataclass
class ReferenceDetectionResult:
    """Result of reference object detection"""
    detected: bool
    object_type: ReferenceObjectType
    scale_factor: float  # mm per pixel
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h


# Real-world dimensions of reference objects in millimeters
REFERENCE_DIMENSIONS = {
    # ISO/IEC 7810 ID-1 standard credit card
    ReferenceObjectType.CREDIT_CARD: {
        "width": 85.6,
        "height": 53.98,
        "description": "Standard credit/debit card"
    },
    # UK coin dimensions (diameter in mm)
    ReferenceObjectType.COIN_GBP_1P: {"diameter": 20.3, "description": "1 penny"},
    ReferenceObjectType.COIN_GBP_2P: {"diameter": 25.9, "description": "2 pence"},
    ReferenceObjectType.COIN_GBP_5P: {"diameter": 18.0, "description": "5 pence"},
    ReferenceObjectType.COIN_GBP_10P: {"diameter": 24.5, "description": "10 pence"},
    ReferenceObjectType.COIN_GBP_20P: {"diameter": 21.4, "description": "20 pence"},
    ReferenceObjectType.COIN_GBP_50P: {"diameter": 27.3, "description": "50 pence (7-sided)"},
    ReferenceObjectType.COIN_GBP_1: {"diameter": 23.43, "description": "£1 coin"},
    ReferenceObjectType.COIN_GBP_2: {"diameter": 28.4, "description": "£2 coin"},
    # Ruler (10mm segment assumed)
    ReferenceObjectType.RULER: {"segment": 10.0, "description": "Ruler (10mm segments)"},
}


class ReferenceDetectionService:
    """
    Service for detecting reference objects and calculating scale factors
    
    Supported objects:
    - Credit cards (rectangle detection)
    - GBP coins (circle detection)
    - Rulers (line/segment detection)
    """
    
    def __init__(self):
        """Initialize detection parameters"""
        self.min_contour_area = 1000  # Minimum contour area to consider
        self.card_aspect_ratio = 85.6 / 53.98  # ~1.586
        # T2-5: tightened from 0.15 (15%) to 0.06 (6%).
        # 15% tolerance accepted aspect ratios 1.348–1.823, which allowed
        # false positives from folded receipts, book corners, door frames, etc.
        # 6% → accepted range 1.491–1.681, much more selective while still
        # covering slight card angle / perspective distortion in real photos.
        self.card_aspect_tolerance = 0.06
    
    def detect_credit_card(self, image: np.ndarray) -> Optional[ReferenceDetectionResult]:
        """
        Detect credit card in image using contour detection
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            ReferenceDetectionResult or None if not detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_match = None
        best_confidence = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Credit cards have 4 corners
            if len(approx) == 4:
                # Get bounding rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = box.astype(np.int32)
                
                # Calculate aspect ratio
                width = rect[1][0]
                height = rect[1][1]
                if height == 0:
                    continue
                    
                aspect_ratio = max(width, height) / min(width, height)
                
                # Check if aspect ratio matches credit card
                ratio_diff = abs(aspect_ratio - self.card_aspect_ratio)
                if ratio_diff < self.card_aspect_ratio * self.card_aspect_tolerance:
                    # Calculate confidence based on aspect ratio match
                    confidence = 1.0 - (ratio_diff / self.card_aspect_ratio)
                    
                    if confidence > best_confidence:
                        # Calculate scale factor (mm per pixel)
                        longer_side_px = max(width, height)
                        scale_factor = REFERENCE_DIMENSIONS[ReferenceObjectType.CREDIT_CARD]["width"] / longer_side_px
                        
                        x, y, w, h = cv2.boundingRect(contour)
                        best_match = ReferenceDetectionResult(
                            detected=True,
                            object_type=ReferenceObjectType.CREDIT_CARD,
                            scale_factor=scale_factor,
                            confidence=round(confidence, 3),
                            bounding_box=(x, y, w, h)
                        )
                        best_confidence = confidence
        
        return best_match
    
    def detect_coin(
        self,
        image: np.ndarray,
        coin_type: ReferenceObjectType,
    ) -> Optional[ReferenceDetectionResult]:
        """
        Detect a coin in image using circle detection (Hough Transform).

        Radius bounds are computed dynamically from the image dimensions so the
        detector works correctly across a wide range of resolutions (from a low-res
        720p crop to a full 20MP phone shot).

        Args:
            image: BGR image as numpy array
            coin_type: Type of coin to detect

        Returns:
            ReferenceDetectionResult or None if not detected
        """
        if coin_type not in REFERENCE_DIMENSIONS or "diameter" not in REFERENCE_DIMENSIONS[coin_type]:
            return None

        height, width = image.shape[:2]
        shorter_side = min(height, width)

        # A coin held at arm's length next to a face typically subtends roughly
        # 3%–25% of the shorter image dimension.  These bounds are intentionally
        # generous to handle a range of distances and phone orientations.
        min_radius = max(10, int(shorter_side * 0.03))
        max_radius = max(min_radius + 10, int(shorter_side * 0.25))

        logger.debug(
            "Coin detection: image=%dx%d  radius range=[%d, %d]",
            width, height, min_radius, max_radius,
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles using Hough Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(shorter_side * 0.05),   # also dynamic: ~5% of shorter side
            param1=100,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        
        if circles is None:
            return None
        
        circles = np.uint16(np.around(circles))
        
        # Find the most prominent circle (largest area that fits in image)
        height, width = image.shape[:2]
        best_circle = None
        best_score = 0
        
        for circle in circles[0, :]:
            x, y, r = circle
            
            # Check if circle is fully within image bounds
            if x - r >= 0 and x + r < width and y - r >= 0 and y + r < height:
                # Score based on size and centrality
                size_score = r / max(height, width)
                center_score = 1 - (abs(x - width/2) + abs(y - height/2)) / (width + height)
                score = size_score * 0.5 + center_score * 0.5
                
                if score > best_score:
                    best_circle = circle
                    best_score = score
        
        if best_circle is None:
            return None

        x, y, r = best_circle

        # --- Sub-pixel circle refinement ---
        # Re-run Canny on a tight ROI around the detected circle and fit
        # an ellipse to edge pixels for sub-pixel radius accuracy.
        refined_r = self._subpixel_circle_radius(gray, int(x), int(y), int(r))
        if refined_r > 0:
            logger.debug(
                "Sub-pixel refinement: Hough r=%d → refined r=%.2f (Δ=%.2f px)",
                r, refined_r, refined_r - r,
            )
            r = refined_r

        diameter_px  = 2.0 * r
        diameter_mm  = REFERENCE_DIMENSIONS[coin_type]["diameter"]
        scale_factor = diameter_mm / diameter_px

        confidence = min(0.95, best_score + 0.5)

        return ReferenceDetectionResult(
            detected=True,
            object_type=coin_type,
            scale_factor=scale_factor,
            confidence=round(confidence, 3),
            bounding_box=(int(x - r), int(y - r), int(2 * r), int(2 * r)),
        )
    
    def _subpixel_circle_radius(
        self, gray: np.ndarray, cx: int, cy: int, r: int, margin: int = 6
    ) -> float:
        """
        Refine a Hough-detected circle radius to sub-pixel accuracy.

        Strategy:
          1. Crop a tight ROI around the circle.
          2. Apply Canny edge detection.
          3. Collect edge pixels whose distance from the centre falls within
             [r - margin, r + margin].
          4. Return the median of those distances as the refined radius.
             Falls back to 0.0 (i.e. keep original) if fewer than 8 edge
             pixels are found.
        """
        h, w = gray.shape[:2]
        # Define ROI bounds (clamped to image)
        pad  = r + margin + 4
        x0   = max(0, cx - pad)
        y0   = max(0, cy - pad)
        x1   = min(w, cx + pad)
        y1   = min(h, cy + pad)
        roi  = gray[y0:y1, x0:x1]
        if roi.size == 0:
            return 0.0

        # Canny on ROI
        blurred = cv2.GaussianBlur(roi, (5, 5), 1)
        edges   = cv2.Canny(blurred, 50, 120)

        # Collect edge-pixel distances from circle centre (in ROI coords)
        cx_roi = cx - x0
        cy_roi = cy - y0
        ys, xs = np.where(edges.astype(np.uint8) > 0)
        if len(xs) == 0:
            return 0.0

        dists = np.sqrt((xs - cx_roi).astype(float) ** 2 + (ys - cy_roi).astype(float) ** 2)
        # Keep only those near the expected radius
        mask  = (dists >= r - margin) & (dists <= r + margin)
        near  = dists[mask]
        if len(near) < 8:
            return 0.0

        return float(np.median(near))

    def detect_ruler(self, image: np.ndarray) -> Optional[ReferenceDetectionResult]:
        """
        Detect ruler markings in image
        
        Looks for evenly spaced vertical lines (ruler tick marks)
        Assumes 10mm spacing between major ticks
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            ReferenceDetectionResult or None if not detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=20,
            maxLineGap=5
        )
        
        if lines is None or len(lines) < 3:
            return None
        
        # Find vertical lines (ruler tick marks)
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Check if line is roughly vertical (within 20 degrees)
            if abs(abs(angle) - 90) < 20:
                vertical_lines.append((x1, y1, x2, y2))
        
        if len(vertical_lines) < 3:
            return None
        
        # Sort lines by x position
        vertical_lines.sort(key=lambda l: (l[0] + l[2]) / 2)
        
        # Calculate spacing between adjacent lines
        spacings = []
        for i in range(1, len(vertical_lines)):
            x1 = (vertical_lines[i-1][0] + vertical_lines[i-1][2]) / 2
            x2 = (vertical_lines[i][0] + vertical_lines[i][2]) / 2
            spacings.append(abs(x2 - x1))
        
        if not spacings:
            return None
        
        # Find most common spacing (likely 10mm marks)
        avg_spacing = np.median(spacings)
        
        # Assume 10mm segment
        scale_factor = REFERENCE_DIMENSIONS[ReferenceObjectType.RULER]["segment"] / avg_spacing
        
        # Calculate confidence based on spacing consistency
        spacing_std = np.std(spacings)
        consistency = 1 - min(1, spacing_std / avg_spacing)
        confidence = consistency * 0.8  # Max 80% confidence for ruler
        
        return ReferenceDetectionResult(
            detected=True,
            object_type=ReferenceObjectType.RULER,
            scale_factor=scale_factor,
            confidence=round(confidence, 3),
            bounding_box=None
        )
    
    def detect(
        self, 
        image: np.ndarray, 
        reference_type: str
    ) -> Optional[ReferenceDetectionResult]:
        """
        Detect specified reference object type
        
        Args:
            image: BGR image as numpy array
            reference_type: Type of reference to detect
            
        Returns:
            ReferenceDetectionResult or None
        """
        try:
            ref_type = ReferenceObjectType(reference_type)
        except ValueError:
            return None
        
        if ref_type == ReferenceObjectType.CREDIT_CARD:
            return self.detect_credit_card(image)
        elif ref_type == ReferenceObjectType.RULER:
            return self.detect_ruler(image)
        elif ref_type.value.startswith("coin_gbp"):
            return self.detect_coin(image, ref_type)
        
        return None
