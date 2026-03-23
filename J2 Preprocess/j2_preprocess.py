"""
audiolens — j2 image preprocessing

prepares a raw phone-captured document image for ocr.
each preprocessing step is a separate function so they can be
tested, tuned, or swapped out individually as needed.

pipeline order:
  1. to_grayscale    — converts colour input to grayscale
  2. deskew          — corrects tilt from phone capture angle
  3. denoise         — removes grain and compression artifacts
  4. enhance_contrast — applies clahe for local contrast improvement
  5. binarise        — converts to clean black/white via otsu threshold
  6. preprocess      — runs all steps in order (main entry point)

no downloads needed. import preprocess() directly into the pipeline.
"""

import numpy as np
import cv2


def to_grayscale(image):
    """
    converts a bgr colour image to grayscale.
    if image is already grayscale, returns a copy unchanged.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def deskew(gray):
    """
    detects and corrects the dominant tilt angle of the document.
    common when a user photographs a document at a slight angle.

    uses the minimum area bounding box of dark pixel clusters to
    estimate the skew angle, then rotates to correct it.
    angles under 0.5 degrees are ignored to avoid introducing
    unnecessary interpolation artifacts on near-straight images.
    """
    coords = np.column_stack(np.where(gray < 128))

    # not enough dark pixels to estimate angle reliably
    if len(coords) < 50:
        return gray

    angle = cv2.minAreaRect(coords)[-1]

    # minAreaRect returns angles in [-90, 0) — normalise to [-45, 45]
    if angle < -45:
        angle = 90 + angle

    # skip tiny corrections
    if abs(angle) < 0.5:
        return gray

    h, w    = gray.shape
    center  = (w // 2, h // 2)
    matrix  = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def denoise(gray):
    """
    removes noise, grain, and jpeg compression artifacts from the image.
    uses opencv's non-local means denoising which is effective on
    document scans and phone camera captures without blurring text edges.

    h=10 is a conservative strength — enough to clean grain but
    not so aggressive that it softens thin strokes in small text.
    """
    return cv2.fastNlMeansDenoising(gray, h=10)


def enhance_contrast(gray):
    """
    applies clahe (contrast limited adaptive histogram equalisation).
    unlike global histogram equalisation, clahe works on small tiles
    so it handles documents with uneven lighting — e.g. a shadow
    across part of a medicine label or a receipt photographed in dim light.

    cliplimit=2.0 prevents over-amplification of noise in flat regions.
    tileGridSize=(8, 8) gives a good balance between local and global correction.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def binarise(gray):
    """
    converts the grayscale image to a clean black and white binary image.
    uses otsu's method which automatically finds the optimal threshold
    value based on the image's intensity histogram — no manual tuning needed.

    binarisation removes any remaining grey tones and produces the
    high-contrast input that ocr models perform best on.
    """
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary


def preprocess(image):
    """
    runs the full preprocessing pipeline on a raw document image.
    this is the main entry point called from the audiolens pipeline.

    input:  numpy array — bgr colour or grayscale, any resolution
    output: numpy array — grayscale binarised image, same resolution

    pipeline: grayscale → deskew → denoise → enhance_contrast → binarise
    """
    image = to_grayscale(image)
    image = deskew(image)
    image = denoise(image)
    image = enhance_contrast(image)
    image = binarise(image)
    return image