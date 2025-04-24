import cv2
import numpy as np


def otsu_threshold(image):
    """Vypočítá optimální práh metodou Otsu."""
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    hist = hist.astype(float)
    prob = hist / hist.sum()

    max_var = 0
    optimal_thresh = 0

    for t in range(1, 256):
        w0 = prob[:t].sum()
        w1 = prob[t:].sum()
        if w0 == 0 or w1 == 0:
            continue
        mu0 = (np.arange(t) * prob[:t]).sum() / w0
        mu1 = (np.arange(t, 256) * prob[t:]).sum() / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > max_var:
            max_var = var
            optimal_thresh = t

    return optimal_thresh


def recursive_otsu(image, levels=3, min_delta=3, min_val=0, max_val=255):
    thresholds = []

    def segment_region(region, depth=0):
        if depth >= levels or len(region) < 10:
            return
        t = otsu_threshold(region)
        if t < min_val or t > max_val:
            return
        if not thresholds or all(abs(t - prev_t) >= min_delta for prev_t in thresholds):
            thresholds.append(t)
            segment_region(region[region < t], depth + 1)
            segment_region(region[region >= t], depth + 1)

    segment_region(image.flatten())
    return sorted(set(thresholds))


def sauvola_threshold(image, window_size=15, k=0.2, r=128):
    """Implementace Sauvolaho adaptivního prahování"""
    if window_size % 2 == 0:
        window_size += 1

    pad = window_size // 2
    padded = np.pad(image, pad, mode='reflect')
    threshold_map = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i + window_size, j:j + window_size]
            mean = window.mean()
            std = window.std()
            threshold_map[i, j] = mean * (1 + k * ((std / r) - 1))

    binary = (image > threshold_map).astype(np.uint8) * 255
    return binary


def otsu_recursive_otsu_gui(img_gray, bg_est_window=21, bilateral_r=2, bilateral_s=10,
                            d1=2, d2=26, bg_bilateral_r=3, bg_bilateral_s=10,
                            text_bilateral_r=2, text_bilateral_s=2):
    background = cv2.medianBlur(img_gray, bg_est_window)
    background = cv2.bilateralFilter(background, -1, bg_bilateral_r, bg_bilateral_s)

    no_bg = cv2.subtract(background, img_gray)

    filtered = cv2.bilateralFilter(no_bg, -1, bilateral_r, bilateral_s)
    enhanced = cv2.bilateralFilter(filtered, -1, text_bilateral_r, text_bilateral_s)

    thresholds = recursive_otsu(enhanced, levels=3, min_val=d1, max_val=d2)

    if thresholds:
        best_thresh = thresholds[len(thresholds)//2]
    else:
        best_thresh = otsu_threshold(enhanced)

    _, binary_final = cv2.threshold(enhanced, best_thresh, 255, cv2.THRESH_BINARY)


    binary_final = cv2.bitwise_not(binary_final)

    return binary_final

