import cv2
import numpy as np
from skimage.morphology import remove_small_objects


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


def recursive_otsu(image, levels=3, min_delta=3):
    """
    Vylepšená rekurzivní Otsu metoda:
    - Dynamické hledání optimálních prahů
    - Zajištění minimálního rozdílu mezi úrovněmi (min_delta)
    """
    thresholds = []

    def segment_region(region, depth=0):
        if depth >= levels or len(region) == 0:
            return
        t = otsu_threshold(region)
        if len(thresholds) == 0 or abs(t - thresholds[-1]) >= min_delta:
            thresholds.append(t)
            segment_region(region[region >= t], depth + 1)

    segment_region(image.flatten())
    return sorted(thresholds)
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

def estimate_background(image, window_size=21):
    """Odhad pozadí pomocí mediánového filtru"""
    return cv2.medianBlur(image, window_size)

def remove_background(image, background):
    """Odstranění pozadí odečtením od odhadu"""
    return np.clip(background - image, 0, 255).astype(np.uint8)

def bilateral_filter(image, sigma_color=2, sigma_space=10):
    """Redukce šumu pomocí bilateral filtru"""
    return cv2.bilateralFilter(image, -1, sigma_color, sigma_space)

def selective_bilateral_filter(image, mask, fg_params=(2, 2), bg_params=(10, 3)):
    """Selektivní vyhlazení podle počáteční segmentace"""
    foreground = cv2.bilateralFilter(image, -1, fg_params[0], fg_params[1])
    background = cv2.bilateralFilter(image, -1, bg_params[0], bg_params[1])
    return np.where(mask > 0, foreground, background)

def despeckle(binary_image, min_size=5):
    """Odstranění malých komponent"""
    cleaned = remove_small_objects(binary_image.astype(bool), min_size=min_size)
    return cleaned.astype(np.uint8) * 255

def invert_image(image):
    """Inverze intenzit obrázku: převrátí černobílý obraz (text bude černý)"""
    return 255 - image

def apply_recursive_otsu_advanced(image):
    """
    Kompletní pipeline pokročilého rekurzivního Otsu:
    1. Odhad pozadí
    2. Odstranění pozadí
    3. Bilaterální filtr
    4. Rekurzivní Otsu
    5. Selektivní bilateral filtr
    6. Finální prahování a despeckling
    """
    # 1. Odhad pozadí
    background = estimate_background(image)
    # 2. Odstranění pozadí
    no_bg = remove_background(image, background)
    # 3. Počáteční bilateral filtr
    filtered = bilateral_filter(no_bg)
    # 4. Rekurzivní Otsu
    thresholds = recursive_otsu(filtered)
    if not thresholds:
        thresholds = [otsu_threshold(filtered)]
    binary = np.zeros_like(filtered)
    for thresh in thresholds:
        binary = np.maximum(binary, (filtered < thresh).astype(np.uint8) * 255)
    # 5. Selektivní bilateral filtr
    selective_filtered = selective_bilateral_filter(filtered, binary)
    # 6. Finální rekurzivní Otsu a despeckling
    final_thresholds = recursive_otsu(selective_filtered)
    final_binary = np.zeros_like(selective_filtered)
    for thresh in final_thresholds:
        final_binary = np.maximum(final_binary, (selective_filtered < thresh).astype(np.uint8) * 255)
    final_result = despeckle(final_binary)
    # Invertování výsledku, pokud je třeba (aby byl text černý)
    return invert_image(final_result)


import cv2
import numpy as np
from skimage.morphology import remove_small_objects


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


def recursive_otsu(image, levels=3, min_delta=3):
    """
    Vylepšená rekurzivní Otsu metoda:
    - Dynamické hledání optimálních prahů
    - Zajištění minimálního rozdílu mezi úrovněmi (min_delta)
    """
    thresholds = []

    def segment_region(region, depth=0):
        if depth >= levels or len(region) == 0:
            return
        t = otsu_threshold(region)
        if len(thresholds) == 0 or abs(t - thresholds[-1]) >= min_delta:
            thresholds.append(t)
            segment_region(region[region >= t], depth + 1)

    segment_region(image.flatten())
    return sorted(thresholds)


def apply_recursive_otsu_simple(image, num_classes=3):
    """
    Aplikuje rekurzivní Otsu pro segmentaci obrazu.
    """
    thresholds = recursive_otsu(image, levels=num_classes)

    # Ověření správného počtu prahů
    while len(thresholds) < num_classes - 1:
        thresholds.append(otsu_threshold(image))
        thresholds = sorted(set(thresholds))  # Odstranění duplicit

    segmented = np.digitize(image, bins=thresholds) * (255 // num_classes)

    # Invertujeme výsledek, aby text zůstal černý a pozadí bílé
    return 255 - segmented.astype(np.uint8)
