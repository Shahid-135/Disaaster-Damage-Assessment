import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# Directory containing input images
output_dir = "gradcam_outputs"

# Image identifiers and their corresponding damage labels
image_info = {
    "nodamage1": "No Damage",
    "mild2": "Mild Damage",
    "severe1": "Severe Damage",
    "severe2": "Severe Damage"
}

columns = ["Original", "Gaussian", "Salt & Pepper", "Speckle", "Poisson"]

# ---------------------------------------------------------------------------
# Noise functions
# ---------------------------------------------------------------------------

def add_gaussian_noise(image, mean=0, std=60):
    """Apply high-variance Gaussian noise."""
    gauss = np.random.normal(mean, std, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + gauss, 0, 255).astype(np.uint8)
    return noisy

def add_salt_and_pepper_noise(image, amount=0.15):
    """Apply heavy salt-and-pepper noise."""
    output = image.copy()
    h, w, _ = image.shape
    num_salt = int(amount * h * w / 2)
    num_pepper = int(amount * h * w / 2)

    # Salt
    ys = np.random.randint(0, h, num_salt)
    xs = np.random.randint(0, w, num_salt)
    output[ys, xs] = 255

    # Pepper
    ys = np.random.randint(0, h, num_pepper)
    xs = np.random.randint(0, w, num_pepper)
    output[ys, xs] = 0

    return output

def add_speckle_noise(image, intensity=0.7):
    """Apply high-intensity speckle noise."""
    noise = np.random.randn(*image.shape) * intensity
    noisy = image + image * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_poisson_noise(image, severity=30):
    """Apply amplified Poisson noise."""
    img_f = image.astype(np.float32) / 255.0
    scaled = img_f * severity
    noisy = np.random.poisson(scaled) / severity
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 18))
fig.suptitle("Heavy Noise Effects on Original Images", fontsize=20, fontweight="bold")

for row_idx, (img_name, damage_label) in enumerate(image_info.items()):
    img_path = os.path.join(output_dir, f"{img_name}_orig.jpg")
    orig = cv2.imread(img_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    noisy_versions = [
        orig,
        add_gaussian_noise(orig),
        add_salt_and_pepper_noise(orig),
        add_speckle_noise(orig),
        add_poisson_noise(orig)
    ]

    for col_idx, img in enumerate(noisy_versions):
        ax = axes[row_idx, col_idx]
        ax.imshow(img)
        ax.axis("off")

        if row_idx == 0:
            ax.set_title(columns[col_idx], fontsize=14, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel(damage_label, fontsize=14, fontweight="bold")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("heavily_noised_images.png", dpi=300)
plt.show() 
