import numpy as np
import matplotlib.pyplot as plt


def plot_complex_plane(xs, ys):
    """Plot a complex→complex function using HSV color encoding for phase and magnitude.
    xs is a 2D grid (from np.meshgrid) of complex-plane x-values, ys is the complex output.
    NaN or inf values in ys are handled gracefully and shown as transparent.
    """

    # Mask invalid data
    valid_mask = np.isfinite(ys)
    if not np.any(valid_mask):
        raise ValueError("All ys values are NaN or inf — nothing to plot.")

    # Compute phase and magnitude safely
    phase = np.angle(np.where(valid_mask, ys, 0))
    mag = np.abs(np.where(valid_mask, ys, 0))
    max_mag = np.nanmax(mag)
    mag = mag / max_mag if max_mag != 0 else mag

    # HSV mapping
    hue = (phase + np.pi) / (2 * np.pi)
    value = mag

    # HSV → RGB
    rgb = plt.cm.hsv(hue)
    rgb[..., :3] *= value[..., None]

    # Add transparency for invalid values
    alpha = np.where(valid_mask, 1.0, 0.0)
    rgb[..., -1] = alpha

    # Compute plotting extents (robust to NaNs)
    x_real = np.real(xs)
    y_imag = np.imag(xs)
    x_min, x_max = np.nanmin(x_real), np.nanmax(x_real)
    y_min, y_max = np.nanmin(y_imag), np.nanmax(y_imag)

    # Plot
    plt.imshow(
        rgb,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        interpolation="nearest",
        aspect="equal",  # maintain correct aspect ratio
    )


def plot_complex(xs, ys):
    """
    Plot a real -> complex function
    """
    plt.plot(xs, ys.real, label="re")
    plt.plot(xs, ys.imag, label="im")



