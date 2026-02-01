import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

from integrator import ComplexIntegrationResult


def plot_complex_plane(xs, ys, ax=None, cmap_factor=1):
    """
    Plot a complex -> complex function using HSV color encoding for phase and magnitude.
    xs is a 2D grid (from np.meshgrid) of complex-plane x-values, ys is the complex output.
    NaN or inf values in ys are shown as transparent.
    """

    if ax is None:
        ax = plt.gca()

    # Mask invalid data
    valid_mask = np.isfinite(ys)
    if not np.any(valid_mask):
        raise ValueError("All ys values are NaN or inf — nothing to plot.")

    # Compute phase and magnitude safely
    phase = np.angle(np.where(valid_mask, ys, 0))

    # HSV mapping
    hue = (phase + np.pi) / (2 * np.pi)
    value = np.abs(np.where(valid_mask, ys, 0))

    
    value = value / (value.max() + 1e-10)
    value = np.log(cmap_factor * value + 1)
    value = value / (value.max() + 1e-10)

    # HSV → RGB
    rgb = plt.cm.hsv(hue) # type: ignore
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
        extent=[x_min, x_max, y_min, y_max], # type: ignore
        interpolation="nearest",
        aspect="equal",  # maintain correct aspect ratio
    )


def plot_complex(xs, ys):
    """
    Plot a real -> complex function
    """
    plt.plot(xs, ys.real, label="re")
    plt.plot(xs, ys.imag, label="im")


def get_contour(x_lim, y_lim, z_lim, f, res=100):
    """
    Get a pyvista contour of a 3D function
    """

    x = np.linspace(x_lim[0], x_lim[1], res)
    y = np.linspace(y_lim[0], y_lim[1], res)
    z = np.linspace(z_lim[0], z_lim[1], res)
    xs = np.stack(np.meshgrid(x, y, z), axis=-1)
    vals = f(xs)
    grid = pv.ImageData() # type: ignore
    grid.dimensions = np.array(vals.shape) # type: ignore
    grid.origin = (x[0], y[0], z[0])
    grid.spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    grid.point_data["vals"] = vals.flatten(order="F")
    return grid.contour([0]) # type: ignore


def plot_complex_integration_with_ref(
    res: list[ComplexIntegrationResult],
    ref_same_x: np.typing.ArrayLike,
    x_values,
    ref: np.typing.ArrayLike,
    ref_x_values
):
    """
    Plots the real and imaginary parts of a complex integration result on a single plot
    """
    real_avg = np.array([r.real_avg for r in res])
    real_err = np.array([r.real_err for r in res])
    imag_avg = np.array([r.imag_avg for r in res])
    imag_err = np.array([r.imag_err for r in res])

    fig, axs = plt.subplots(
        4, 1,
        sharex=True,
        figsize=(10, 10),
        gridspec_kw={"hspace": 0},
        height_ratios=[2, 1, 2, 1]
    )
    

    #### Real part ####
    axs[0].plot(ref_x_values, ref.real, label="Reference", c = 'k') # type: ignore
    axs[0].errorbar(
        x_values,
        real_avg,
        real_err,
        fmt="o",
        capsize=3,
        label="Numerical"
    )
    axs[0].set_ylabel("Re(value)")
    axs[0].legend()

    axs[1].errorbar(
        x_values,
        100 * (real_avg - ref_same_x.real) / (ref_same_x.real+1e-10), # type: ignore
        100 * real_err / np.abs(ref_same_x.real+1e-10), # type: ignore
        fmt="o",
        capsize=3,
        label="Deviation"
    )
    axs[1].axhline(0, ls = "--", c = "black")
    axs[1].set_ylabel("Re(Deviation) [%]")

    #### Imaginary part ###
    axs[2].plot(ref_x_values, ref.imag, label="Reference", c = 'k') # type: ignore
    axs[2].errorbar(
        x_values,
        imag_avg,
        imag_err,
        fmt="o",
        capsize=3,
        label="Numerical"
    )
    axs[2].set_ylabel("Im(value)")
    axs[2].legend()

    axs[3].errorbar(
        x_values,
        100 * (imag_avg - ref_same_x.imag) / (ref_same_x.imag+1e-10), # type: ignore
        100 * imag_err / np.abs(ref_same_x.imag+1e-10), # type: ignore
        fmt="o",
        capsize=3,
        label="Deviation"
    )
    axs[3].axhline(0, ls = "--", c = "black")
    axs[3].set_ylabel("Im(Deviation) [%]")
    for ax in axs:
        ax.label_outer()
    fig.subplots_adjust(hspace=0)
    return fig