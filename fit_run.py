from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import constants as const
import read_raw_to_cvs
import xrd_math_models


@dataclass(frozen=True)
class PeakFitResult:
    file_path: str
    fit_bounds: tuple[float, float]
    success: bool
    message: str
    amplitude: float
    center_two_theta_deg: float
    fwhm_deg: float
    eta: float
    background_slope: float
    background_offset: float
    rmse: float
    x: np.ndarray
    y: np.ndarray
    y_fit: np.ndarray
    y_ka1: np.ndarray
    y_ka2: np.ndarray
    residuals: np.ndarray


def load_csv(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    angle, intensity = np.loadtxt(file_path, delimiter=",", unpack=True, skiprows=1)
    return angle, intensity


def preview_data(x: np.ndarray, y: np.ndarray, filename: str = "Data Preview") -> None:
    print(f"Showing preview for: {filename}")
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=2, color="black", alpha=0.6, label="Raw Data")
    plt.title(f"Preview: {filename}")
    plt.xlabel("2Theta [deg]")
    plt.ylabel("Intensity [counts]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show(block=True)


def determine_roi_from_header(csv_file_path: str, window: float) -> tuple[float, float, float]:
    raw_file_path = str(Path(csv_file_path).with_suffix(".raw"))
    _, tths_header, _ = read_raw_to_cvs.extract_pixel_data_and_metadata_from_raw(raw_file_path)
    fit_window_min = tths_header - window
    fit_window_max = tths_header + window
    return fit_window_min, fit_window_max, tths_header


def _build_initial_guess(x: np.ndarray, y: np.ndarray, scan_center: float) -> tuple[list[float], list[tuple[float, float]]]:
    dx = float(np.median(np.diff(x)))
    background_guess = float(np.percentile(y, 10))
    amplitude_guess = max(float(np.max(y) - background_guess), 1.0)
    fwhm_guess = max(0.2, 8.0 * dx)
    fwhm_min = max(3.0 * dx, np.finfo(float).eps)

    initial_guess = [
        amplitude_guess,
        scan_center,
        fwhm_guess,
        0.5,
        0.0,
        background_guess,
    ]

    bounds = [
        (0.0, max(amplitude_guess * 3.0, amplitude_guess + 1.0)),
        (float(np.min(x)), float(np.max(x))),
        (fwhm_min, 2.0),
        (0.0, 1.0),
        (-np.inf, np.inf),
        (0.0, np.inf),
    ]
    return initial_guess, bounds


def fit_peak_file(
    csv_file_path: str,
    window: float = 1.0,
    lam_k1: float = const.LAM_K1,
    lam_k2: float = const.LAM_K2,
    ratio: float = const.INTENSITY_RATIO,
    preview: bool = False,
) -> PeakFitResult:
    angle, intensity = load_csv(csv_file_path)

    if preview:
        preview_data(angle, intensity, filename=Path(csv_file_path).name)

    fit_window_min, fit_window_max, tths_header = determine_roi_from_header(csv_file_path, window=window)
    mask = (angle >= fit_window_min) & (angle <= fit_window_max)
    x = angle[mask]
    y = intensity[mask]

    if len(x) == 0:
        raise ValueError(
            f"No data points found in fit window [{fit_window_min:.3f}, {fit_window_max:.3f}] "
            f"for file {csv_file_path}."
        )

    initial_guess, bounds = _build_initial_guess(x, y, scan_center=tths_header)
    optimization = xrd_math_models.fit_doublet_peaks(
        x,
        y,
        lam_k1,
        lam_k2,
        ratio,
        initial_guess,
        bounds,
    )
    amplitude, x01, fwhm, eta, slope, offset = optimization.x
    y_fit = xrd_math_models.doublet_model_f(optimization.x, x, lam_k1, lam_k2, ratio)
    residuals = y - y_fit

    theta1 = np.deg2rad(x01 / 2.0)
    theta2 = np.arcsin(np.sin(theta1) * (lam_k2 / lam_k1))
    x02 = np.rad2deg(2.0 * theta2)

    background = xrd_math_models.background_f(x, slope, offset)
    y_ka1 = xrd_math_models.pseudo_voigt_f(x, amplitude, x01, fwhm, eta) + background
    y_ka2 = xrd_math_models.pseudo_voigt_f(x, amplitude * ratio, x02, fwhm, eta) + background
    rmse = float(np.sqrt(np.mean(residuals**2)))

    return PeakFitResult(
        file_path=csv_file_path,
        fit_bounds=(fit_window_min, fit_window_max),
        success=bool(optimization.success),
        message=str(optimization.message),
        amplitude=float(amplitude),
        center_two_theta_deg=float(x01),
        fwhm_deg=float(fwhm),
        eta=float(eta),
        background_slope=float(slope),
        background_offset=float(offset),
        rmse=rmse,
        x=x,
        y=y,
        y_fit=y_fit,
        y_ka1=y_ka1,
        y_ka2=y_ka2,
        residuals=residuals,
    )


def get_peak_position(csv_file_path: str, window: float = 1.0) -> float | None:
    result = fit_peak_file(csv_file_path, window=window)
    if not result.success:
        return None
    return result.center_two_theta_deg


def plot_fit_results(result: PeakFitResult, block: bool = True) -> None:
    _, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}, figsize=(8, 6))
    ax_fit, ax_res = ax

    ax_fit.scatter(result.x, result.y, s=15, color="black", alpha=0.6, label="ROI data")
    ax_fit.plot(result.x, result.y_fit, color="red", linewidth=2, label="Doublet fit")
    ax_fit.plot(result.x, result.y_ka1, color="green", linestyle="--", linewidth=1, alpha=0.7, label="Kalpha1")
    ax_fit.plot(result.x, result.y_ka2, color="blue", linestyle="--", linewidth=1, alpha=0.7, label="Kalpha2")
    ax_fit.set_ylabel("Intensity [counts]")
    ax_fit.legend()
    ax_fit.grid(True, alpha=0.3)
    ax_fit.set_title(
        f"Fit result: x0={result.center_two_theta_deg:.4f} deg, "
        f"FWHM={result.fwhm_deg:.4f} deg, RMSE={result.rmse:.4f}"
    )

    ax_res.scatter(result.x, result.residuals, s=10, color="blue", alpha=0.6)
    ax_res.axhline(0.0, color="black", linestyle="-", linewidth=1)
    ax_res.set_ylabel("Residuals")
    ax_res.set_xlabel("2Theta [deg]")
    ax_res.grid(True, alpha=0.3)

    fit_min, fit_max = result.fit_bounds
    plt.xlim(fit_min - 0.2, fit_max + 0.2)
    plt.subplots_adjust(hspace=0.05)
    plt.show(block=block)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a Kalpha doublet in a single CSV scan.")
    parser.add_argument("csv_file", help="Path to the converted CSV scan.")
    parser.add_argument("--window", type=float, default=1.0, help="Half-width of the ROI around the header center.")
    parser.add_argument("--preview", action="store_true", help="Show the raw scan before fitting.")
    parser.add_argument("--plot", action="store_true", help="Plot the final fit and residuals.")
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    result = fit_peak_file(args.csv_file, window=args.window, preview=args.preview)
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"2Theta center [deg]: {result.center_two_theta_deg:.6f}")
    print(f"FWHM [deg]: {result.fwhm_deg:.6f}")
    print(f"RMSE: {result.rmse:.6f}")

    if args.plot:
        plot_fit_results(result)


if __name__ == "__main__":
    main()
