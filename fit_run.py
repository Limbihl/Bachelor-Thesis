from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import constants as const
import read_raw_to_cvs
import xrd_math_models


@dataclass(frozen=True)
class ROISelection:
    min_two_theta_deg: float
    max_two_theta_deg: float
    reference_center_deg: float
    source: str


@dataclass(frozen=True)
class InitialGuess:
    amplitude: float
    center_two_theta_deg: float
    fwhm_deg: float
    eta: float
    background_slope: float
    background_offset: float


@dataclass(frozen=True)
class InitialGuessOverride:
    amplitude: float | None = None
    center_two_theta_deg: float | None = None
    fwhm_deg: float | None = None
    eta: float | None = None
    background_slope: float | None = None
    background_offset: float | None = None


@dataclass(frozen=True)
class FitBounds:
    amplitude: tuple[float, float]
    center_two_theta_deg: tuple[float, float]
    fwhm_deg: tuple[float, float]
    eta: tuple[float, float]
    background_slope: tuple[float, float]
    background_offset: tuple[float, float]


@dataclass(frozen=True)
class BoundsOverride:
    amplitude: tuple[float, float] | None = None
    center_two_theta_deg: tuple[float, float] | None = None
    fwhm_deg: tuple[float, float] | None = None
    eta: tuple[float, float] | None = None
    background_slope: tuple[float, float] | None = None
    background_offset: tuple[float, float] | None = None


@dataclass(frozen=True)
class FitConfig:
    window: float = 1.0
    roi_override: tuple[float, float] | None = None
    guess_override: InitialGuessOverride | None = None
    bounds_override: BoundsOverride | None = None
    preview: bool = False


@dataclass(frozen=True)
class FitQualityConfig:
    relative_rmse_threshold: float = 0.15
    absolute_rmse_threshold: float | None = None
    roi_edge_margin_fraction: float = 0.05
    roi_edge_margin_deg_min: float = 0.02
    center_offset_fraction: float = 0.35
    bound_hit_fraction: float = 0.02


@dataclass(frozen=True)
class PeakFitResult:
    file_path: str
    fit_bounds: tuple[float, float]
    roi_source: str
    roi_reference_center_deg: float
    guess_source: str
    auto_initial_guess: InitialGuess
    used_initial_guess: InitialGuess
    guess_override_fields: tuple[str, ...]
    bounds_source: str
    auto_bounds: FitBounds
    used_bounds: FitBounds
    bounds_override_fields: tuple[str, ...]
    success: bool
    message: str
    amplitude: float
    center_two_theta_deg: float
    fwhm_deg: float
    eta: float
    background_slope: float
    background_offset: float
    rmse: float
    relative_rmse: float
    x: np.ndarray
    y: np.ndarray
    y_fit: np.ndarray
    y_ka1: np.ndarray
    y_ka2: np.ndarray
    residuals: np.ndarray
    prefit_flags: tuple[str, ...]
    postfit_flags: tuple[str, ...]
    quality_flags: tuple[str, ...]
    review_required: bool


@dataclass(frozen=True)
class FitReviewSession:
    file_path: str
    attempts: tuple[PeakFitResult, ...]
    accepted: bool
    accepted_attempt_index: int | None
    final_config: FitConfig

    @property
    def final_result(self) -> PeakFitResult | None:
        if not self.attempts:
            return None
        if self.accepted_attempt_index is None:
            return self.attempts[-1]
        return self.attempts[self.accepted_attempt_index]


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


def determine_default_roi_from_header(csv_file_path: str, window: float) -> ROISelection:
    raw_file_path = str(Path(csv_file_path).with_suffix(".raw"))
    _, tths_header, _ = read_raw_to_cvs.extract_pixel_data_and_metadata_from_raw(raw_file_path)
    return ROISelection(
        min_two_theta_deg=tths_header - window,
        max_two_theta_deg=tths_header + window,
        reference_center_deg=tths_header,
        source="header_default",
    )


def resolve_roi(
    csv_file_path: str,
    window: float,
    roi_override: tuple[float, float] | None = None,
) -> ROISelection:
    if roi_override is not None:
        roi_min, roi_max = sorted(float(value) for value in roi_override)
        if roi_min == roi_max:
            raise ValueError("Manual ROI requires two distinct bounds.")
        return ROISelection(
            min_two_theta_deg=roi_min,
            max_two_theta_deg=roi_max,
            reference_center_deg=0.5 * (roi_min + roi_max),
            source="manual_override",
        )

    return determine_default_roi_from_header(csv_file_path, window=window)


def evaluate_prefit_roi(x: np.ndarray, y: np.ndarray) -> tuple[str, ...]:
    flags: list[str] = []

    if len(x) < 10:
        flags.append("roi_low_point_count")
        return tuple(flags)

    peak_index = int(np.argmax(y))
    edge_margin_points = min(3, max(len(x) // 10, 1))
    if peak_index <= edge_margin_points:
        flags.append("roi_raw_max_near_left_edge")
    if peak_index >= len(x) - 1 - edge_margin_points:
        flags.append("roi_raw_max_near_right_edge")

    return tuple(flags)


def build_default_initial_guess(x: np.ndarray, y: np.ndarray, scan_center: float) -> InitialGuess:
    dx = float(np.median(np.diff(x)))
    background_guess = float(np.percentile(y, 10))
    amplitude_guess = max(float(np.max(y) - background_guess), 1.0)
    fwhm_guess = max(0.2, 8.0 * dx)
    return InitialGuess(
        amplitude=amplitude_guess,
        center_two_theta_deg=scan_center,
        fwhm_deg=fwhm_guess,
        eta=0.5,
        background_slope=0.0,
        background_offset=background_guess,
    )


def apply_initial_guess_override(
    initial_guess: InitialGuess,
    guess_override: InitialGuessOverride | None = None,
) -> tuple[InitialGuess, tuple[str, ...], str]:
    if guess_override is None:
        return initial_guess, tuple(), "auto_default"

    updates = {
        field_name: value
        for field_name, value in vars(guess_override).items()
        if value is not None
    }
    if not updates:
        return initial_guess, tuple(), "auto_default"

    override_fields = tuple(updates.keys())
    return replace(initial_guess, **updates), override_fields, "manual_override"


def initial_guess_to_parameter_list(initial_guess: InitialGuess) -> list[float]:
    return [
        initial_guess.amplitude,
        initial_guess.center_two_theta_deg,
        initial_guess.fwhm_deg,
        initial_guess.eta,
        initial_guess.background_slope,
        initial_guess.background_offset,
    ]


def validate_initial_guess(initial_guess: InitialGuess, x: np.ndarray) -> None:
    if initial_guess.amplitude <= 0.0:
        raise ValueError("Initial guess amplitude must be positive.")
    if initial_guess.fwhm_deg <= 0.0:
        raise ValueError("Initial guess FWHM must be positive.")
    if not 0.0 <= initial_guess.eta <= 1.0:
        raise ValueError("Initial guess eta must lie between 0 and 1.")
    if not float(np.min(x)) <= initial_guess.center_two_theta_deg <= float(np.max(x)):
        raise ValueError("Initial guess center must lie inside the selected ROI.")


def build_default_bounds(x: np.ndarray, initial_guess: InitialGuess) -> FitBounds:
    dx = float(np.median(np.diff(x)))
    fwhm_min = max(3.0 * dx, np.finfo(float).eps)
    return FitBounds(
        amplitude=(0.0, max(initial_guess.amplitude * 3.0, initial_guess.amplitude + 1.0)),
        center_two_theta_deg=(float(np.min(x)), float(np.max(x))),
        fwhm_deg=(fwhm_min, 2.0),
        eta=(0.0, 1.0),
        background_slope=(-np.inf, np.inf),
        background_offset=(0.0, np.inf),
    )


def _normalize_bound(bound: tuple[float, float], field_name: str) -> tuple[float, float]:
    lower, upper = float(bound[0]), float(bound[1])
    if lower > upper:
        lower, upper = upper, lower
    if lower == upper:
        raise ValueError(f"Bounds for {field_name} must not collapse to a single value.")
    return lower, upper


def apply_bounds_override(
    bounds: FitBounds,
    bounds_override: BoundsOverride | None = None,
) -> tuple[FitBounds, tuple[str, ...], str]:
    if bounds_override is None:
        return bounds, tuple(), "auto_default"

    updates = {}
    for field_name, value in vars(bounds_override).items():
        if value is not None:
            updates[field_name] = _normalize_bound(value, field_name)

    if not updates:
        return bounds, tuple(), "auto_default"

    override_fields = tuple(updates.keys())
    return replace(bounds, **updates), override_fields, "manual_override"


def validate_fit_bounds(bounds: FitBounds, initial_guess: InitialGuess) -> None:
    for field_name, bound in vars(bounds).items():
        lower, upper = bound
        if lower >= upper:
            raise ValueError(f"Bounds for {field_name} are invalid: lower >= upper.")

    if not bounds.amplitude[0] <= initial_guess.amplitude <= bounds.amplitude[1]:
        raise ValueError("Initial guess amplitude must lie inside amplitude bounds.")
    if not bounds.center_two_theta_deg[0] <= initial_guess.center_two_theta_deg <= bounds.center_two_theta_deg[1]:
        raise ValueError("Initial guess center must lie inside center bounds.")
    if not bounds.fwhm_deg[0] <= initial_guess.fwhm_deg <= bounds.fwhm_deg[1]:
        raise ValueError("Initial guess FWHM must lie inside FWHM bounds.")
    if not bounds.eta[0] <= initial_guess.eta <= bounds.eta[1]:
        raise ValueError("Initial guess eta must lie inside eta bounds.")
    if not bounds.background_slope[0] <= initial_guess.background_slope <= bounds.background_slope[1]:
        raise ValueError("Initial guess slope must lie inside slope bounds.")
    if not bounds.background_offset[0] <= initial_guess.background_offset <= bounds.background_offset[1]:
        raise ValueError("Initial guess background offset must lie inside offset bounds.")


def fit_bounds_to_optimizer_bounds(bounds: FitBounds) -> list[tuple[float, float]]:
    return [
        bounds.amplitude,
        bounds.center_two_theta_deg,
        bounds.fwhm_deg,
        bounds.eta,
        bounds.background_slope,
        bounds.background_offset,
    ]


def _value_near_bound(value: float, bound: tuple[float, float], margin_fraction: float) -> tuple[bool, bool]:
    lower, upper = bound
    if not np.isfinite(lower) and not np.isfinite(upper):
        return False, False

    finite_values = [candidate for candidate in (lower, upper) if np.isfinite(candidate)]
    scale = max(abs(candidate) for candidate in finite_values) if finite_values else 1.0
    span = upper - lower if np.isfinite(lower) and np.isfinite(upper) else scale
    margin = max(margin_fraction * max(span, scale, 1.0), 1e-9)

    near_lower = np.isfinite(lower) and abs(value - lower) <= margin
    near_upper = np.isfinite(upper) and abs(upper - value) <= margin
    return near_lower, near_upper


def evaluate_postfit_roi(result: PeakFitResult, quality_config: FitQualityConfig) -> tuple[str, ...]:
    flags: list[str] = []

    fit_min, fit_max = result.fit_bounds
    roi_width = fit_max - fit_min
    edge_margin = max(
        quality_config.roi_edge_margin_fraction * roi_width,
        quality_config.roi_edge_margin_deg_min,
    )

    if result.center_two_theta_deg - fit_min <= edge_margin:
        flags.append("fit_center_near_left_roi_edge")
    if fit_max - result.center_two_theta_deg <= edge_margin:
        flags.append("fit_center_near_right_roi_edge")
    if abs(result.center_two_theta_deg - result.roi_reference_center_deg) >= quality_config.center_offset_fraction * roi_width:
        flags.append("fit_center_far_from_roi_reference")

    return tuple(flags)


def _deduplicate_flags(flags: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(flags))


def evaluate_fit_quality(result: PeakFitResult, quality_config: FitQualityConfig) -> tuple[tuple[str, ...], bool]:
    flags = list(result.prefit_flags + result.postfit_flags)

    if not result.success:
        flags.append("optimizer_failed")

    lower_hit, upper_hit = _value_near_bound(
        result.fwhm_deg,
        result.used_bounds.fwhm_deg,
        quality_config.bound_hit_fraction,
    )
    if lower_hit:
        flags.append("fit_fwhm_near_lower_bound")
    if upper_hit:
        flags.append("fit_fwhm_near_upper_bound")

    if quality_config.absolute_rmse_threshold is not None and result.rmse > quality_config.absolute_rmse_threshold:
        flags.append("fit_rmse_high_absolute")
    if result.relative_rmse > quality_config.relative_rmse_threshold:
        flags.append("fit_rmse_high_relative")

    quality_flags = _deduplicate_flags(tuple(flags))
    return quality_flags, bool(quality_flags)


def fit_peak_file_with_config(
    csv_file_path: str,
    config: FitConfig,
    quality_config: FitQualityConfig | None = None,
) -> PeakFitResult:
    quality_config = quality_config or FitQualityConfig()
    angle, intensity = load_csv(csv_file_path)

    if config.preview:
        preview_data(angle, intensity, filename=Path(csv_file_path).name)

    roi = resolve_roi(csv_file_path, window=config.window, roi_override=config.roi_override)
    fit_window_min = roi.min_two_theta_deg
    fit_window_max = roi.max_two_theta_deg
    mask = (angle >= fit_window_min) & (angle <= fit_window_max)
    x = angle[mask]
    y = intensity[mask]

    if len(x) == 0:
        raise ValueError(
            f"No data points found in fit window [{fit_window_min:.3f}, {fit_window_max:.3f}] "
            f"for file {csv_file_path}."
        )

    prefit_flags = evaluate_prefit_roi(x, y)
    auto_initial_guess = build_default_initial_guess(x, y, scan_center=roi.reference_center_deg)
    used_initial_guess, guess_override_fields, guess_source = apply_initial_guess_override(
        auto_initial_guess,
        guess_override=config.guess_override,
    )
    validate_initial_guess(used_initial_guess, x)

    auto_bounds = build_default_bounds(x, used_initial_guess)
    used_bounds, bounds_override_fields, bounds_source = apply_bounds_override(
        auto_bounds,
        bounds_override=config.bounds_override,
    )
    validate_fit_bounds(used_bounds, used_initial_guess)

    optimization = xrd_math_models.fit_doublet_peaks(
        x,
        y,
        const.LAM_K1,
        const.LAM_K2,
        const.INTENSITY_RATIO,
        initial_guess_to_parameter_list(used_initial_guess),
        fit_bounds_to_optimizer_bounds(used_bounds),
    )
    amplitude, x01, fwhm, eta, slope, offset = optimization.x
    y_fit = xrd_math_models.doublet_model_f(optimization.x, x, const.LAM_K1, const.LAM_K2, const.INTENSITY_RATIO)
    residuals = y - y_fit

    theta1 = np.deg2rad(x01 / 2.0)
    theta2 = np.arcsin(np.sin(theta1) * (const.LAM_K2 / const.LAM_K1))
    x02 = np.rad2deg(2.0 * theta2)

    background = xrd_math_models.background_f(x, slope, offset)
    y_ka1 = xrd_math_models.pseudo_voigt_f(x, amplitude, x01, fwhm, eta) + background
    y_ka2 = xrd_math_models.pseudo_voigt_f(x, amplitude * const.INTENSITY_RATIO, x02, fwhm, eta) + background
    rmse = float(np.sqrt(np.mean(residuals**2)))
    relative_rmse = float(rmse / max(float(np.max(y) - np.min(y)), 1.0))

    result = PeakFitResult(
        file_path=csv_file_path,
        fit_bounds=(fit_window_min, fit_window_max),
        roi_source=roi.source,
        roi_reference_center_deg=roi.reference_center_deg,
        guess_source=guess_source,
        auto_initial_guess=auto_initial_guess,
        used_initial_guess=used_initial_guess,
        guess_override_fields=guess_override_fields,
        bounds_source=bounds_source,
        auto_bounds=auto_bounds,
        used_bounds=used_bounds,
        bounds_override_fields=bounds_override_fields,
        success=bool(optimization.success),
        message=str(optimization.message),
        amplitude=float(amplitude),
        center_two_theta_deg=float(x01),
        fwhm_deg=float(fwhm),
        eta=float(eta),
        background_slope=float(slope),
        background_offset=float(offset),
        rmse=rmse,
        relative_rmse=relative_rmse,
        x=x,
        y=y,
        y_fit=y_fit,
        y_ka1=y_ka1,
        y_ka2=y_ka2,
        residuals=residuals,
        prefit_flags=prefit_flags,
        postfit_flags=tuple(),
        quality_flags=tuple(),
        review_required=False,
    )
    postfit_flags = evaluate_postfit_roi(result, quality_config)
    result = replace(result, postfit_flags=postfit_flags)
    quality_flags, review_required = evaluate_fit_quality(result, quality_config)
    return replace(result, quality_flags=quality_flags, review_required=review_required)


def fit_peak_file(
    csv_file_path: str,
    window: float = 1.0,
    roi_override: tuple[float, float] | None = None,
    guess_override: InitialGuessOverride | None = None,
    bounds_override: BoundsOverride | None = None,
    preview: bool = False,
    quality_config: FitQualityConfig | None = None,
) -> PeakFitResult:
    config = FitConfig(
        window=window,
        roi_override=roi_override,
        guess_override=guess_override,
        bounds_override=bounds_override,
        preview=preview,
    )
    return fit_peak_file_with_config(csv_file_path, config=config, quality_config=quality_config)


def get_peak_position(csv_file_path: str, window: float = 1.0) -> float | None:
    result = fit_peak_file(csv_file_path, window=window)
    if not result.success:
        return None
    return result.center_two_theta_deg


def _format_guess(initial_guess: InitialGuess) -> str:
    return (
        f"A={initial_guess.amplitude:.3f}, "
        f"center={initial_guess.center_two_theta_deg:.4f}, "
        f"FWHM={initial_guess.fwhm_deg:.4f}, "
        f"eta={initial_guess.eta:.3f}, "
        f"slope={initial_guess.background_slope:.3f}, "
        f"offset={initial_guess.background_offset:.3f}"
    )


def plot_fit_results(result: PeakFitResult, block: bool = True) -> None:
    _, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}, figsize=(9, 6))
    ax_fit, ax_res = ax

    ax_fit.scatter(result.x, result.y, s=15, color="black", alpha=0.6, label="ROI data")
    ax_fit.plot(result.x, result.y_fit, color="red", linewidth=2, label="Doublet fit")
    ax_fit.plot(result.x, result.y_ka1, color="green", linestyle="--", linewidth=1, alpha=0.7, label="Kalpha1")
    ax_fit.plot(result.x, result.y_ka2, color="blue", linestyle="--", linewidth=1, alpha=0.7, label="Kalpha2")
    ax_fit.axvline(
        result.roi_reference_center_deg,
        color="purple",
        linestyle=":",
        linewidth=1,
        alpha=0.8,
        label=f"ROI ref ({result.roi_source})",
    )
    if abs(result.used_initial_guess.center_two_theta_deg - result.roi_reference_center_deg) > 1e-12:
        ax_fit.axvline(
            result.used_initial_guess.center_two_theta_deg,
            color="orange",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            label=f"Initial guess ({result.guess_source})",
        )

    ax_fit.set_ylabel("Intensity [counts]")
    ax_fit.grid(True, alpha=0.3)
    ax_fit.legend()
    ax_fit.set_title(
        f"x0={result.center_two_theta_deg:.4f} deg, "
        f"FWHM={result.fwhm_deg:.4f} deg, "
        f"RMSE={result.rmse:.4f}, relRMSE={result.relative_rmse:.3f}"
    )

    info_lines = [
        f"Guess source: {result.guess_source}",
        f"Bounds source: {result.bounds_source}",
        f"Review required: {result.review_required}",
    ]
    if result.quality_flags:
        info_lines.append("Flags: " + ", ".join(result.quality_flags))
    ax_fit.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax_fit.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
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


def print_fit_summary(result: PeakFitResult, attempt_index: int | None = None) -> None:
    prefix = f"Attempt {attempt_index}: " if attempt_index is not None else ""
    print(f"{prefix}Success: {result.success}")
    print(f"{prefix}Message: {result.message}")
    print(f"{prefix}ROI source: {result.roi_source}")
    print(f"{prefix}ROI bounds [deg]: {result.fit_bounds[0]:.6f}, {result.fit_bounds[1]:.6f}")
    print(f"{prefix}Guess source: {result.guess_source}")
    print(f"{prefix}Used guess: {_format_guess(result.used_initial_guess)}")
    print(f"{prefix}Bounds source: {result.bounds_source}")
    print(
        f"{prefix}Used FWHM bounds [deg]: "
        f"{result.used_bounds.fwhm_deg[0]:.6f}, {result.used_bounds.fwhm_deg[1]:.6f}"
    )
    print(f"{prefix}2Theta center [deg]: {result.center_two_theta_deg:.6f}")
    print(f"{prefix}FWHM [deg]: {result.fwhm_deg:.6f}")
    print(f"{prefix}RMSE: {result.rmse:.6f}")
    print(f"{prefix}Relative RMSE: {result.relative_rmse:.6f}")
    print(f"{prefix}Review required: {result.review_required}")
    if result.guess_override_fields:
        print(f"{prefix}Guess override fields: {', '.join(result.guess_override_fields)}")
    if result.bounds_override_fields:
        print(f"{prefix}Bounds override fields: {', '.join(result.bounds_override_fields)}")
    if result.quality_flags:
        print(f"{prefix}Quality flags: {', '.join(result.quality_flags)}")


def _prompt_float(
    prompt: str,
    current_value: float | None,
    allow_clear: bool = False,
) -> tuple[bool, float | None]:
    while True:
        current_text = "None" if current_value is None else f"{current_value:.6f}"
        raw_value = input(f"{prompt} [current={current_text}] ").strip()
        if raw_value == "":
            return False, current_value
        if allow_clear and raw_value.lower() in {"clear", "auto", "default", "none"}:
            return True, None
        try:
            return True, float(raw_value)
        except ValueError:
            print("Please enter a valid number, press Enter to keep the current value, or type 'auto' to clear.")


def _prompt_bound(
    prompt: str,
    current_value: tuple[float, float] | None,
) -> tuple[bool, tuple[float, float] | None]:
    current_text = "None" if current_value is None else f"({current_value[0]:.6f}, {current_value[1]:.6f})"
    raw_value = input(f"{prompt} [current={current_text}] ").strip()
    if raw_value == "":
        return False, current_value
    if raw_value.lower() in {"clear", "auto", "default", "none"}:
        return True, None

    for separator in (",", " "):
        parts = [part for part in raw_value.replace(separator, " ").split() if part]
        if len(parts) == 2:
            try:
                return True, (float(parts[0]), float(parts[1]))
            except ValueError:
                break

    print("Please enter two numbers separated by space or comma, or type 'auto' to clear.")
    return _prompt_bound(prompt, current_value)


def prompt_roi_override(current_config: FitConfig) -> tuple[float, float] | None:
    current_roi = current_config.roi_override
    changed_min, roi_min = _prompt_float(
        "Manual ROI min in deg (Enter keeps current, 'auto' clears override):",
        None if current_roi is None else current_roi[0],
        allow_clear=True,
    )
    if changed_min and roi_min is None:
        return None

    changed_max, roi_max = _prompt_float(
        "Manual ROI max in deg (Enter keeps current, 'auto' clears override):",
        None if current_roi is None else current_roi[1],
        allow_clear=True,
    )
    if changed_max and roi_max is None:
        return None

    if not changed_min and not changed_max:
        return current_roi
    if roi_min is None or roi_max is None:
        print("Manual ROI requires both min and max. Keeping previous ROI.")
        return current_roi

    return float(roi_min), float(roi_max)


def prompt_guess_override(current_override: InitialGuessOverride | None) -> InitialGuessOverride | None:
    current_override = current_override or InitialGuessOverride()
    updates = {}
    fields = [
        ("amplitude", "Manual amplitude guess (or 'auto')"),
        ("center_two_theta_deg", "Manual center guess in deg (or 'auto')"),
        ("fwhm_deg", "Manual FWHM guess in deg (or 'auto')"),
        ("eta", "Manual eta guess (or 'auto')"),
        ("background_slope", "Manual background slope guess (or 'auto')"),
        ("background_offset", "Manual background offset guess (or 'auto')"),
    ]
    any_change = False
    for field_name, label in fields:
        changed, value = _prompt_float(label + ":", getattr(current_override, field_name), allow_clear=True)
        if changed:
            updates[field_name] = value
            any_change = True

    if not any_change:
        return current_override if any(vars(current_override).values()) else None

    merged = replace(current_override, **updates)
    if not any(value is not None for value in vars(merged).values()):
        return None
    return merged


def prompt_bounds_override(current_override: BoundsOverride | None) -> BoundsOverride | None:
    current_override = current_override or BoundsOverride()
    updates = {}
    fields = [
        ("amplitude", "Manual amplitude bounds min,max (or 'auto')"),
        ("center_two_theta_deg", "Manual center bounds min,max in deg (or 'auto')"),
        ("fwhm_deg", "Manual FWHM bounds min,max in deg (or 'auto')"),
        ("eta", "Manual eta bounds min,max (or 'auto')"),
        ("background_slope", "Manual background slope bounds min,max (or 'auto')"),
        ("background_offset", "Manual background offset bounds min,max (or 'auto')"),
    ]
    any_change = False
    for field_name, label in fields:
        changed, value = _prompt_bound(label + ":", getattr(current_override, field_name))
        if changed:
            updates[field_name] = value
            any_change = True

    if not any_change:
        return current_override if any(value is not None for value in vars(current_override).values()) else None

    merged = replace(current_override, **updates)
    if not any(value is not None for value in vars(merged).values()):
        return None
    return merged


def review_fit_file(
    csv_file_path: str,
    config: FitConfig | None = None,
    quality_config: FitQualityConfig | None = None,
    plot_each_attempt: bool = True,
) -> FitReviewSession:
    current_config = config or FitConfig()
    attempts: list[PeakFitResult] = []
    preview_shown = False

    while True:
        attempt_config = current_config if not preview_shown else replace(current_config, preview=False)
        result = fit_peak_file_with_config(csv_file_path, config=attempt_config, quality_config=quality_config)
        attempts.append(result)
        preview_shown = preview_shown or attempt_config.preview

        print()
        print_fit_summary(result, attempt_index=len(attempts))
        if plot_each_attempt:
            plot_fit_results(result, block=True)

        while True:
            action = input(
                "Action: [a]ccept, [r]oi, [g]uess, [b]ounds, [p]lot again, [q]uit > "
            ).strip().lower()

            if action in {"a", "accept"}:
                return FitReviewSession(
                    file_path=csv_file_path,
                    attempts=tuple(attempts),
                    accepted=True,
                    accepted_attempt_index=len(attempts) - 1,
                    final_config=current_config,
                )
            if action in {"q", "quit"}:
                return FitReviewSession(
                    file_path=csv_file_path,
                    attempts=tuple(attempts),
                    accepted=False,
                    accepted_attempt_index=None,
                    final_config=current_config,
                )
            if action in {"p", "plot"}:
                plot_fit_results(result, block=True)
                continue
            if action in {"r", "roi"}:
                current_config = replace(current_config, roi_override=prompt_roi_override(current_config))
                break
            if action in {"g", "guess"}:
                current_config = replace(
                    current_config,
                    guess_override=prompt_guess_override(current_config.guess_override),
                )
                break
            if action in {"b", "bounds"}:
                current_config = replace(
                    current_config,
                    bounds_override=prompt_bounds_override(current_config.bounds_override),
                )
                break

            print("Unknown action. Please choose a, r, g, b, p, or q.")


def _parse_optional_bound_pair(lower: float | None, upper: float | None, label: str) -> tuple[float, float] | None:
    if lower is None and upper is None:
        return None
    if (lower is None) ^ (upper is None):
        raise ValueError(f"Please provide both lower and upper values for {label}.")
    return float(lower), float(upper)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a Kalpha doublet in a single CSV scan.")
    parser.add_argument("csv_file", help="Path to the converted CSV scan.")
    parser.add_argument("--window", type=float, default=1.0, help="Half-width of the default ROI around the header center.")
    parser.add_argument("--roi-min", type=float, help="Manual lower ROI bound in 2Theta [deg].")
    parser.add_argument("--roi-max", type=float, help="Manual upper ROI bound in 2Theta [deg].")
    parser.add_argument("--guess-center", type=float, help="Manual initial guess for peak center in 2Theta [deg].")
    parser.add_argument("--guess-fwhm", type=float, help="Manual initial guess for FWHM in [deg].")
    parser.add_argument("--guess-amplitude", type=float, help="Manual initial guess for peak amplitude.")
    parser.add_argument("--bound-center-min", type=float, help="Manual lower bound for peak center in 2Theta [deg].")
    parser.add_argument("--bound-center-max", type=float, help="Manual upper bound for peak center in 2Theta [deg].")
    parser.add_argument("--bound-fwhm-min", type=float, help="Manual lower bound for FWHM in [deg].")
    parser.add_argument("--bound-fwhm-max", type=float, help="Manual upper bound for FWHM in [deg].")
    parser.add_argument("--bound-amplitude-min", type=float, help="Manual lower bound for amplitude.")
    parser.add_argument("--bound-amplitude-max", type=float, help="Manual upper bound for amplitude.")
    parser.add_argument("--relative-rmse-threshold", type=float, default=0.15, help="Flag fits above this relative RMSE.")
    parser.add_argument("--absolute-rmse-threshold", type=float, help="Optional absolute RMSE threshold.")
    parser.add_argument("--preview", action="store_true", help="Show the raw scan before fitting.")
    parser.add_argument("--plot", action="store_true", help="Plot the fit and residuals.")
    parser.add_argument("--review", action="store_true", help="Run an interactive review/refit loop for this file.")
    return parser


def main() -> None:
    args = _build_cli().parse_args()

    roi_override = _parse_optional_bound_pair(args.roi_min, args.roi_max, "manual ROI")
    guess_override = InitialGuessOverride(
        amplitude=args.guess_amplitude,
        center_two_theta_deg=args.guess_center,
        fwhm_deg=args.guess_fwhm,
    )
    if not any(value is not None for value in vars(guess_override).values()):
        guess_override = None

    bounds_override = BoundsOverride(
        amplitude=_parse_optional_bound_pair(args.bound_amplitude_min, args.bound_amplitude_max, "amplitude bounds"),
        center_two_theta_deg=_parse_optional_bound_pair(args.bound_center_min, args.bound_center_max, "center bounds"),
        fwhm_deg=_parse_optional_bound_pair(args.bound_fwhm_min, args.bound_fwhm_max, "FWHM bounds"),
    )
    if not any(value is not None for value in vars(bounds_override).values()):
        bounds_override = None

    config = FitConfig(
        window=args.window,
        roi_override=roi_override,
        guess_override=guess_override,
        bounds_override=bounds_override,
        preview=args.preview,
    )
    quality_config = FitQualityConfig(
        relative_rmse_threshold=args.relative_rmse_threshold,
        absolute_rmse_threshold=args.absolute_rmse_threshold,
    )

    if args.review:
        session = review_fit_file(
            args.csv_file,
            config=config,
            quality_config=quality_config,
            plot_each_attempt=True,
        )
        final_result = session.final_result
        print()
        print(f"Accepted: {session.accepted}")
        print(f"Attempts: {len(session.attempts)}")
        if final_result is not None:
            print_fit_summary(final_result)
        return

    result = fit_peak_file_with_config(args.csv_file, config=config, quality_config=quality_config)
    print_fit_summary(result)
    if args.plot:
        plot_fit_results(result)


if __name__ == "__main__":
    main()
