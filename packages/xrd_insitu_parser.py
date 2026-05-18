"""
XRD In-Situ Annealing Data Parser
==================================
Parses .xy (spectra + metadata) and .lst (Rietveld refinement) files
from Profex/BGMN for in-situ annealing XRD experiments.

Usage:
    python xrd_insitu_parser.py --folder /path/to/data --sample NdCeFeB_5_3_Cerich_2

Then use the returned XRDDataset object to plot phase fractions,
lattice parameters, or check individual spectra.
"""

import re
import os
import glob
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    """Lattice parameters and R-factor for one phase from a .lst file."""

    name: str
    fraction: float = np.nan  # phase weight fraction (Q value)
    fraction_err: float = np.nan
    a: float = np.nan  # Å or nm, unit as in .lst
    a_err: float = np.nan
    b: float = np.nan
    b_err: float = np.nan
    c: float = np.nan
    c_err: float = np.nan
    unit: str = "nm"  # NM or ANG
    rphase: float = np.nan  # per-phase R factor (%)
    volume: float = np.nan  # unit cell volume (computed)


@dataclass
class ScanResult:
    """All data for one scan (one temperature point)."""

    scan_index: int
    filename_xy: str
    filename_lst: str
    temperature: float = np.nan  # °C from nanodacse_in1 signal
    energy_keV: float = np.nan
    wavelength_A: float = np.nan
    # 1-D spectrum
    q: np.ndarray = field(default_factory=lambda: np.array([]))
    intensity: np.ndarray = field(default_factory=lambda: np.array([]))
    # Refinement quality
    rwp: float = np.nan
    rp: float = np.nan
    r: float = np.nan
    # Per-phase results  {phase_name: PhaseResult}
    phases: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_value_err(text: str) -> tuple[float, float]:
    """Parse 'value+-error' or plain 'value'. Returns (value, error)."""
    text = text.strip()
    if "+-" in text:
        parts = text.split("+-")
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return np.nan, np.nan
    elif text.upper() in ("UNDEF", ""):
        return np.nan, np.nan
    else:
        try:
            return float(text), np.nan
        except ValueError:
            return np.nan, np.nan


def parse_xy(filepath: str) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Parse a .xy file.

    Returns
    -------
    temperature : float   (nanodacse_in1 signal, °C)
    energy_keV  : float
    wavelength_A : float
    q           : np.ndarray  (first column)
    intensity   : np.ndarray  (second column)
    """
    temperature = np.nan
    energy_keV = np.nan
    wavelength_A = np.nan
    data_lines = []

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                if "nanodacse_in1/signal:" in line:
                    try:
                        temperature = float(line.split(":")[-1])
                    except ValueError:
                        pass
                elif re.search(r"#instrument/energy/signal:", line):
                    try:
                        energy_keV = float(line.split(":")[-1])
                    except ValueError:
                        pass
                elif re.search(r"#instrument/wavelength/signal:", line):
                    try:
                        wavelength_A = float(line.split(":")[-1])
                    except ValueError:
                        pass
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        data_lines.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass

    data = np.array(data_lines)
    if data.size:
        q, intensity = data[:, 0], data[:, 1]
    else:
        q, intensity = np.array([]), np.array([])

    return temperature, energy_keV, wavelength_A, q, intensity


def parse_lst(filepath: str) -> tuple[float, float, float, dict, dict]:
    """
    Parse a Profex/BGMN .lst Rietveld refinement result file.

    Returns
    -------
    rwp, rp, r : float  (global R factors, %)
    global_fractions : dict  {phase_name: (fraction, fraction_err)}
    phases : dict  {phase_name: PhaseResult}
    """
    rwp = rp = r = np.nan
    global_fractions: dict = {}
    phases: dict = {}

    with open(filepath, "r") as fh:
        content = fh.read()

    # --- Global R factors ---
    m = re.search(
        r"Rp=([0-9.]+)%\s+Rpb=[0-9.%]+\s+R=([0-9.]+)%\s+Rwp=([0-9.]+)%", content
    )
    if m:
        rp = float(m.group(1))
        r = float(m.group(2))
        rwp = float(m.group(3))

    # --- Global phase fractions (Q... lines before first "Local parameters") ---
    global_section = content.split("Local parameters")[0]
    for match in re.finditer(r"^Q(\w+)=([^\n]+)", global_section, re.MULTILINE):
        phase_name = match.group(1)
        val_str = match.group(2).strip()
        val, err = _parse_value_err(val_str)
        global_fractions[phase_name] = (val, err)

    # --- Per-phase blocks ---
    phase_blocks = re.split(r"Local parameters and GOALs for phase\s+", content)
    for block in phase_blocks[1:]:
        lines = block.strip().splitlines()
        phase_name = lines[0].strip()
        pr = PhaseResult(name=phase_name)

        um = re.search(r"^UNIT=(\w+)", block, re.MULTILINE)
        if um:
            pr.unit = um.group(1)

        rpm = re.search(r"^Rphase=([0-9.]+)%", block, re.MULTILINE)
        if rpm:
            pr.rphase = float(rpm.group(1))

        for param, attr, err_attr in [
            ("A", "a", "a_err"),
            ("B", "b", "b_err"),
            ("C", "c", "c_err"),
        ]:
            pm = re.search(rf"^{param}=([^\n]+)", block, re.MULTILINE)
            if pm:
                val, err = _parse_value_err(pm.group(1))
                setattr(pr, attr, val)
                setattr(pr, err_attr, err)

        for gname, (gval, gerr) in global_fractions.items():
            if gname.lower() == phase_name.lower():
                pr.fraction = gval
                pr.fraction_err = gerr
                break

        # Compute unit-cell volume (convert NM -> Å)
        factor = 10.0 if pr.unit == "NM" else 1.0
        a = pr.a * factor if not np.isnan(pr.a) else np.nan
        b = pr.b * factor if not np.isnan(pr.b) else np.nan
        c = pr.c * factor if not np.isnan(pr.c) else np.nan
        if not np.isnan(a):
            if np.isnan(b) and np.isnan(c):
                pr.volume = a**3
            elif np.isnan(b) and not np.isnan(c):
                pr.volume = a**2 * c
            elif not np.isnan(b) and not np.isnan(c):
                pr.volume = a * b * c

        phases[phase_name] = pr

    return rwp, rp, r, global_fractions, phases


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class XRDDataset:
    """
    Container for all scans of an in-situ annealing experiment.

    Attributes
    ----------
    scans : list[ScanResult]  sorted by scan index
    df    : pd.DataFrame      summary table (one row per scan)
    """

    def __init__(self, scans: list):
        self.scans: list[ScanResult] = sorted(scans, key=lambda s: s.scan_index)
        self.df = self._build_dataframe()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_dataframe(self) -> pd.DataFrame:
        rows = []
        for s in self.scans:
            row = {
                "scan_index": s.scan_index,
                "temperature_C": s.temperature,
                "energy_keV": s.energy_keV,
                "wavelength_A": s.wavelength_A,
                "rwp": s.rwp,
                "rp": s.rp,
                "r": s.r,
                "xy_file": s.filename_xy,
                "lst_file": s.filename_lst,
            }
            for phase_name, pr in s.phases.items():
                safe = phase_name.replace(" ", "_")
                row[f"{safe}_fraction"] = pr.fraction
                row[f"{safe}_fraction_err"] = pr.fraction_err
                row[f"{safe}_a_nm"] = pr.a
                row[f"{safe}_a_err_nm"] = pr.a_err
                row[f"{safe}_c_nm"] = pr.c
                row[f"{safe}_c_err_nm"] = pr.c_err
                row[f"{safe}_volume_A3"] = pr.volume
                row[f"{safe}_rphase"] = pr.rphase
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _colorbar_layout(
        z_min: float,
        z_max: float,
        precision: int = 1,
        title: str = "",
        prefix: str = "",
    ) -> dict[str, Any]:
        """Shared colorbar style matching your PlotMixin."""
        z_mid = (z_min + z_max) / 2
        return dict(
            title=dict(
                text=prefix + title + "<br>&nbsp;<br>",
                font=dict(size=24),
            ),
            tickmode="array",
            tickvals=[
                z_min,
                (z_min + z_mid) / 2,
                z_mid,
                (z_max + z_mid) / 2,
                z_max,
            ],
            ticktext=[
                f"{z_min:.{precision}f}",
                f"{(z_min + z_mid) / 2:.{precision}f}",
                f"{z_mid:.{precision}f}",
                f"{(z_max + z_mid) / 2:.{precision}f}",
                f"{z_max:.{precision}f}",
            ],
            tickfont=dict(size=24),
            ticklen=8,
            thickness=25,
        )

    def _filter_scans(self, scan_start: Optional[int]) -> list[ScanResult]:
        """Return scans with index >= scan_start (or all if None)."""
        if scan_start is None:
            return self.scans
        return [s for s in self.scans if s.scan_index >= scan_start]

    def _filtered_df(self, scan_start: Optional[int]) -> pd.DataFrame:
        if scan_start is None:
            return self.df.copy()
        return self.df[self.df["scan_index"] >= scan_start].copy()

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_scan(self, index: int) -> Optional[ScanResult]:
        """Return ScanResult by scan index."""
        for s in self.scans:
            if s.scan_index == index:
                return s
        return None

    def phase_names(self) -> list[str]:
        if self.scans:
            return list(self.scans[0].phases.keys())
        return []

    # ------------------------------------------------------------------
    # Plotting — Phase fractions
    # ------------------------------------------------------------------

    def plot_phase_fractions(
        self,
        phases: Optional[list[str]] = None,
        scan_start: Optional[int] = None,
        show_errorbars: bool = True,
        colorscale: str = "plasma",
        marker_size: int = 6,
        width: int = 950,
        height: int = 650,
        title: str = "Phase fractions vs Temperature",
        show: bool = True,
    ) -> go.Figure:
        """
        Plot phase weight fractions vs temperature.

        Parameters
        ----------
        phases        : list of phase names to plot (None = all)
        scan_start    : only include scans with index >= this value
        show_errorbars: draw +/-1sigma error bars (default True)
        colorscale    : Plotly colorscale name used to colour the phases
        """
        df = self._filtered_df(scan_start).sort_values("temperature_C")
        if phases is None:
            phases = self.phase_names()

        n = max(len(phases), 1)
        colors = px.colors.sample_colorscale(
            colorscale, [i / (n - 1) if n > 1 else 0.5 for i in range(n)]
        )

        fig = go.Figure()
        for ph, color in zip(phases, colors):
            col = f"{ph}_fraction"
            err_col = f"{ph}_fraction_err"
            if col not in df.columns:
                continue

            y = df[col].values
            yerr = (
                df[err_col].values
                if (show_errorbars and err_col in df.columns)
                else None
            )

            eb = (
                dict(type="data", array=yerr, visible=True, thickness=1.5, width=4)
                if yerr is not None
                else None
            )

            fig.add_trace(
                go.Scatter(
                    x=df["temperature_C"],
                    y=y,
                    error_y=eb,
                    mode="markers+lines",
                    marker=dict(size=marker_size, color=color),
                    line=dict(color=color, width=1.5),
                    name=ph,
                    customdata=df["scan_index"].values,
                    hovertemplate="T: %{x:.1f} °C<br>Fraction: %{y:.4f}<br>Scan: %{customdata}<extra></extra>",
                )
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis=dict(
                title="Temperature (°C)",
                tickfont=dict(size=24),
                title_font=dict(size=24),
            ),
            yaxis=dict(
                title="Phase fraction",
                tickfont=dict(size=24),
                title_font=dict(size=24),
            ),
            legend=dict(font=dict(size=18)),
            width=width,
            height=height,
        )
        if show:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # Plotting — Lattice parameters
    # ------------------------------------------------------------------

    def plot_lattice_param(
        self,
        phase: str,
        param: str = "a",
        scan_start: Optional[int] = None,
        show_errorbars: bool = True,
        color: str = "#00B9DE",
        marker_size: int = 6,
        width: int = 950,
        height: int = 650,
        precision: int = 4,
        show: bool = True,
    ) -> go.Figure:
        """
        Plot one lattice parameter vs temperature.

        Parameters
        ----------
        phase         : phase name as in .lst (e.g. 'R2Fe14B')
        param         : 'a', 'c', or 'volume'
        scan_start    : only include scans with index >= this value
        show_errorbars: draw +/-1sigma error bars (default True)
        color         : marker/line colour (any Plotly-accepted string)
        """
        df = self._filtered_df(scan_start).sort_values("temperature_C")
        safe = phase.replace(" ", "_")

        if param in ("a", "c"):
            col = f"{safe}_{param}_nm"
            err_col = f"{safe}_{param}_err_nm"
            y_label = f"{param} (nm)"
        elif param == "volume":
            col = f"{safe}_volume_A3"
            err_col = None
            y_label = "Unit-cell volume (A^3)"
        else:
            raise ValueError(f"Unknown param '{param}'. Choose 'a', 'c', or 'volume'.")

        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found. Available phases: {self.phase_names()}"
            )

        y = df[col].values
        yerr = (
            df[err_col].values
            if (show_errorbars and err_col and err_col in df.columns)
            else None
        )

        eb = (
            dict(type="data", array=yerr, visible=True, thickness=1.5, width=4)
            if yerr is not None
            else None
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["temperature_C"],
                y=y,
                error_y=eb,
                mode="markers+lines",
                marker=dict(size=marker_size, color=color),
                line=dict(color=color, width=1.5),
                name=f"{phase} {param}",
                customdata=df["scan_index"].values,
                hovertemplate="T: %{x:.1f} °C<br>"
                + y_label
                + ": %{y:.6f}<br>Scan: %{customdata}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(text=f"{phase} — {y_label} vs Temperature", font=dict(size=20)),
            xaxis=dict(
                title="Temperature (°C)",
                tickfont=dict(size=24),
                title_font=dict(size=24),
            ),
            yaxis=dict(
                title=y_label,
                tickfont=dict(size=24),
                title_font=dict(size=24),
            ),
            legend=dict(font=dict(size=18)),
            width=width,
            height=height,
        )
        if show:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # Plotting — Generic 1-D scatter (mirrors your PlotMixin.plot_1d)
    # ------------------------------------------------------------------

    def plot_1d(
        self,
        x: list | np.ndarray,
        y: list | np.ndarray,
        color: Optional[list | np.ndarray] = None,
        yerr: Optional[list | np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        color_label: str = "color",
        range_color: Optional[tuple[float, float]] = None,
        x_range: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        colorscale: str = "plasma",
        marker_size: int = 8,
        width: int = 950,
        height: int = 650,
        title: str = "",
        precision: int = 2,
        show_errorbars: bool = True,
        show: bool = True,
    ) -> go.Figure:
        """
        Generic scatter/line plot with optional colour encoding and error bars.
        Mirrors PlotMixin.plot_1d, extended with yerr and show_errorbars.

        Parameters
        ----------
        x, y          : data arrays
        color         : optional colour-encoding array
        yerr          : optional y-error array (shown only when show_errorbars=True)
        show_errorbars: toggle error bars (default True)
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        arrays_to_check = [x, y]
        if color is not None:
            color = np.asarray(color, dtype=float)
            arrays_to_check.append(color)
        if yerr is not None:
            yerr = np.asarray(yerr, dtype=float)

        valid = np.ones(len(x), dtype=bool)
        for a in arrays_to_check:
            valid &= ~np.isnan(a)

        x = x[valid]
        y = y[valid]
        if color is not None:
            color = color[valid]
        if yerr is not None:
            yerr = yerr[valid]

        order = np.argsort(x)
        x, y = x[order], y[order]
        if color is not None:
            color = color[order]
        if yerr is not None:
            yerr = yerr[order]

        if scan_ids is not None:
            scan_ids = np.asarray(scan_ids)[valid][order]

        df = pd.DataFrame({x_label: x, y_label: y})
        if color is not None:
            df[color_label] = color
            if range_color is None:
                range_color = (float(np.nanmin(color)), float(np.nanmax(color)))

        fig = px.scatter(
            df,
            x=x_label,
            y=y_label,
            color=color_label if color is not None else None,
            range_color=range_color,
            color_continuous_scale=colorscale if color is not None else None,
            width=width,
            height=height,
            title=title or f"{y_label} vs {x_label}",
        )
        fig.update_traces(marker={"size": marker_size})

        # Error bars overlaid as a separate transparent-marker trace
        if show_errorbars and yerr is not None:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    error_y=dict(
                        type="data", array=yerr, visible=True, thickness=1.5, width=4
                    ),
                    mode="markers",
                    marker=dict(size=marker_size, opacity=0),
                    showlegend=False,
                )
            )

        if scan_ids is not None:
            fig.update_traces(
                customdata=scan_ids,
                hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>Scan: %{{customdata}}<extra></extra>",
            )

        fig.update_layout(
            xaxis=dict(tickfont=dict(size=24), title_font=dict(size=24)),
            yaxis=dict(tickfont=dict(size=24), title_font=dict(size=24)),
        )
        if x_range:
            fig.update_xaxes(range=list(x_range))
        if y_range:
            fig.update_yaxes(range=list(y_range))
        if color is not None and range_color is not None:
            fig.update_coloraxes(
                colorbar=self._colorbar_layout(
                    z_min=range_color[0],
                    z_max=range_color[1],
                    precision=precision,
                    title=color_label,
                )
            )
        if show:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # Plotting — Waterfall spectra
    # ------------------------------------------------------------------

    def plot_spectra_stack(
        self,
        scan_indices: Optional[list[int]] = None,
        scan_start: Optional[int] = None,
        offset_factor: float = 0.15,
        colorscale: str = "plasma",
        xlim: Optional[tuple[float, float]] = None,
        width: int = 950,
        height: int = 700,
        x_label: str = "q or 2theta",
        show: bool = True,
    ) -> go.Figure:
        """
        Waterfall plot of 1-D diffraction patterns coloured by temperature.

        Parameters
        ----------
        scan_indices  : explicit list of scan indices (overrides scan_start)
        scan_start    : only include scans with index >= this value
        offset_factor : fraction of max intensity used as vertical step per pattern
        colorscale    : Plotly colorscale (default 'plasma')
        xlim          : optional (xmin, xmax) zoom window
        """
        if scan_indices is not None:
            scans = [s for s in self.scans if s.scan_index in scan_indices]
        else:
            scans = self._filter_scans(scan_start)

        if not scans:
            raise ValueError("No scans match the given filters.")

        temps = np.array([s.temperature for s in scans])
        t_min, t_max = np.nanmin(temps), np.nanmax(temps)

        max_int = max(
            (s.intensity.max() for s in scans if s.intensity.size), default=1.0
        )
        offset_step = max_int * offset_factor

        n = len(scans)
        norm_temps = (temps - t_min) / (t_max - t_min + 1e-9)
        colors = px.colors.sample_colorscale(colorscale, norm_temps.tolist())

        # Label every ~10th trace to keep legend readable
        label_step = max(1, n // 10)

        fig = go.Figure()
        for i, (s, color) in enumerate(zip(scans, colors)):
            if s.q.size == 0:
                continue
            offset = i * offset_step
            fig.add_trace(
                go.Scatter(
                    x=s.q,
                    y=s.intensity + offset,
                    mode="lines",
                    line=dict(color=color, width=0.8),
                    name=f"{s.temperature:.0f} C",
                    showlegend=(i % label_step == 0),
                    legendgroup=str(i),
                    customdata=np.full(len(s.q), s.scan_index),
                    hovertemplate="q: %{x:.4f}<br>I: %{y:.0f}<br>Scan: %{customdata}<extra></extra>",
                )
            )

        # Colorbar via a dummy invisible scatter
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    colorscale=colorscale,
                    cmin=t_min,
                    cmax=t_max,
                    color=[t_min, t_max],
                    colorbar=self._colorbar_layout(
                        z_min=t_min, z_max=t_max, precision=0, title="Temperature (C)"
                    ),
                    showscale=True,
                    size=0,
                ),
                showlegend=False,
            )
        )

        fig.update_layout(
            title=dict(
                text="Diffraction patterns — coloured by temperature",
                font=dict(size=20),
            ),
            xaxis=dict(
                title=x_label,
                tickfont=dict(size=24),
                title_font=dict(size=24),
                range=list(xlim) if xlim else None,
            ),
            yaxis=dict(
                title="Intensity (offset)",
                tickfont=dict(size=24),
                title_font=dict(size=24),
            ),
            width=width,
            height=height,
        )
        if show:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # Plotting — Peak intensity
    # ------------------------------------------------------------------

    def plot_peak_intensity(
        self,
        q_center: float,
        q_width: float = 0.05,
        scan_start: Optional[int] = None,
        color: str = "#636EFA",
        marker_size: int = 6,
        width: int = 950,
        height: int = 650,
        show: bool = True,
    ) -> go.Figure:
        """
        Integrate intensity in [q_center +/- q_width] and plot vs temperature.

        Parameters
        ----------
        q_center   : centre of the integration window
        q_width    : half-width of the integration window
        scan_start : only include scans with index >= this value
        """
        scans = self._filter_scans(scan_start)
        temps, integrals, scan_ids = [], [], []
        for s in scans:
            if s.q.size == 0:
                continue
            mask = np.abs(s.q - q_center) <= q_width
            if mask.sum() == 0:
                continue
            temps.append(s.temperature)
            integrals.append(float(s.intensity[mask].sum()))
            scan_ids.append(s.scan_index)

        if not temps:
            raise ValueError("No data in the requested q range / scan range.")

        idx = np.argsort(temps)
        temps = np.array(temps)[idx]
        integrals = np.array(integrals)[idx]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=temps,
                y=integrals,
                mode="markers+lines",
                marker=dict(size=marker_size, color=color),
                line=dict(color=color, width=1.5),
                customdata=scan_ids,
                hovertemplate="T: %{x:.1f} °C<br>Intensity: %{y:.0f}<br>Scan: %{customdata}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(
                text=f"Peak intensity — q = {q_center} +/- {q_width}",
                font=dict(size=20),
            ),
            xaxis=dict(
                title="Temperature (C)",
                tickfont=dict(size=24),
                title_font=dict(size=24),
            ),
            yaxis=dict(
                title="Integrated intensity",
                tickfont=dict(size=24),
                title_font=dict(size=24),
            ),
            width=width,
            height=height,
        )
        if show:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_csv(self, path: str):
        """Save the summary DataFrame to CSV."""
        self.df.to_csv(path, index=False)
        print(f"Saved summary table -> {path}")

    def save_spectra_npz(self, path: str):
        """Save all spectra as a compressed .npz archive."""
        arrays = {}
        for s in self.scans:
            arrays[f"q_{s.scan_index}"] = s.q
            arrays[f"I_{s.scan_index}"] = s.intensity
            arrays[f"T_{s.scan_index}"] = np.array([s.temperature])
        np.savez_compressed(path, **arrays)
        print(f"Saved spectra -> {path}")


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_dataset(
    folder: str,
    sample_name: str,
    k_start: int = 6,
    k_end: int = None,
    temp_key: str = "nanodacse_in1",
) -> XRDDataset:
    """
    Load all .xy/.lst pairs for *sample_name* in *folder*.

    Files are expected as:
        <folder>/<sample_name>_<k>.xy
        <folder>/<sample_name>_<k>.lst

    Parameters
    ----------
    folder      : directory containing the files
    sample_name : base name without index (e.g. 'NdCeFeB_5_3_Cerich_2')
    k_start     : first index to load (default 6)
    k_end       : last index (inclusive); None = auto-detect from glob
    temp_key    : header key whose /signal: value is the temperature
    """
    folder = Path(folder)
    pattern = str(folder / f"{sample_name}_*.xy")
    xy_files = sorted(glob.glob(pattern))

    if not xy_files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found. " f"Check folder and sample_name."
        )

    index_re = re.compile(rf"{re.escape(sample_name)}_(\d+)\.xy$")
    scans = []

    for xy_path in xy_files:
        m = index_re.search(xy_path)
        if not m:
            continue
        k = int(m.group(1))
        if k < k_start:
            continue
        if k_end is not None and k > k_end:
            continue

        lst_path = xy_path.replace(".xy", ".lst")
        if not os.path.exists(lst_path):
            print(f"  [warn] Missing .lst for scan {k}, skipping.")
            continue

        try:
            temp, energy, wl, q, intensity = parse_xy(xy_path)
        except Exception as exc:
            print(f"  [warn] Could not parse {xy_path}: {exc}")
            continue

        try:
            rwp, rp, r, _, phases = parse_lst(lst_path)
        except Exception as exc:
            print(f"  [warn] Could not parse {lst_path}: {exc}")
            phases, rwp, rp, r = {}, np.nan, np.nan, np.nan

        scan = ScanResult(
            scan_index=k,
            filename_xy=xy_path,
            filename_lst=lst_path,
            temperature=temp,
            energy_keV=energy,
            wavelength_A=wl,
            q=q,
            intensity=intensity,
            rwp=rwp,
            rp=rp,
            r=r,
            phases=phases,
        )
        scans.append(scan)
        print(f"  Loaded scan {k:>4d}  T={temp:.1f}C  phases={list(phases.keys())}")

    print(f"\nTotal scans loaded: {len(scans)}")
    return XRDDataset(scans)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Parse in-situ XRD annealing data (.xy + .lst)"
    )
    parser.add_argument("--folder", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--k_start", type=int, default=6)
    parser.add_argument("--k_end", type=int, default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--npz", default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    ds = load_dataset(
        folder=args.folder,
        sample_name=args.sample,
        k_start=args.k_start,
        k_end=args.k_end,
    )

    if args.csv:
        ds.save_csv(args.csv)
    if args.npz:
        ds.save_spectra_npz(args.npz)
    if args.plot:
        ds.plot_phase_fractions()
        ds.plot_spectra_stack()

    return ds


if __name__ == "__main__":
    main()
