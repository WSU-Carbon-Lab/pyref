"""
Interactive NEXAFS widget using Panel and Bokeh.

Supports widget-driven pre/post edge (FloatInputs) and box-select on the plot
to set pre or post edge. Requires a Panel/Bokeh server for box-select callbacks.
"""

from __future__ import annotations

from typing import Any

import pandas as pd  # noqa: TC002

from pyref.nexafs.directory import NexafsDirectory  # noqa: TC001
from pyref.nexafs.normalization import NormalizationScheme, fit_normalization


def _current_df_for_angle(
    dfs: list[pd.DataFrame],
    angles: list[float],
    angle: float | None,
) -> pd.DataFrame | None:
    if not dfs or not angles or angle is None:
        return None
    for df in dfs:
        if float(df["Angle"].iloc[0]) == angle:
            return df
    return dfs[0]


def _build_bokeh_plot(
    angle: float | None,
    pre_lo: float,
    pre_hi: float,
    post_lo: float,
    post_hi: float,
    norm_scheme: NormalizationScheme,
    *,
    view_bg_subtracted: bool,
    sample_name: str,
    formula: str,
    directory: NexafsDirectory,
    dfs: list[pd.DataFrame],
    angles: list[float],
    pre_lo_widget: Any,
    pre_hi_widget: Any,
    post_lo_widget: Any,
    post_hi_widget: Any,
    selection_applies_to: int,
) -> Any:
    from bokeh.models import BoxAnnotation, BoxSelectTool, ColumnDataSource
    from bokeh.models.ranges import Range1d
    from bokeh.plotting import figure

    df = _current_df_for_angle(dfs, angles, angle)
    p = figure(
        width=550,
        height=400,
        x_axis_label="Energy",
        y_axis_label="Intensity",
        title=f"Sample: {sample_name}",
        tools="pan,wheel_zoom,box_zoom,save,reset",
    )
    if df is None or df.empty:
        return p
    energy = df["Energy"].to_numpy()
    intensity = df["PD Corrected"].to_numpy()
    if "Norm Abs" in df.columns:
        intensity = df["Norm Abs"].to_numpy()
    e_min = float(energy.min())
    e_max = float(energy.max())
    margin = max(2.0, (e_max - e_min) * 0.02)
    p.x_range = Range1d(start=e_min - margin, end=e_max + margin)
    source = ColumnDataSource(data={"x": energy.tolist(), "y": intensity.tolist()})
    p.scatter(source=source, x="x", y="y", size=4, color="#1f77b4", alpha=0.8)
    p.line(source=source, x="x", y="y", line_width=2, color="#1f77b4")
    pre_edge = (pre_lo, pre_hi)
    post_edge = (post_lo, post_hi)
    result = None
    if norm_scheme != NormalizationScheme.NONE:
        try:
            mu_arrays = directory.build_mu_arrays(energy, formula)
            result = fit_normalization(
                norm_scheme,
                energy,
                intensity,
                pre_edge,
                post_edge,
                mu_arrays["chemical"],
                mu_si=mu_arrays.get("Si"),
                mu_o=mu_arrays.get("O"),
            )
        except (ValueError, RuntimeError):
            result = None
    if result is not None:
        full_curve = result["full_curve"]
        background_curve = result["background_curve"]
        scaled_mu = result["scaled_mu"]
        if view_bg_subtracted:
            bg_sub = intensity - background_curve
            p.line(
                x=energy,
                y=bg_sub,
                line_width=2,
                color="#ff7f0e",
                legend_label="Background subtracted",
            )
            p.line(
                x=energy,
                y=scaled_mu,
                line_width=1,
                color="#2ca02c",
                line_dash="dashed",
                alpha=0.8,
                legend_label="Scale * mu(E)",
            )
        else:
            p.line(
                x=energy,
                y=full_curve,
                line_width=2,
                color="#ff7f0e",
                legend_label="Fit",
            )
            p.line(
                x=energy,
                y=background_curve,
                line_width=1,
                color="#2ca02c",
                line_dash="dashed",
                alpha=0.7,
                legend_label="Background",
            )
    if pre_lo is not None or pre_hi is not None:
        lo = pre_lo if pre_lo is not None else float(energy.min())
        hi = pre_hi if pre_hi is not None else float(energy.max())
        pre_box = BoxAnnotation(
            left=lo,
            right=hi,
            fill_alpha=0.15,
            fill_color="blue",
        )
        p.add_layout(pre_box)
    if post_lo is not None or post_hi is not None:
        lo = post_lo if post_lo is not None else float(energy.min())
        hi = post_hi if post_hi is not None else float(energy.max())
        post_box = BoxAnnotation(
            left=lo,
            right=hi,
            fill_alpha=0.15,
            fill_color="orange",
        )
        p.add_layout(post_box)
    box_select = BoxSelectTool(dimensions="width")
    p.add_tools(box_select)
    p.toolbar.active_drag = box_select

    def on_selection_change(attr: str, old: object, new: object) -> None:
        try:
            sel = getattr(source.selected, "indices", None)
            if sel is None or len(sel) == 0:
                return
            x_data = source.data["x"]
            x_vals = [float(x_data[i]) for i in sel]  # type: ignore[arg-type]
            x_min = min(x_vals)
            x_max = max(x_vals)
            if selection_applies_to == 0:
                pre_lo_widget.value = x_min
                pre_hi_widget.value = x_max
            else:
                post_lo_widget.value = x_min
                post_hi_widget.value = x_max
            source.selected.indices = []
        except Exception:
            pass

    source.on_change("selected", on_selection_change)
    p.legend.location = "top_right"
    return p


class NexafsWidget:
    """
    Panel + Bokeh NEXAFS widget: angle, pre/post edge, norm, view.

    Pre/post edge can be set via widgets or box-select on the plot.
    Use .show() to run with a server (required for box-select to update widgets).
    """

    def __init__(
        self,
        directory: NexafsDirectory,
        sample_name: str,
        formula: str = "C8H8",
        default_pre_edge: tuple[float | None, float | None] = (None, 280.0),
        default_post_edge: tuple[float | None, float | None] = (360.0, None),
    ) -> None:
        import panel as pn

        self._directory = directory
        self._sample_name = sample_name
        self._formula = formula
        self._dfs = directory.get_sample_dfs(
            sample_name,
            formula=formula,
            pre_edge=default_pre_edge,
            post_edge=default_post_edge,
        )
        if not self._dfs:
            self._angles = []
        else:
            self._angles = sorted(
                {float(df["Angle"].iloc[0]) for df in self._dfs}
            )
        energy_sample = None
        if self._dfs:
            energy_sample = self._dfs[0]["Energy"].to_numpy()
        e_min = float(energy_sample.min()) if energy_sample is not None else 0.0
        e_max = float(energy_sample.max()) if energy_sample is not None else 400.0
        pre_lo_def, pre_hi_def = default_pre_edge
        post_lo_def, post_hi_def = default_post_edge
        pre_lo_init = pre_lo_def if pre_lo_def is not None else e_min
        pre_hi_init = pre_hi_def if pre_hi_def is not None else 280.0
        post_lo_init = post_lo_def if post_lo_def is not None else 360.0
        post_hi_init = post_hi_def if post_hi_def is not None else e_max
        angle_options = (
            [(f"{a:.1f} deg", a) for a in self._angles]
            if self._angles
            else [("(no data)", None)]
        )
        angle_select = pn.widgets.Select(
            name="Angle",
            options=dict(angle_options),
            value=self._angles[0] if self._angles else None,
        )
        pre_lo_input = pn.widgets.FloatInput(
            name="Pre lo (eV)",
            value=pre_lo_init,
            step=1.0,
        )
        pre_hi_input = pn.widgets.FloatInput(
            name="Pre hi (eV)",
            value=pre_hi_init,
            step=1.0,
        )
        post_lo_input = pn.widgets.FloatInput(
            name="Post lo (eV)",
            value=post_lo_init,
            step=1.0,
        )
        post_hi_input = pn.widgets.FloatInput(
            name="Post hi (eV)",
            value=post_hi_init,
            step=1.0,
        )
        norm_options = {s.value: s for s in NormalizationScheme}
        norm_select = pn.widgets.Select(
            name="Norm",
            options=norm_options,
            value=NormalizationScheme.BARE_ATOM,
        )
        view_toggle = pn.widgets.Toggle(
            name="View: norm / bg-sub",
            value=False,
        )
        selection_applies = pn.widgets.RadioButtonGroup(
            name="Selection applies to",
            options={"Set pre edge": 0, "Set post edge": 1},
            value=0,
        )

        plot_deps: list[Any] = [
            angle_select,
            pre_lo_input,
            pre_hi_input,
            post_lo_input,
            post_hi_input,
            norm_select,
            view_toggle,
            selection_applies,
        ]
        @pn.depends(*plot_deps)
        def plot_pane(
            _angle: Any,
            _pre_lo: Any,
            _pre_hi: Any,
            _post_lo: Any,
            _post_hi: Any,
            _norm: Any,
            _view: Any,
            _sel: Any,
        ) -> Any:
            norm_scheme = (
                _norm
                if isinstance(_norm, NormalizationScheme)
                else NormalizationScheme.BARE_ATOM
            )
            return _build_bokeh_plot(
                angle=_angle,
                pre_lo=float(_pre_lo),  # type: ignore[arg-type]
                pre_hi=float(_pre_hi),  # type: ignore[arg-type]
                post_lo=float(_post_lo),  # type: ignore[arg-type]
                post_hi=float(_post_hi),  # type: ignore[arg-type]
                norm_scheme=norm_scheme,
                view_bg_subtracted=bool(_view),
                sample_name=sample_name,
                formula=formula,
                directory=directory,
                dfs=self._dfs,
                angles=self._angles,
                pre_lo_widget=pre_lo_input,
                pre_hi_widget=pre_hi_input,
                post_lo_widget=post_lo_input,
                post_hi_widget=post_hi_input,
                selection_applies_to=int(_sel) if _sel is not None else 0,
            )

        angle_label = pn.pane.Markdown("**Angle**")
        angle_block = pn.Column(angle_label, angle_select)
        pre_label = pn.pane.Markdown("**Pre edge (eV)**")
        pre_block = pn.Column(pre_label, pn.Row(pre_lo_input, pre_hi_input))
        post_label = pn.pane.Markdown("**Post edge (eV)**")
        post_block = pn.Column(post_label, pn.Row(post_lo_input, post_hi_input))
        norm_label = pn.pane.Markdown("**Norm**")
        norm_block = pn.Column(norm_label, norm_select)
        controls = pn.Column(
            angle_block,
            pre_block,
            post_block,
            norm_block,
            view_toggle,
            selection_applies,
            width=360,
        )
        self._layout = pn.Row(
            pn.pane.Bokeh(plot_pane, sizing_mode="stretch_both"),
            controls,
            sizing_mode="stretch_width",
        )

    def show(self) -> None:
        """Display the widget inline in Jupyter, or in browser otherwise."""
        try:
            from IPython.core.getipython import get_ipython
            if get_ipython() is not None:
                from IPython.display import display
                display(self._layout)
                return
        except ImportError:
            pass
        import panel as pn
        pn.serve(self._layout, threaded=True, show=True)
