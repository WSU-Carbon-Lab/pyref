"""ALS beamline utilities for generating run files and macros.

This module provides widgets and utilities for creating beamline scan scripts
for ALS beamline 11.0.1.2, including support for:
- Polarized X-ray Reflectivity (PXR) scans
- Energy scans (ESCAN)
- Resonant Soft X-ray Scattering (RSOXS) scans
"""

from pyref.beamline.base_widgets import (
    ALS_ExperimentWidget,
    ALS_MeasurementWidget,
    ALS_ScriptGenWidget,
    clean_script,
    pad_digits,
)
from pyref.beamline.beamline_scan_macros import (
    XRR_lineup,
    add_comment,
    add_prompt,
    analog_from_file,
    begin_macro,
    build_macro,
    clear_instruments,
    finish_line,
    jog_motor,
    move_motor,
    move_trajectory,
    piezo_toggle,
    relative_generic_scan,
    relative_photodiode_scan,
    run_dict_trajectory,
    run_I0,
    sample_photo,
    save_trajectory,
    set_DIO,
    set_instrument,
    set_motor,
    time_scan,
    windows_path,
)

__all__ = [
    "ALS_ExperimentWidget",
    "ALS_MeasurementWidget",
    "ALS_ScriptGenWidget",
    "XRR_lineup",
    "add_comment",
    "add_prompt",
    "analog_from_file",
    "begin_macro",
    "build_macro",
    "clean_script",
    "clear_instruments",
    "finish_line",
    "jog_motor",
    "move_motor",
    "move_trajectory",
    "pad_digits",
    "piezo_toggle",
    "relative_generic_scan",
    "relative_photodiode_scan",
    "run_I0",
    "run_dict_trajectory",
    "sample_photo",
    "save_trajectory",
    "set_DIO",
    "set_instrument",
    "set_motor",
    "time_scan",
    "windows_path",
]
