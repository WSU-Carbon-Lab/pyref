# Beamline Controls and Motor Naming

> Reference for: Xray Pro
> Load when: Figuring out a motor name, AI, etc.

---

## Overview

This is a detailed documentation describing what each motor is, the naming scheme, and the axes alignments with regard to the instrument and the lab reference frame

---

## Fundamental Naming

Controls are sent asynchronously through the BCS API. They are separated into core functionalities.

* **Analog Inputs (AI)**: These are each of the sensors at the beamline, including the Photodiode (photocurrent measured from the synchrotron beam hitting the photodiode sensor), Beamline Energy (from the mono), Ai 3 Izero (upstream beam intensity measured from a gold mesh), and others. Common AI channels are listed in the Analog Input Channels section below.
    | Command              | Description                                                            |
    |----------------------|------------------------------------------------------------------------|
    | acquire_data         | Acquires for `time` seconds, or `counts` counts.                       |
    | get_acquired         | Retrieve the average value from the most recent single-shot acquisition (see `start_acquire`). |
    | get_acquired_array   | Retrieve the array acquired from the most recent single-shot acquisition (see `start_acquire`). |
    | get_acquire_status   | Read the current acquisition state of the AI subsystem.                |
    | get_freerun          | Get freerun AI data for one or more channels.                          |
    | get_freerun_array    | Retrieve most recent AI freerun data.                                  |
    | list_ais             | Return the list of all AI channels that are defined on the server.     |
    | start_acquire        | Start Acquisition for either `time` or `counts`, whichever is non-zero.|
    | stop_acquire         | Stop acquisition in progress.                                          |

* **Digital Input Output (DIO)**: This contains toggles for each of the the toggleable components. This includes the shutter input, the in chamber light and camera trigger, and others.
    | Command     | Description                                                                                      |
    |-------------|--------------------------------------------------------------------------------------------------|
    | get_di      | Get digital input (DI) channel value                                                             |
    | list_dios   | Retrieves the complete list of digital input channel names defined on the server.                |
    | set_do      | Sets the digital output (DO) channel `chan` to `value`                                           |

* **Files and Folders**: This gets and sets the disk locations where the data is being saved. In all cases data is saved to the drive, this lets the user control where these files are being written to.
    | Command            | Description                                                                                                                     |
    |--------------------|---------------------------------------------------------------------------------------------------------------------------------|
    | get_folder_listing | Get lists of all files and folders in a location descended from “C:\Beamline Controls\BCS Setup Data”                           |
    | get_text_file      | Get contents of a text file (usually acquired data) from any file in a location descended from “C:\Beamline Controls\BCS Setup Data”. |

* **Global (State) Variables**: This contains state variables for the instrument. In most instances these should not be touched, changed, or set programmatically.
    | Command              | Description                                                                                     |
    |----------------------|-------------------------------------------------------------------------------------------------|
    | get_state_variable   | Get the value of the named BCS State Variable.                                                  |
    | list_state_variables | Get the complete list of state variables and their types (Boolean, Integer, String, Double).    |
    | set_state_variable   | Set the value of the named BCS State Variable.                                                  |

* **Instrument Data Acquisition**: This subsystem handles data acquisition from various instruments connected to the beamline, such as detectors and spectrometers. It provides commands to start, stop, and retrieve data from instrument acquisitions.
    | Command                        | Description                                                                                     |
    |--------------------------------|-------------------------------------------------------------------------------------------------|
    | get_instrument_acquired1d      | Retrieve data (1D array) from the most recent acquisition.                                      |
    | get_instrument_acquired2d      | Retrieve data (2D array) from the most recent acquisition.                                      |
    | get_instrument_acquired3d      | Retrieve data (3D array) from the most recent acquisition.                                     |
    | get_instrument_acquisition_info | Retrieve miscellaneous info about the instrument and acquisition: file_name, sensor temperature (if applicable), live time, and dead time. |
    | get_instrument_acquisition_status | Retrieve all available status bits from the instrument subsystem for the named instrument.    |
    | get_instrument_count_rates     | Retrieve instrument count rates.                                                                |
    | get_instrument_driver_status   | Returns the status of the BCS Instrument Driver for the named instrument.                     |
    | list_instruments               | Return the list of instruments that are defined on the server.                                 |
    | start_instrument_acquire       | Starts an instrument acquisition and waits for it to complete.                                  |
    | start_instrument_driver        | Starts the named instrument driver (does nothing if the driver is already running).            |
    | stop_instrument_acquire        | Stop acquisition on the named instrument (does nothing if the instrument is not acquiring).      |
    | stop_instrument_driver         | Stops the named instrument driver (does nothing if the driver is not running).                 |

* **Miscellaneous**: Utility commands for system status, panel images, and video feeds.
    | Command            | Description                                                      |
    |--------------------|------------------------------------------------------------------|
    | get_panel_image    | Send a current image of the LabVIEW panel, in jpg format.       |
    | get_subsystem_status | Returns the current status of all BCS subsystems on the server. |
    | get_video_image    | Return the most recent image from the named camera, in jpg format. |

* **Motors and Motion**: This subsystem controls all motorized components of the beamline. Motors can be moved individually or in coordinated trajectories. Motors can be enabled, disabled, homed, and their positions can be queried. Trajectories allow multiple motors to move in a coordinated fashion to predefined positions. Key motors on this instrument are broken down into Sample (X, Y, Z, Theta) and CCD (X, Y, Theta). These motors define the coordinate system of the beamline are are core to the alignment of the system.
    | Command              | Description                                                                                     |
    |----------------------|-------------------------------------------------------------------------------------------------|
    | at_preset            | Checks if the associated motor is at the preset position.                                       |
    | at_trajectory        | Checks if all requirements have been satisfied to be "At trajectory" (usually just means that each motor is at its trajectory goal). |
    | command_motor        | Command one or more motors.                                                                     |
    | disable_breakpoints | Disables the breakpoint output (output-on-position) of motor controller for the named motor.   |
    | disable_motor        | Disables the named motor.                                                                       |
    | enable_motor         | Enables the named motor.                                                                         |
    | get_flying_positions | Retrieve the locations of motor name that will trigger acquisitions in a flying scan.          |
    | get_motor            | Get information and status for the motors in motors.                                            |
    | get_motor_full       | Returns the complete state of the requested motors.                                              |
    | home_motor           | Home the motors in the input array, motors.                                                    |
    | list_motors          | Return the list of motors that are defined on the server.                                       |
    | list_presets         | Return array of motor preset positions.                                                          |
    | list_trajectories    | List all trajectories.                                                                          |
    | move_motor           | Command one or more motors to begin moves to supplied goals.                                   |
    | move_to_preset       | Move to preset positions.                                                                       |
    | move_to_trajectory   | Move to trajectory positions.                                                                   |
    | set_breakpoints      | Set Breakpoints for the named motor.                                                            |
    | set_motor_velocity   | Set motor speeds.                                                                               |
    | start_flying_scan    | Start a flying scan with the named motor.                                                       |
    | stop_motor           | Immediately issue stop command for the motors provided in motors.                               |

* **Scans**: This subsystem provides high-level scan routines for common experimental procedures. Scans coordinate motor movements with data acquisition. Various scan types are available for different experimental geometries and measurement types.
    | Command                    | Description                                                                                     |
    |----------------------------|-------------------------------------------------------------------------------------------------|
    | sc_alsu_mirror_vibration   | Setup the ALSU Mirror Vibration Scan. This scan moves one motor in a specific pattern and records the analog data. |
    | sc_auto_coll_single_axis_scan | This is VERY close to a Single Motor Scan.                                                   |
    | sc_barf_scan              | The Barf scan acquires from a spectrometer and then fits the ruby peaks to estimate the pressure. |
    | sc_from_file_scan         | This interface hides all the scan detail inside of a 'from file' scan file.                    |
    | sc_image_from_file_scan   | Image scan from file configuration.                                                              |
    | sc_image_one_motor_scan   | Image scan with one motor.                                                                      |
    | sc_image_time_scan        | Image scan over time.                                                                            |
    | sc_image_two_motor_scan   | Image scan with two motors.                                                                     |
    | sc_ltp_single_axis_scan   | LTP single axis scan.                                                                            |
    | sc_move_motor             | Setup this simple scan so that it will move a motor.                                            |
    | sc_move_trajectory        | Setup this simple scan so that it will activate a trajectory.                                  |
    | sc_powder_diffraction     | Powder diffraction scan.                                                                        |
    | sc_set_dio                | Setup scan to set digital output channels.                                                      |
    | sc_single_crystal_scan    | Setup the single motor scan that also records an image from a ccd camera instrument, with each data point. |
    | sc_single_motor_flying_scan | Setup a single motor flying scan.                                                              |
    | sc_single_motor_scan      | Single motor scan that moves one motor and acquires data at each position.                      |
    | sc_temperature_scan       | The Temperature Scan acquires data from a spectrometer and then with background and calibration data calculates the temperature. |
    | sc_time_scan              | Time scan that acquires data over time without motor movement.                                  |
    | sc_two_motor_scan         | Two motor scan that moves two motors and acquires data at each position combination.           |

* **Scan Management**: This subsystem provides commands to manage scan execution, including starting, stopping, and querying scan status. Scans can be executed synchronously or asynchronously.
    | Command              | Description                                                                                     |
    |----------------------|-------------------------------------------------------------------------------------------------|
    | current_scan_running | Returns the currently running 'integrated scan' run, or an empty string if none is running.   |
    | last_scan_run        | Returns the last 'integrated scan' run, or an empty string if none have run.                  |
    | scan_status          | Returns information about the 'integrated scan' system, namely the last scan (last_scan) run, the currently running scan, and the scanner status. |
    | stop_scan            | Immediately issue stop command for the currently running 'integrated scan'.                     |

## Coordinates and Axes

The beamline is configured in two regions, the chamber, and the region up stream from the chamber.

### Upstream Components

The beamline is set up with an X-ray beam incident into a spherical chamber from an Undulator source. Here is an explanation of the X-ray flight path into the chamber.

1. **EPU Source**: The beam is generated at an EPU (Elliptically Polarizing Undulator) source. Elliptically polarizing undulator (EPU) – 5 cm period; 165-1,800 eV [A. T. Young et al., J. Synchrotron Rad., 2002, 9, 270-274]
   - 4 rows of permanent magnets placed along axis of electron beam
   - 2 rows above the plane of the storage ring on either side of the beam; same for the 2 rows below
   - Linear polarization from 0° to 90° are available for energies above 160 eV
   - Using the fundamental output from the undulator, pure circularly polarized X-rays can be produced from 130-600 eV
   - For higher energies, 3rd and 5th harmonics must be used, but this leads to elliptical (P = 0.8 to 0.9) rather than circular polarization

2. **Horizontal Exit Slit**: After the EPU, the beam is selected using the Horizontal Exit Slit.

3. **M101 Monochromator**: From there the energy is selected at the M101 MONO (monochromator). The monochromator uses gratings (250 lines/mm for 150-550 eV, 500 lines/mm for energies above 550 eV) to select the desired X-ray energy.

4. **Mirror Focusing**: The beam is then focused through a series of mirrors, including the M103 Mirror, which provides focusing and beam conditioning.

5. **Higher Order Suppressor (HOS)**: The beam then hits the Higher Order Suppressor. This is a four-bounce reflective mirror assembly designed to add linear attenuation to the beam and remove higher order light making it past the mono. The incident angle of the mirrors can be changed between 4°-8°, or moved out of the beam path to let unfiltered light into the chamber.

6. **Upstream JJ Scatter Slits**: The beam then passes through the Upstream JJ scatter slits (approximately 1.5 m upstream of sample chamber). There are four motors for these slits:
   - Upstream JJ Vert Trans: vertical translational position
   - Upstream JJ Horz Trans: horizontal translational position
   - Upstream JJ Vert Aperture: vertical aperture size
   - Upstream JJ Horz Aperture: horizontal aperture size

7. **Shutter System and Beam Monitoring**: The beam then passes through the shutter system and Ai 3 Izero instrument for monitoring beam flux.
   - After the 6° deflection mirror, there is a gold mesh assembly that can be lowered into the beam and connected to a picoammeter to measure photoelectric current
   - This gold mesh is upstream of the HOS, so the spectrum of X-rays measured here is not suitable when doing scattering experiments at lower energies (e.g. near carbon dip) that need to use the HOS
   - Second gold mesh assembly directly downstream of the HOS, but before the first set of slits is used to monitor incident flux (Ai 3 Izero)

8. **Middle JJ Slits**: The beam then passes through the Middle JJ slits (approximately 0.6 m upstream of sample). These have the same 4 motors, but use the "Middle" prefix. These remove most of the remaining parasitic scattering from the first set of slits.

9. **In-Chamber JJ Slits**: Lastly, the beam passes through the In-Chamber JJ slits (approximately 0.2 m upstream of sample). Again these have the same 4 motors, but use the "In-Chamber" prefix. These provide final beam definition before the sample.

### In-Chamber Motors

The chamber is circular in nature. The beam is incident from upstream and focused onto the sample plate. The detector is then downstream from that.

#### Sample Coordinate System

The sample is driven by 4 primary motors: Sample X, Sample Y, Sample Z, and Sample Theta. Plates are rectangular in nature and are designed to be inserted with samples affixed to the top of the plate, and the top of the plate facing upwards towards the top of the chamber in the Lab reference frame.

**Sample Theta (θ)** dictates the orientation of the plate relative to the incident X-ray beam:
- **Sample θ = 90°**: The X-ray beam is normal to the sample plate surface (perpendicular incidence)
- 0°: Plate facing upwards towards the top of the chamber
- 180°: Plate facing downwards towards the back of the chamber
- Negative angles are valid: -90° = 270°

The sample coordinate system (Sample X, Y, Z) originates from the center of the sample plate:
- **Sample X**: Translation along the axis of rotation (colinear with Sample Theta axis)
  - Sample X + moves the plate inwards towards the motor controlling the motion
  - Sample X - moves the plate outwards towards the chamber door
- **Sample Y**: Translation perpendicular to the beam at Sample Theta = 0
  - Sample Y + moves the plate away from the beam
  - Sample Y - moves the plate towards the beam
- **Sample Z**: Translation perpendicular to the sample plate
  - At Sample Theta = 0°, Sample Z + moves the plate up (in Lab Frame)
  - At Sample Theta = 0°, Sample Z - moves the plate down (in Lab Frame)
  - **Note**: Changing Sample Z will also change the sample-detector distance (except when Sample θ is at 0°)

This forms a right-handed coordinate system. When Sample Theta rotates, the Sample Y and Z axes rotate with it, while Sample X remains colinear with the rotation axis.

#### Detector Coordinate System

![Geometry](coords.png "Optional title")

The Sample Theta axis of rotation is colinear with a goniometer arm that controls the CCD motors. **CCD** is the prefix given to all the motors that control the geometry of the instrument positioning arm. This arm rests on a goniometer facilitating rotation through approximately 120°.

The detector system consists of two components mounted on the same arm:
- **CCD detector**: Large area detector (2048 × 2048 pixels, 13.5 µm pixel size) for collecting scattering patterns
- **Photodiode**: 5 mm × 5 mm GaAs detector (Hamamatsu) for direct beam monitoring

**CCD Theta (θ)** controls the detector angle:
- At CCD Theta = 0°: The instrument is directly aligned with the incident beam and positioned to the left of the sample from the chamber door perspective
- CCD Theta + rotates the detector along the same direction as the Sample Theta motor
- Typical range: -25° to +160° for scattering measurements

**CCD Y** controls the sample-detector distance:
- CCD Y + moves the detector further away from the sample (increases sample-detector distance)
- CCD Y - moves the detector closer to the sample (decreases sample-detector distance)
- **Note**: Changing CCD Y will change the sample-detector distance

**CCD X** moves the detector colinear to the Sample X motor (horizontal position).

#### Photodiode Positioning

Both the CCD detector and photodiode are attached to the same arm and controlled by the same motors. To select the Photodiode position:
- Use the "Photodiode Far" trajectory/preset position
- This moves CCD Y = 100 mm, CCD X = 6 mm, CCD Theta close to 0°
- Alternatively, manually position Sample X to move the photodiode into line with the incident beam
- The photodiode is designed to be in the direct beam with CCD out of the way
- There is also a beamstop photodiode (1 mm × 3 mm Si photodiode from Advanced Photonics, accessible via "AI 6 BeamStop") for flux monitoring during scattering measurements

## Complete Motor Reference

This section provides a comprehensive list of all motors available on the beamline control system, organized by functional category.

### Sample Positioning Motors

These motors control the position and orientation of the sample within the chamber.

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| Sample X | Translation along the axis of rotation (colinear with Sample Theta axis). Positive moves toward motor, negative toward chamber door. | mm | Primary sample positioning axis |
| Sample Y | Translation perpendicular to beam at Sample Theta = 0. Positive moves away from beam, negative toward beam. | mm | Forms right-handed coordinate system with Sample X and Z |
| Sample Z | Translation perpendicular to sample plate. At Sample Theta = 0, positive moves up, negative moves down in lab frame. | mm | Height control relative to beam |
| Sample Theta | Rotation about the sample normal axis. 0° = plate facing up, 90° = normal to incident beam, 180° = facing down. | degrees | Primary sample orientation control |
| Sample Azimuthal Rotation | Additional rotational degree of freedom for sample orientation. | degrees | Secondary rotation axis |
| Sample Y Scaled | Scaled version of Sample Y position, used for specific scan geometries. | mm | Derived motor position |

### Detector Positioning Motors

These motors control the position and orientation of the CCD detector and photodiode.

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| CCD Theta | Rotation of detector arm on goniometer. 0° = aligned with incident beam, positioned left of sample from chamber door. Positive rotates in same direction as Sample Theta. Range: ~120°. | degrees | Primary detector angle control |
| CCD X | Translation colinear with Sample X motor. | mm | Detector horizontal position |
| CCD Y | Sample-detector distance. Positive moves detector away from sample, negative moves closer. | mm | Controls scattering geometry |
| Pollux CCD X | Alternative CCD X position for Pollux detector system. | mm | Pollux-specific positioning |
| Pollux CCD Y | Alternative CCD Y position for Pollux detector system. | mm | Pollux-specific positioning |
| T-2T | Theta-2Theta coupling for reflectivity measurements. Automatically couples Sample Theta and CCD Theta. | degrees | Reflectivity scan mode |
| Beam Stop | Position of beam stop to block direct beam from hitting detector. | mm | Protects detector from direct beam |

### Beamline Energy and Monochromator Controls

These motors control the X-ray energy selection and monochromator settings.

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| Beamline Energy | Current X-ray energy from monochromator. | eV | Primary energy control |
| Beamline Energy Goal | Target energy for energy scans. | eV | Used in energy scan routines |
| Mono Energy | Monochromator energy setting. | eV | Internal mono control |
| Mono 101 Grating | Grating selection on M101 monochromator. | unitless | Selects grating for energy range |
| Mono 101 Vessel | Monochromator vessel position. | mm | Vessel translation control |
| M101 Feedback | Feedback control for M101 monochromator. | unitless | Energy stabilization |
| M101 Horizontal Deflection | Horizontal beam deflection at M101. | mm | Beam steering |
| M101 Vertical Deflection | Vertical beam deflection at M101. | mm | Beam steering |

### EPU (Elliptically Polarizing Undulator) Controls

These motors control the X-ray source polarization and gap.

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| EPU Gap | Undulator gap setting. Controls fundamental energy and flux. | mm | Primary EPU control |
| EPU Z | Undulator Z position. | mm | Longitudinal position |
| EPU Polarization | Polarization setting. EPU=1 (or 0.9) for circular, EPU=100 for S-polarized, EPU=190 for P-polarized. | unitless | Polarization control |

### Mirror Controls

These motors control the focusing and steering mirrors.

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| M103 Yaw | M103 mirror yaw angle. | degrees | Mirror alignment |
| M103 Bend Up | M103 mirror upward bending. | mm | Mirror focusing |
| M103 Bend Down | M103 mirror downward bending. | mm | Mirror focusing |
| M121 Translation | M121 mirror translation position. | mm | Mirror positioning |

### Slit Controls

These motors control the entrance and exit slits that define the beam size and position.

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| Entrance Slit Width | Width of entrance slit to monochromator. | mm | Beam size control. Note: Also exists as "Entrance Slit width" (lowercase) in some configurations - these refer to the same motor. |
| Exit Slit Top | Top blade position of exit slit. | mm | Vertical beam definition |
| Exit Slit Bottom | Bottom blade position of exit slit. | mm | Vertical beam definition |
| Exit Slit Left | Left blade position of exit slit. | mm | Horizontal beam definition |
| Exit Slit Right | Right blade position of exit slit. | mm | Horizontal beam definition |
| Horizontal Exit Slit Size | Size of horizontal exit slit. | mm | Horizontal beam size |
| Horizontal Exit Slit Position | Position of horizontal exit slit. | mm | Horizontal beam position |
| Vertical Exit Slit Size | Size of vertical exit slit. | mm | Vertical beam size |
| Vertical Exit Slit Position | Position of vertical exit slit. | mm | Vertical beam position |
| Vertical Slit Position | Alternative vertical slit position control. | mm | Additional vertical control |
| Vertical Slit Size | Alternative vertical slit size control. | mm | Additional vertical control |
| Horizontal Slit Position | Alternative horizontal slit position control. | mm | Additional horizontal control |
| Horizontal Slit Size | Alternative horizontal slit size control. | mm | Additional horizontal control |

### Scatter Slit Controls (JJ Slits)

These motors control the Jaws-Jaws (JJ) scatter slits at three positions along the beamline. Each set has four motors: vertical and horizontal translation, and vertical and horizontal aperture.

#### Upstream JJ Slits
Located approximately 1.5 m upstream of sample chamber. First set of slits to define beam.

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| Upstream JJ Vert Trans | Vertical translational position of upstream slits. | mm | Vertical beam position |
| Upstream JJ Horz Trans | Horizontal translational position of upstream slits. | mm | Horizontal beam position |
| Upstream JJ Vert Aperture | Vertical aperture size of upstream slits. | mm | Vertical beam size |
| Upstream JJ Horz Aperture | Horizontal aperture size of upstream slits. | mm | Horizontal beam size |

#### Middle JJ Slits
Located approximately 0.6 m upstream of sample. Removes remaining parasitic scattering.

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| Middle JJ Vert Trans | Vertical translational position of middle slits. | mm | Vertical beam position |
| Middle JJ Horz Trans | Horizontal translational position of middle slits. | mm | Horizontal beam position |
| Middle JJ Vert Aperture | Vertical aperture size of middle slits. | mm | Vertical beam size |
| Middle JJ Horz Aperture | Horizontal aperture size of middle slits. | mm | Horizontal beam size |

#### In-Chamber JJ Slits
Located approximately 0.2 m upstream of sample. Final beam definition before sample.

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| In-Chamber JJ Vert Trans | Vertical translational position of in-chamber slits. | mm | Vertical beam position |
| In-Chamber JJ Horz Trans | Horizontal translational position of in-chamber slits. | mm | Horizontal beam position |
| In-Chamber JJ Vert Aperture | Vertical aperture size of in-chamber slits. | mm | Vertical beam size |
| In-Chamber JJ Horz Aperture | Horizontal aperture size of in-chamber slits. | mm | Horizontal beam size |

### Higher Order Suppressor

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| Higher Order Suppressor | Four-bounce mirror assembly position. Removes higher-order harmonics from monochromator. Incident angle adjustable between 4°-8°, or moved out of beam path. | mm | Critical for low-energy experiments |

### Shutter Controls

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| PiezoShutter Trans | Piezo-actuated shutter translation position. | mm | Fast shutter control |
| PZT Shutter | Piezoelectric shutter control. | unitless | Alternative shutter system |

### Temperature and Environmental Controls

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| Temperature Controller | Hot stage temperature setpoint. Calibration curve relates setpoint (K) to actual temperature (°C). | K | Sample temperature control |
| Coolstage | Cooled sample stage temperature control. | K | Low-temperature experiments |
| Camera Temp Setpoint | CCD camera temperature setpoint. Typically set to -45°C for operation. | °C | Detector cooling |

### Camera and Detector Controls

These are motor-like parameters that control camera settings and readout.

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| CCD Camera Shutter Inhibit | Inhibit signal for CCD camera shutter. | unitless | Shutter control |
| CCD Shutter Control | CCD camera shutter control signal. | unitless | Shutter control |
| Camera ROI X | Region of Interest X position. | pixels | Image cropping |
| Camera ROI Y | Region of Interest Y position. | pixels | Image cropping |
| Camera ROI Width | Region of Interest width. | pixels | Image size control |
| Camera ROI Height | Region of Interest height. | pixels | Image size control |
| Camera ROI X Bin | Binning factor in X direction. | unitless | Pixel binning for faster readout |
| Camera ROI Y Bin | Binning factor in Y direction. | unitless | Pixel binning for faster readout |

### Sample Rotation Motors

Additional rotational degrees of freedom for specialized sample holders.

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| SampleRot0 | Sample rotation axis 0. | degrees | Additional rotation |
| SampleRot1 | Sample rotation axis 1. | degrees | Additional rotation |
| SampleRot2 | Sample rotation axis 2. | degrees | Additional rotation |
| SampleRot3 | Sample rotation axis 3. | degrees | Additional rotation |
| SampleRot4 | Sample rotation axis 4. | degrees | Additional rotation |

### Multi-Channel Scaler (MCS) Axes

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| MCS_axis0 | Multi-channel scaler axis 0. | unitless | MCS control |
| MCS_axis1 | Multi-channel scaler axis 1. | unitless | MCS control |
| MCS_axis2 | Multi-channel scaler axis 2. | unitless | MCS control |
| MCS_axis3 | Multi-channel scaler axis 3. | unitless | MCS control |
| MCS_axis4 | Multi-channel scaler axis 4. | unitless | MCS control |

### Additional Controls

| Motor Name | Description | Units | Notes |
|------------|-------------|-------|-------|
| Sample Number | Sample identifier number. | unitless | Sample tracking |
| Piezo Vertical | Piezo vertical position control. | mm | Fine positioning |
| Piezo Horiz | Piezo horizontal position control. | mm | Fine positioning |
| AO 0 | Analog output channel 0. | V | General purpose analog output |
| AO 1 | Analog output channel 1. | V | General purpose analog output |
| OSP Adjustment | Optical sample position adjustment. | mm | Fine sample positioning |
| Diag 106 | Diagnostic element 106 position. | mm | Beam diagnostics |

## Analog Input (AI) Channels

Complete list of analog input channels available on the beamline for data acquisition and monitoring. Channels are numbered sequentially from 0-20.

| CH # | Channel Name | Description | Typical Units | Notes |
|------|--------------|-------------|---------------|-------|
| 0 | EPU Polarization | EPU polarization setting readback. | unitless | Readback of EPU polarization value (0.9 for circular, 100 for S-polarized, 190 for P-polarized). |
| 1 | Coolstage Temp C | Cooled sample stage temperature reading. | °C | Temperature of the coolstage sample holder. |
| 2 | CCD Temperature | CCD detector temperature reading. | °C | Temperature of the CCD detector. Typically operated at -45°C. |
| 3 | Beam Current | Storage ring beam current. | mA | Synchrotron beam current measurement. Negative values may indicate measurement direction or offset. |
| 4 | TEY signal | Total Electron Yield signal from sample. | A or V | Surface-sensitive detection mode for NEXAFS. Measures electron yield from sample surface. |
| 5 | Izero | Upstream beam intensity normalization signal. | A or V | Beam intensity measurement for normalization. Typically from gold mesh upstream of sample. |
| 6 | Photodiode | Main photodiode signal. Photocurrent from 5 mm × 5 mm GaAs photodiode (Hamamatsu) in direct beam path. | A or V | Primary beam intensity monitor. Use when CCD is out of beam path. |
| 7 | AI 0 | General purpose analog input channel 0. | V | Configurable analog input channel. |
| 8 | AI 3 Izero | Upstream beam intensity measured from gold mesh assembly (Ai 3). Located downstream of HOS, before first set of slits. | A or V | Used for flux normalization. Gold mesh can be lowered into beam and connected to picoammeter. |
| 9 | AI 5 | General purpose analog input channel 5. | V | Configurable analog input channel. |
| 10 | AI 6 BeamStop | Beamstop photodiode signal. Signal from 1 mm × 3 mm Si photodiode (Advanced Photonics) in beamstop. | A or V | Flux monitoring during scattering measurements. Can be used for absorption measurements while collecting scattering data. |
| 11 | AI 7 | General purpose analog input channel 7. | V | Configurable analog input channel. |
| 12 | Temperature Controller | Hot stage temperature controller readback. | K or °C | Temperature reading from the hot stage temperature controller. |
| 13 | PZT Shutter | Piezoelectric shutter position/status readback. | unitless | Status or position of the PZT (piezoelectric) shutter. |
| 14 | Pause Trigger | Pause trigger signal status. | unitless | Status of pause trigger for scan operations. |
| 15 | LV Memory | LabVIEW memory usage. | bytes | LabVIEW program memory usage indicator. |
| 16 | Deriv Photodiode | Derivative of photodiode signal. | V/s or A/s | Rate of change of photodiode signal. Useful for detecting beam fluctuations. |
| 17 | Time Stamp Error | Time synchronization error. | s | Error in time synchronization between systems. |
| 18 | Time Stamp Transmit Time | Time stamp transmission time. | s | Time taken to transmit time stamp data. |
| 19 | Time Stamp Server Time | Server time stamp. | s | Server-side time stamp value. |
| 20 | Camera Temp Setpoint | CCD camera temperature setpoint. | °C | Target temperature setting for CCD camera cooling. Typically set to -45°C for operation. |

Note: The complete list of available AI channels can be obtained using the `list_ais` command. Channel numbers and names are fixed as listed above. Some channels may show negative values due to measurement direction, signal inversion, or offset calibration.

## Digital Input/Output (DIO) Channels

Complete list of digital input/output channels available on the beamline for controlling components and reading status signals.

| Channel Name | Description | Type | Notes |
|--------------|------------|------|-------|
| Shutter Rev | Shutter reverse/reverse direction control. | DO | Controls shutter reverse operation. |
| Lightfiled Frame Loss | Light field frame loss detection signal. | DI | Indicates when light field frame is lost. |
| Nothing | Unused or placeholder DIO channel. | - | Reserved or unused channel. |
| Camera Scan | Camera scan trigger signal. | DO | Triggers camera acquisition during scans. |
| Shutter Output | Main beam shutter output control. Opens/closes X-ray beam. | DO | Critical safety component. Primary shutter control. Always verify shutter state. |
| Air Shutter Output | Air shutter output control. | DO | Controls air-operated shutter mechanism. |
| Light Output | Chamber illumination light control. | DO | Controls in-chamber lighting for sample viewing and alignment. |
| Beam Dumped | Beam dump status indicator. | DI | Indicates when beam is dumped (stopped). Read-only status signal. |
| PZT Shutter Status | Piezoelectric shutter status readback. | DI | Status of the PZT (piezoelectric) shutter position/state. |
| Do Pause Trigger | Pause trigger output control. | DO | Controls pause trigger for scan operations. |
| Trigger Pause Trigger | Trigger signal for pause trigger. | DO | Triggers the pause mechanism during scans. |
| Shutter Inhibit | Shutter inhibit signal. Prevents shutter from opening. | DO | Safety feature to inhibit shutter operation. |
| Trigger + Inhibit | Combined trigger and inhibit signal. | DO | Combined control signal for trigger and inhibit functions. |

Note: The complete list of available DIO channels can be obtained using the `list_dios` command. The list above includes all visible channels from the DIO Monitor interface. Additional channels may exist that are not shown in the visible portion of the list. Channel types (DI = Digital Input, DO = Digital Output) are inferred from channel names and typical beamline configurations.
