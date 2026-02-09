# Beamline Axis of Rotation Line Up

> Reference for: Xray Pro
> Load when: At the start of a beamtime experiment. When planning code associated with linning up and sample alignment.

---

## Overview

This is dedicated for lining up a sample to the incident beam on a given sample. This is a critical component to measuring anything at the beamline, especial NEXAFS spectrsocopy and Reflectivity. These measurements require preciese knowledge of the Sample Z postition, and Sample Theta position with respect to the incident beam. This accounts for beam drift, and motor drift thoughout an experiment.

First refference `references\beamline-controls.md` to understant the core api, motors, and coordinates.

## Alignement Algorithm

The alignement algorithm functions in three stages. First we algin the sensors to the direct beam, by aligning the CCD Theta motor positions to center the beam on the Photodiode, then we align the Sample Z and Sampe Theta positions. This sample alignment step is first done on a large detector, before happening on a more precise instrument.

### Instrument Alignement and Centering

1. Move the detector X motor such that we are centered on the photodiode instruemnt and not the Area detector.

2. Move the Sample Z motor down such that it the beam passes straight though the chamber onto the Photodiode.

3. Rock the CCD Theta motor arround zero to find the CCD Theta position associated with the maximum intensity of the Photodiode. Save this position for later use

4. Jog up by about 4 degrees to center the the secondary photodiode placed on the bottom of the detector. This is a slitted photidode to improve the precision of the alignment algorithm.

5. Rock the CCD Theta arround the 4 degree mark to find the CCD Theta position associated with the maximum intensity on teh AI 6 Beamstop AI. This is the name given to the slitten beamstop. Save this position for later use.

6. Move back to the large photodiode position saved in step 3. The Instrument is now ligned up.

### Sample Alginment

Sample alignement needs to be done for each new sample, and sometimes for each new energy. It may be the case that a quick alignement at the start is allways prefered to the possibility of a mis-aligned measurement.

What follows is a algorithmic iterative approach for sample alighnment.

1. Ensure the sample is below the beam allowing the full signal to hit the detector.

2. Move the sample up (+z) and record the instensity on the large photodiode.

3. Find the Sample Z position that cut's the beam intesnity in half. Alternatively, fit the signal to a sigmoid/error function, and determine it's position z0. Record this position for note keeping.

4. Move the sample theta motor to 1 degrees, and jog the sample theta motor by 2 degrees. We will be looking at the reflected intensity on the Photodiode, so ensure that we calculate the new position using the saves position from step 3 of the Instrument alignemtn.

5. Scan the sample theta position from about 0 - 2 degrees to find the angle that maximizes the reflected intensity as measured by the photodiode. A better measure of this is to find the centroid of the peak that this trace displays. This angle is a better guess of what 1 degrees really is.

6. Calculate the offset between 1 deg and this "True" angle. Save this offset for note keeping. Move to this offset and treat it as zero.,

7. Move back to step 1. Repeat until a convergence tolerence is reached.

### Fine grained sample alignement

This follows the same rules, but uses the slitted photodiode. To use this instrument, be sure that the offset angle for it found in the Instrument Alignment is properly accounted for.

## Algorithm Components

The alignment algorithm is broken down into reusable components. Each component includes a description, pseudo code, and Python implementation with checkpointing support.

### 1. Finding Maximum Value in Data

**Description:** Finds the position and value of the maximum intensity in a dataset. Used for peak finding in alignment scans.

**Pseudo Code:**
```
FUNCTION FIND_MAX(data_points):
    // data_points is array of (x, y) pairs
    SET max_y = -infinity
    SET max_x = undefined
    FOR each (x, y) IN data_points:
        IF y > max_y:
            SET max_y = y
            SET max_x = x
        END IF
    END FOR
    RETURN (max_x, max_y)
END FUNCTION
```

**Python Implementation:**
```python
def find_max(data_points):
    """
    Find the position and value of maximum intensity.

    Parameters:
    -----------
    data_points : list of tuples
        List of (x, y) pairs where x is position and y is intensity

    Returns:
    --------
    max_x : float
        Position of maximum intensity
    max_y : float
        Maximum intensity value

    Checkpoint:
    -----------
    Saves: max_x, max_y
    """
    max_y = float('-inf')
    max_x = None

    for x, y in data_points:
        if y > max_y:
            max_y = y
            max_x = x

    # Checkpoint: Save maximum values
    checkpoint = {
        'max_position': max_x,
        'max_intensity': max_y,
        'total_points': len(data_points)
    }
    # insert code to save checkpoint here

    return max_x, max_y
```

---

### 2. Linear Interpolation for Half-Maximum Finding

**Description:** Finds the position where intensity equals a target value (typically half-maximum) using linear interpolation between data points. More accurate than finding the nearest point.

**Pseudo Code:**
```
FUNCTION INTERPOLATE(x_values, y_values, target_y):
    // Linear interpolation to find x where y = target_y
    FOR i = 0 to LENGTH(x_values) - 2:
        IF (y_values[i] <= target_y AND target_y <= y_values[i+1]) OR
           (y_values[i] >= target_y AND target_y >= y_values[i+1]):
            SET slope = (y_values[i+1] - y_values[i]) / (x_values[i+1] - x_values[i])
            SET x_interp = x_values[i] + (target_y - y_values[i]) / slope
            RETURN x_interp
        END IF
    END FOR
    RETURN undefined
END FUNCTION
```

**Python Implementation:**
```python
def interpolate(x_values, y_values, target_y):
    """
    Find x position where y equals target_y using linear interpolation.

    Parameters:
    -----------
    x_values : list of float
        Position values
    y_values : list of float
        Intensity values corresponding to positions
    target_y : float
        Target intensity value to find

    Returns:
    --------
    x_interp : float or None
        Interpolated position where y = target_y, or None if not found

    Checkpoint:
    -----------
    Saves: target_y, x_interp, interpolation_method
    """
    for i in range(len(x_values) - 1):
        y1, y2 = y_values[i], y_values[i+1]
        x1, x2 = x_values[i], x_values[i+1]

        # Check if target_y is between y1 and y2
        if (y1 <= target_y <= y2) or (y1 >= target_y >= y2):
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) > 1e-10:  # Avoid division by zero
                x_interp = x1 + (target_y - y1) / slope
            else:
                x_interp = (x1 + x2) / 2  # Average if slope is zero

            # Checkpoint: Save interpolation result
            checkpoint = {
                'target_intensity': target_y,
                'interpolated_position': x_interp,
                'interpolation_method': 'linear',
                'bracket_indices': (i, i+1),
                'bracket_positions': (x1, x2),
                'bracket_intensities': (y1, y2)
            }
            # insert code to save checkpoint here

            return x_interp

    # Checkpoint: No interpolation found
    checkpoint = {
        'target_intensity': target_y,
        'interpolated_position': None,
        'interpolation_method': 'linear',
        'error': 'target_y not found in range'
    }
    # insert code to save checkpoint here

    return None
```

---

### 3. Finding Peak Centroid

**Description:** Calculates the centroid (center of mass) of a peak, which is more robust than finding the simple maximum. The centroid accounts for the shape of the peak distribution.

**Pseudo Code:**
```
FUNCTION FIND_CENTROID(x_values, y_values):
    // Calculate centroid (center of mass) of peak
    SET total_mass = SUM(y_values)
    SET weighted_sum = 0
    FOR i = 0 to LENGTH(x_values) - 1:
        SET weighted_sum = weighted_sum + x_values[i] * y_values[i]
    END FOR
    SET centroid_x = weighted_sum / total_mass

    // Find peak maximum for reference
    SET (max_x, max_y) = FIND_MAX(zip(x_values, y_values))

    RETURN (centroid_x, max_x)
END FUNCTION
```

**Python Implementation:**
```python
def find_centroid(x_values, y_values):
    """
    Calculate centroid (center of mass) of intensity peak.

    Parameters:
    -----------
    x_values : list of float
        Position values
    y_values : list of float
        Intensity values

    Returns:
    --------
    centroid_x : float
        Centroid position (weighted average)
    max_x : float
        Position of maximum intensity (for reference)

    Checkpoint:
    -----------
    Saves: centroid_x, max_x, max_intensity, total_mass
    """
    total_mass = sum(y_values)

    if total_mass == 0:
        # Fallback to maximum if no signal
        max_idx = y_values.index(max(y_values))
        centroid_x = x_values[max_idx]
        max_x = x_values[max_idx]
    else:
        weighted_sum = sum(x * y for x, y in zip(x_values, y_values))
        centroid_x = weighted_sum / total_mass

        # Also find maximum for reference
        max_idx = y_values.index(max(y_values))
        max_x = x_values[max_idx]

    # Checkpoint: Save centroid calculation
    checkpoint = {
        'centroid_position': centroid_x,
        'max_position': max_x,
        'max_intensity': max(y_values),
        'total_mass': total_mass,
        'num_points': len(x_values),
        'centroid_max_difference': abs(centroid_x - max_x)
    }
    # insert code to save checkpoint here

    return centroid_x, max_x
```

---

### 4. Scanning Theta Range and Measuring Intensity

**Description:** Scans a range of theta (angle) positions and measures intensity at each position. Used for finding optimal detector or sample angles.

**Pseudo Code:**
```
FUNCTION SCAN_THETA_RANGE(theta_start, theta_end, theta_step, motor_name, ai_channel):
    SET positions = []
    SET intensities = []
    SET current_theta = theta_start

    WHILE current_theta <= theta_end:
        SET motor_name = current_theta
        WAIT for motor movement
        intensity = READ ai_channel signal
        APPEND current_theta to positions
        APPEND intensity to intensities
        SET current_theta = current_theta + theta_step
    END WHILE

    RETURN (positions, intensities)
END FUNCTION
```

**Python Implementation:**
```python
def scan_theta_range(theta_start, theta_end, theta_step, motor_name, ai_channel):
    """
    Scan a range of theta positions and measure intensity.

    Parameters:
    -----------
    theta_start : float
        Starting theta position (degrees)
    theta_end : float
        Ending theta position (degrees)
    theta_step : float
        Step size for scan (degrees)
    motor_name : str
        Name of motor to move (e.g., "CCD Theta", "Sample Theta")
    ai_channel : str
        Name of AI channel to read (e.g., "Photodiode", "AI 6 BeamStop")

    Returns:
    --------
    positions : list of float
        Theta positions scanned
    intensities : list of float
        Intensity values at each position

    Checkpoint:
    -----------
    Saves: scan_range, step_size, num_points, max_intensity, max_position
    """
    positions = []
    intensities = []

    current_theta = theta_start
    while current_theta <= theta_end:
        # insert code to move motor_name to current_theta here
        # insert code to wait for motor movement here
        intensity = 0  # insert code to read ai_channel signal here

        positions.append(current_theta)
        intensities.append(intensity)
        current_theta += theta_step

    # Checkpoint: Save scan results
    max_intensity = max(intensities) if intensities else 0
    max_idx = intensities.index(max_intensity) if intensities else 0
    max_position = positions[max_idx] if positions else None

    checkpoint = {
        'scan_type': 'theta_range',
        'motor_name': motor_name,
        'ai_channel': ai_channel,
        'theta_start': theta_start,
        'theta_end': theta_end,
        'theta_step': theta_step,
        'num_points': len(positions),
        'max_intensity': max_intensity,
        'max_position': max_position,
        'positions': positions,
        'intensities': intensities
    }
    # insert code to save checkpoint here

    return positions, intensities
```

---

### 5. Scanning Z Range and Measuring Intensity

**Description:** Scans a range of Z positions and measures intensity at each position. Used for finding the half-maximum position for sample alignment.

**Pseudo Code:**
```
FUNCTION SCAN_Z_RANGE(z_start, z_end, z_step, ai_channel):
    SET positions = []
    SET intensities = []
    SET current_z = z_start

    WHILE current_z <= z_end:
        SET Sample_Z = current_z
        WAIT for motor movement
        intensity = READ ai_channel signal
        APPEND current_z to positions
        APPEND intensity to intensities
        SET current_z = current_z + z_step
    END WHILE

    RETURN (positions, intensities)
END FUNCTION
```

**Python Implementation:**
```python
def scan_z_range(z_start, z_end, z_step, ai_channel):
    """
    Scan a range of Z positions and measure intensity.

    Parameters:
    -----------
    z_start : float
        Starting Z position (mm)
    z_end : float
        Ending Z position (mm)
    z_step : float
        Step size for scan (mm)
    ai_channel : str
        Name of AI channel to read (e.g., "Photodiode", "AI 6 BeamStop")

    Returns:
    --------
    positions : list of float
        Z positions scanned
    intensities : list of float
        Intensity values at each position

    Checkpoint:
    -----------
    Saves: scan_range, step_size, num_points, max_intensity, max_position
    """
    positions = []
    intensities = []

    current_z = z_start
    while current_z <= z_end:
        # insert code to move Sample Z to current_z here
        # insert code to wait for motor movement here
        intensity = 0  # insert code to read ai_channel signal here

        positions.append(current_z)
        intensities.append(intensity)
        current_z += z_step

    # Checkpoint: Save scan results
    max_intensity = max(intensities) if intensities else 0
    max_idx = intensities.index(max_intensity) if intensities else 0
    max_position = positions[max_idx] if positions else None

    checkpoint = {
        'scan_type': 'z_range',
        'ai_channel': ai_channel,
        'z_start': z_start,
        'z_end': z_end,
        'z_step': z_step,
        'num_points': len(positions),
        'max_intensity': max_intensity,
        'max_position': max_position,
        'positions': positions,
        'intensities': intensities
    }
    # insert code to save checkpoint here

    return positions, intensities
```

---

### 6. Finding Half-Maximum Z Position

**Description:** Finds the Z position where intensity is half of the maximum. This position corresponds to the beam edge and is used for precise sample alignment.

**Pseudo Code:**
```
FUNCTION FIND_Z_HALF_MAX(z_positions, intensities):
    SET max_intensity = MAX(intensities)
    SET half_max = max_intensity / 2

    // Option A: Linear interpolation
    SET z_center = INTERPOLATE(z_positions, intensities, half_max)

    // Option B: If interpolation fails, use maximum position
    IF z_center is undefined:
        SET max_idx = INDEX_OF_MAX(intensities)
        SET z_center = z_positions[max_idx]
    END IF

    RETURN z_center
END FUNCTION
```

**Python Implementation:**
```python
def find_z_half_max(z_positions, intensities):
    """
    Find Z position where intensity is half of maximum.

    Parameters:
    -----------
    z_positions : list of float
        Z positions from scan
    intensities : list of float
        Intensity values from scan

    Returns:
    --------
    z_center : float
        Z position at half-maximum intensity

    Checkpoint:
    -----------
    Saves: z_center, max_intensity, half_max, method_used
    """
    if not intensities:
        return None

    max_intensity = max(intensities)
    half_max = max_intensity / 2.0

    # Try linear interpolation
    z_center = interpolate(z_positions, intensities, half_max)
    method_used = 'interpolation'

    # Fallback to maximum position if interpolation fails
    if z_center is None:
        max_idx = intensities.index(max_intensity)
        z_center = z_positions[max_idx]
        method_used = 'maximum_fallback'

    # Checkpoint: Save half-maximum finding result
    checkpoint = {
        'z_center': z_center,
        'max_intensity': max_intensity,
        'half_max_intensity': half_max,
        'method_used': method_used,
        'num_points': len(z_positions),
        'z_range': (min(z_positions), max(z_positions))
    }
    # insert code to save checkpoint here

    return z_center
```

---

### 7. Finding Optimal Theta Position

**Description:** Finds the optimal theta position by scanning a range and calculating the centroid of the reflected intensity peak. More accurate than finding the simple maximum.

**Pseudo Code:**
```
FUNCTION FIND_THETA_OPTIMAL(theta_start, theta_end, theta_step,
                            base_theta, theta_offset, theta_photodiode, ai_channel):
    SET theta_positions = []
    SET reflected_intensities = []
    SET current_theta = theta_start

    WHILE current_theta <= theta_end:
        SET sample_theta = base_theta + theta_offset + current_theta
        SET expected_ccd_theta = theta_photodiode + 2 * sample_theta

        SET Sample_Theta = sample_theta
        SET CCD_Theta = expected_ccd_theta
        WAIT for motor movements
        intensity = READ ai_channel signal

        APPEND sample_theta to theta_positions
        APPEND intensity to reflected_intensities
        SET current_theta = current_theta + theta_step
    END WHILE

    SET (theta_optimal, max_theta) = FIND_CENTROID(theta_positions, reflected_intensities)

    RETURN theta_optimal
END FUNCTION
```

**Python Implementation:**
```python
def find_theta_optimal(theta_start, theta_end, theta_step, base_theta,
                       theta_offset, theta_photodiode, ai_channel):
    """
    Find optimal theta position by scanning and finding peak centroid.

    Parameters:
    -----------
    theta_start : float
        Starting theta offset (degrees)
    theta_end : float
        Ending theta offset (degrees)
    theta_step : float
        Step size (degrees)
    base_theta : float
        Base theta position (typically 1.0 degrees)
    theta_offset : float
        Current theta offset to apply
    theta_photodiode : float
        Optimal CCD Theta for photodiode (from instrument alignment)
    ai_channel : str
        AI channel to read (e.g., "Photodiode", "AI 6 BeamStop")

    Returns:
    --------
    theta_optimal : float
        Optimal theta position (centroid of peak)

    Checkpoint:
    -----------
    Saves: theta_optimal, scan_data, centroid_info
    """
    theta_positions = []
    reflected_intensities = []

    current_theta = theta_start
    while current_theta <= theta_end:
        sample_theta = base_theta + theta_offset + current_theta
        # Calculate expected photodiode position for reflection (2*theta geometry)
        expected_ccd_theta = theta_photodiode + 2 * sample_theta

        # insert code to move Sample Theta to sample_theta here
        # insert code to move CCD Theta to expected_ccd_theta here
        # insert code to wait for motor movements here
        intensity = 0  # insert code to read ai_channel signal here

        theta_positions.append(sample_theta)
        reflected_intensities.append(intensity)
        current_theta += theta_step

    # Find centroid of peak
    theta_optimal, max_theta = find_centroid(theta_positions, reflected_intensities)

    # Checkpoint: Save theta optimization result
    checkpoint = {
        'theta_optimal': theta_optimal,
        'max_theta': max_theta,
        'base_theta': base_theta,
        'theta_offset': theta_offset,
        'theta_photodiode': theta_photodiode,
        'scan_range': (theta_start, theta_end),
        'theta_step': theta_step,
        'max_intensity': max(reflected_intensities) if reflected_intensities else 0,
        'positions': theta_positions,
        'intensities': reflected_intensities
    }
    # insert code to save checkpoint here

    return theta_optimal
```

---

### 8. Checking Convergence

**Description:** Checks if the alignment has converged by comparing current values to previous values. Used to determine when to stop iterative alignment.

**Pseudo Code:**
```
FUNCTION CHECK_CONVERGENCE(current_value, previous_value, tolerance):
    IF previous_value is undefined:
        RETURN (converged=False, delta=None)
    END IF

    SET delta = ABS(current_value - previous_value)
    SET converged = (delta < tolerance)

    RETURN (converged, delta)
END FUNCTION
```

**Python Implementation:**
```python
def check_convergence(current_value, previous_value, tolerance):
    """
    Check if alignment has converged.

    Parameters:
    -----------
    current_value : float
        Current measured value
    previous_value : float or None
        Previous measured value (None for first iteration)
    tolerance : float
        Convergence tolerance

    Returns:
    --------
    converged : bool
        True if converged, False otherwise
    delta : float or None
        Difference between current and previous values

    Checkpoint:
    -----------
    Saves: converged, delta, tolerance, iteration_info
    """
    if previous_value is None:
        delta = None
        converged = False
    else:
        delta = abs(current_value - previous_value)
        converged = delta < tolerance

    # Checkpoint: Save convergence check
    checkpoint = {
        'converged': converged,
        'delta': delta,
        'tolerance': tolerance,
        'current_value': current_value,
        'previous_value': previous_value
    }
    # insert code to save checkpoint here

    return converged, delta
```

---

### 9. Positioning Detector for Photodiode

**Description:** Positions the detector to use the photodiode instead of the area detector. This is the first step in instrument alignment.

**Pseudo Code:**
```
FUNCTION POSITION_DETECTOR_PHOTODIODE():
    SET CCD_X = photodiode_center_position
    SET Sample_Z = beam_path_clear_position
    WAIT for motor movements
END FUNCTION
```

**Python Implementation:**
```python
def position_detector_photodiode(photodiode_x_position=6.0, beam_clear_z=-2.0):
    """
    Position detector to use photodiode and clear beam path.

    Parameters:
    -----------
    photodiode_x_position : float
        CCD X position for photodiode (mm, default: 6.0)
    beam_clear_z : float
        Sample Z position to clear beam path (mm, default: -2.0)

    Checkpoint:
    -----------
    Saves: ccd_x_position, sample_z_position
    """
    # insert code to move CCD X to photodiode_x_position here
    # insert code to move Sample Z to beam_clear_z here
    # insert code to wait for motor movements here

    # Checkpoint: Save positioning
    checkpoint = {
        'step': 'position_detector_photodiode',
        'ccd_x_position': photodiode_x_position,
        'sample_z_position': beam_clear_z
    }
    # insert code to save checkpoint here
```

---

### 10. Finding Optimal Photodiode Theta

**Description:** Finds the optimal CCD Theta position that maximizes intensity on the main photodiode. This is the reference position for all subsequent alignments.

**Pseudo Code:**
```
FUNCTION FIND_PHOTODIODE_THETA(theta_range_center=0, theta_range_width=2):
    SET theta_range = [theta_range_center - theta_range_width,
                       theta_range_center + theta_range_width]
    SET (positions, intensities) = SCAN_THETA_RANGE(
        theta_range[0], theta_range[1], step=0.5,
        motor="CCD Theta", ai_channel="Photodiode"
    )
    SET (theta_optimal, max_intensity) = FIND_MAX(zip(positions, intensities))

    RETURN theta_optimal
END FUNCTION
```

**Python Implementation:**
```python
def find_photodiode_theta(theta_range_center=0.0, theta_range_width=2.0, theta_step=0.5):
    """
    Find optimal CCD Theta position for main photodiode.

    Parameters:
    -----------
    theta_range_center : float
        Center of theta scan range (degrees, default: 0.0)
    theta_range_width : float
        Width of theta scan range (degrees, default: 2.0)
    theta_step : float
        Step size for scan (degrees, default: 0.5)

    Returns:
    --------
    theta_photodiode : float
        Optimal CCD Theta position for photodiode

    Checkpoint:
    -----------
    Saves: theta_photodiode, max_intensity, scan_data
    """
    theta_start = theta_range_center - theta_range_width
    theta_end = theta_range_center + theta_range_width

    positions, intensities = scan_theta_range(
        theta_start, theta_end, theta_step,
        motor_name="CCD Theta",
        ai_channel="Photodiode"
    )

    theta_photodiode, max_intensity = find_max(list(zip(positions, intensities)))

    # Move to optimal position
    # insert code to move CCD Theta to theta_photodiode here
    # insert code to wait for motor movement here

    # Checkpoint: Save photodiode theta result
    checkpoint = {
        'step': 'find_photodiode_theta',
        'theta_photodiode': theta_photodiode,
        'max_intensity': max_intensity,
        'scan_center': theta_range_center,
        'scan_width': theta_range_width,
        'scan_step': theta_step
    }
    # insert code to save checkpoint here

    return theta_photodiode
```

---

### 11. Finding Optimal Beamstop Theta

**Description:** Finds the optimal CCD Theta position that maximizes intensity on the slitted beamstop photodiode (AI 6 BeamStop). This provides higher precision alignment.

**Pseudo Code:**
```
FUNCTION FIND_BEAMSTOP_THETA(theta_photodiode, offset_estimate=4.0):
    SET theta_center = theta_photodiode + offset_estimate
    SET theta_range = [theta_center - 1.0, theta_center + 1.0]
    SET (positions, intensities) = SCAN_THETA_RANGE(
        theta_range[0], theta_range[1], step=0.5,
        motor="CCD Theta", ai_channel="AI 6 BeamStop"
    )
    SET (theta_optimal, max_intensity) = FIND_MAX(zip(positions, intensities))

    RETURN theta_optimal
END FUNCTION
```

**Python Implementation:**
```python
def find_beamstop_theta(theta_photodiode, offset_estimate=4.0,
                       theta_range_width=1.0, theta_step=0.5):
    """
    Find optimal CCD Theta position for beamstop photodiode.

    Parameters:
    -----------
    theta_photodiode : float
        Optimal photodiode theta position (from find_photodiode_theta)
    offset_estimate : float
        Estimated offset from photodiode position (degrees, default: 4.0)
    theta_range_width : float
        Width of scan range around estimate (degrees, default: 1.0)
    theta_step : float
        Step size for scan (degrees, default: 0.5)

    Returns:
    --------
    theta_beamstop : float
        Optimal CCD Theta position for beamstop

    Checkpoint:
    -----------
    Saves: theta_beamstop, max_intensity, offset_from_photodiode
    """
    theta_center = theta_photodiode + offset_estimate
    theta_start = theta_center - theta_range_width
    theta_end = theta_center + theta_range_width

    positions, intensities = scan_theta_range(
        theta_start, theta_end, theta_step,
        motor_name="CCD Theta",
        ai_channel="AI 6 BeamStop"
    )

    theta_beamstop, max_intensity = find_max(list(zip(positions, intensities)))

    # Checkpoint: Save beamstop theta result
    offset_from_photodiode = theta_beamstop - theta_photodiode
    checkpoint = {
        'step': 'find_beamstop_theta',
        'theta_beamstop': theta_beamstop,
        'theta_photodiode': theta_photodiode,
        'offset_from_photodiode': offset_from_photodiode,
        'max_intensity': max_intensity,
        'scan_center': theta_center,
        'scan_width': theta_range_width,
        'scan_step': theta_step
    }
    # insert code to save checkpoint here

    return theta_beamstop
```

---

### 12. Complete Instrument Alignment

**Description:** Complete instrument alignment procedure that finds optimal positions for both photodiode and beamstop detectors. This must be done before sample alignment.

**Pseudo Code:**
```
FUNCTION INSTRUMENT_ALIGNMENT():
    POSITION_DETECTOR_PHOTODIODE()
    SET theta_photodiode = FIND_PHOTODIODE_THETA()
    SET theta_beamstop = FIND_BEAMSTOP_THETA(theta_photodiode)
    SET CCD_Theta = theta_photodiode  // Return to photodiode position
    RETURN (theta_photodiode, theta_beamstop)
END FUNCTION
```

**Python Implementation:**
```python
def instrument_alignment():
    """
    Complete instrument alignment to find optimal detector positions.

    Returns:
    --------
    theta_photodiode : float
        Optimal CCD Theta for main photodiode
    theta_beamstop : float
        Optimal CCD Theta for beamstop photodiode

    Checkpoint:
    -----------
    Saves: Complete alignment results, all intermediate steps
    """
    # Step 1-2: Position detector
    position_detector_photodiode()

    # Step 3: Find optimal photodiode theta
    theta_photodiode = find_photodiode_theta()

    # Step 4-5: Find optimal beamstop theta
    theta_beamstop = find_beamstop_theta(theta_photodiode)

    # Step 6: Return to photodiode position
    # insert code to move CCD Theta to theta_photodiode here
    # insert code to wait for motor movement here

    # Checkpoint: Save complete instrument alignment
    checkpoint = {
        'step': 'instrument_alignment_complete',
        'theta_photodiode': theta_photodiode,
        'theta_beamstop': theta_beamstop,
        'offset_photodiode_to_beamstop': theta_beamstop - theta_photodiode
    }
    # insert code to save checkpoint here

    return theta_photodiode, theta_beamstop
```

---

### 13. Single Iteration of Sample Alignment

**Description:** Performs one iteration of sample alignment, finding optimal Z and Theta positions. This is the core of the iterative alignment loop.

**Pseudo Code:**
```
FUNCTION SAMPLE_ALIGNMENT_ITERATION(theta_photodiode, theta_offset,
                                    z_start, z_end, z_step):
    // Find Z center
    SET (z_positions, intensities) = SCAN_Z_RANGE(z_start, z_end, z_step, "Photodiode")
    SET z_center = FIND_Z_HALF_MAX(z_positions, intensities)

    // Find optimal Theta
    SET theta_optimal = FIND_THETA_OPTIMAL(
        theta_start=0, theta_end=2, theta_step=0.1,
        base_theta=1.0, theta_offset=theta_offset,
        theta_photodiode=theta_photodiode, ai_channel="Photodiode"
    )
    SET theta_offset_new = theta_optimal - 1.0

    RETURN (z_center, theta_offset_new)
END FUNCTION
```

**Python Implementation:**
```python
def sample_alignment_iteration(theta_photodiode, theta_offset,
                               z_start=0.0, z_end=5.0, z_step=0.05):
    """
    Perform one iteration of sample alignment.

    Parameters:
    -----------
    theta_photodiode : float
        Optimal CCD Theta for photodiode
    theta_offset : float
        Current theta offset
    z_start : float
        Starting Z position for scan (mm)
    z_end : float
        Ending Z position for scan (mm)
    z_step : float
        Step size for Z scan (mm)

    Returns:
    --------
    z_center : float
        Optimal Z position (half-maximum)
    theta_offset_new : float
        New theta offset

    Checkpoint:
    -----------
    Saves: z_center, theta_offset_new, iteration_data
    """
    # Ensure sample is below beam
    # insert code to move Sample Z to below_beam_position here

    # Find Z center (half-maximum)
    z_positions, intensities = scan_z_range(z_start, z_end, z_step, "Photodiode")
    z_center = find_z_half_max(z_positions, intensities)

    # Find optimal Theta
    theta_optimal = find_theta_optimal(
        theta_start=0.0, theta_end=2.0, theta_step=0.1,
        base_theta=1.0, theta_offset=theta_offset,
        theta_photodiode=theta_photodiode, ai_channel="Photodiode"
    )
    theta_offset_new = theta_optimal - 1.0

    # Apply new theta offset
    # insert code to move Sample Theta to theta_offset_new here
    # insert code to wait for motor movement here

    # Checkpoint: Save iteration results
    checkpoint = {
        'step': 'sample_alignment_iteration',
        'z_center': z_center,
        'theta_offset_old': theta_offset,
        'theta_offset_new': theta_offset_new,
        'theta_optimal': theta_optimal,
        'z_scan_range': (z_start, z_end),
        'z_step': z_step
    }
    # insert code to save checkpoint here

    return z_center, theta_offset_new
```

---

### 14. Complete Sample Alignment

**Description:** Iterative sample alignment that converges to optimal Z and Theta positions. Continues until convergence criteria are met.

**Pseudo Code:**
```
FUNCTION SAMPLE_ALIGNMENT(theta_photodiode, convergence_tolerance=0.01, max_iterations=10):
    SET iteration = 0
    SET theta_offset = 0
    SET previous_z_center = undefined
    SET previous_theta_offset = undefined

    WHILE iteration < max_iterations:
        SET (z_center, theta_offset_new) = SAMPLE_ALIGNMENT_ITERATION(
            theta_photodiode, theta_offset
        )

        // Check Z convergence
        SET (z_converged, z_delta) = CHECK_CONVERGENCE(
            z_center, previous_z_center, convergence_tolerance
        )

        // Check Theta convergence
        SET (theta_converged, theta_delta) = CHECK_CONVERGENCE(
            theta_offset_new, previous_theta_offset, convergence_tolerance
        )

        IF z_converged AND theta_converged AND iteration > 0:
            BREAK
        END IF

        SET previous_z_center = z_center
        SET previous_theta_offset = theta_offset_new
        SET theta_offset = theta_offset_new
        SET iteration = iteration + 1
    END WHILE

    RETURN (z_center, theta_offset)
END FUNCTION
```

**Python Implementation:**
```python
def sample_alignment(theta_photodiode, convergence_tolerance=0.01, max_iterations=10):
    """
    Complete iterative sample alignment.

    Parameters:
    -----------
    theta_photodiode : float
        Optimal CCD Theta for photodiode (from instrument alignment)
    convergence_tolerance : float
        Convergence tolerance (mm or degrees, default: 0.01)
    max_iterations : int
        Maximum number of iterations (default: 10)

    Returns:
    --------
    z_center : float
        Final optimal Z position
    theta_offset : float
        Final theta offset

    Checkpoint:
    -----------
    Saves: Final results, all iteration data, convergence info
    """
    iteration = 0
    theta_offset = 0.0
    previous_z_center = None
    previous_theta_offset = None

    while iteration < max_iterations:
        z_center, theta_offset_new = sample_alignment_iteration(
            theta_photodiode, theta_offset
        )

        # Check Z convergence
        z_converged, z_delta = check_convergence(
            z_center, previous_z_center, convergence_tolerance
        )

        # Check Theta convergence
        theta_converged, theta_delta = check_convergence(
            theta_offset_new, previous_theta_offset, convergence_tolerance
        )

        # Checkpoint: Save iteration info
        checkpoint = {
            'step': 'sample_alignment_iteration',
            'iteration': iteration,
            'z_center': z_center,
            'theta_offset': theta_offset_new,
            'z_converged': z_converged,
            'z_delta': z_delta,
            'theta_converged': theta_converged,
            'theta_delta': theta_delta,
            'convergence_tolerance': convergence_tolerance
        }
        # insert code to save checkpoint here

        # Break if converged (after at least one iteration)
        if z_converged and theta_converged and iteration > 0:
            break

        previous_z_center = z_center
        previous_theta_offset = theta_offset_new
        theta_offset = theta_offset_new
        iteration += 1

    # Checkpoint: Save final alignment results
    final_checkpoint = {
        'step': 'sample_alignment_complete',
        'final_z_center': z_center,
        'final_theta_offset': theta_offset,
        'total_iterations': iteration,
        'converged': (iteration < max_iterations),
        'convergence_tolerance': convergence_tolerance
    }
    # insert code to save checkpoint here

    return z_center, theta_offset
```

---

### 15. Fine-Grained Sample Alignment

**Description:** Same as sample alignment but uses the slitted beamstop photodiode for higher precision. Uses tighter convergence tolerance and smaller step sizes.

**Pseudo Code:**
```
FUNCTION FINE_GRAINED_SAMPLE_ALIGNMENT(theta_beamstop, convergence_tolerance=0.005, max_iterations=10):
    // Same as SAMPLE_ALIGNMENT but:
    // - Use theta_beamstop instead of theta_photodiode
    // - Use "AI 6 BeamStop" instead of "Photodiode"
    // - Use smaller z_step (0.02 instead of 0.05)
    // - Use smaller theta_step (0.05 instead of 0.1)
    // - Use tighter convergence_tolerance
END FUNCTION
```

**Python Implementation:**
```python
def fine_grained_sample_alignment(theta_beamstop, convergence_tolerance=0.005, max_iterations=10):
    """
    Fine-grained sample alignment using slitted beamstop photodiode.

    Parameters:
    -----------
    theta_beamstop : float
        Optimal CCD Theta for beamstop (from instrument alignment)
    convergence_tolerance : float
        Tighter convergence tolerance (default: 0.005)
    max_iterations : int
        Maximum number of iterations (default: 10)

    Returns:
    --------
    z_center : float
        Final optimal Z position
    theta_offset : float
        Final theta offset

    Checkpoint:
    -----------
    Saves: Final results, all iteration data, convergence info
    """
    iteration = 0
    theta_offset = 0.0
    previous_z_center = None
    previous_theta_offset = None

    while iteration < max_iterations:
        # Ensure sample is below beam
        # insert code to move Sample Z to below_beam_position here

        # Find Z center using beamstop (finer step)
        z_positions, intensities = scan_z_range(
            z_start=0.0, z_end=5.0, z_step=0.02,  # Smaller step
            ai_channel="AI 6 BeamStop"
        )
        z_center = find_z_half_max(z_positions, intensities)

        # Find optimal Theta using beamstop (finer step)
        theta_optimal = find_theta_optimal(
            theta_start=0.0, theta_end=2.0, theta_step=0.05,  # Smaller step
            base_theta=1.0, theta_offset=theta_offset,
            theta_photodiode=theta_beamstop,  # Use beamstop theta
            ai_channel="AI 6 BeamStop"  # Use beamstop channel
        )
        theta_offset_new = theta_optimal - 1.0

        # Apply new theta offset
        # insert code to move Sample Theta to theta_offset_new here
        # insert code to wait for motor movement here

        # Check convergence
        z_converged, z_delta = check_convergence(
            z_center, previous_z_center, convergence_tolerance
        )
        theta_converged, theta_delta = check_convergence(
            theta_offset_new, previous_theta_offset, convergence_tolerance
        )

        # Checkpoint: Save iteration info
        checkpoint = {
            'step': 'fine_grained_sample_alignment_iteration',
            'iteration': iteration,
            'z_center': z_center,
            'theta_offset': theta_offset_new,
            'z_converged': z_converged,
            'z_delta': z_delta,
            'theta_converged': theta_converged,
            'theta_delta': theta_delta,
            'convergence_tolerance': convergence_tolerance
        }
        # insert code to save checkpoint here

        # Break if converged
        if z_converged and theta_converged and iteration > 0:
            break

        previous_z_center = z_center
        previous_theta_offset = theta_offset_new
        theta_offset = theta_offset_new
        iteration += 1

    # Checkpoint: Save final alignment results
    final_checkpoint = {
        'step': 'fine_grained_sample_alignment_complete',
        'final_z_center': z_center,
        'final_theta_offset': theta_offset,
        'total_iterations': iteration,
        'converged': (iteration < max_iterations),
        'convergence_tolerance': convergence_tolerance
    }
    # insert code to save checkpoint here

    return z_center, theta_offset
```
