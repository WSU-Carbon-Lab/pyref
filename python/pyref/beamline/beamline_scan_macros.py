"""Beamline scan macro generation functions.

This module provides functions for generating beamline scan macros and run files
for ALS beamline 11.0.1.2.
"""

import os  # For saving macros
import warnings  # For warning people

"""
Some Saved Motor Positions. Update here if needed.
"""

piezo_out = 17.5
piezo_in = 18.857
HOS_in = 5.5


"""
Save macro to run file
"""

def build_macro(name, path, macro):

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            print(f"Error in creating directory {path}: {e}")
            return
    file_path = os.path.join(path, f"{name}.txt")

    try:
        with open(file_path, "w") as f:
            # Write the Description Line
            f.write("DESCRIPTION\n") # First Line
            f.write("\n") # Blank line
            # I currently don't write any description to the file
            f.write("STEPS\n") # Second line that indicates steps
            for step in macro:
                f.write(str(step) + "\n") # Write the line and add a new line
    except Exception as e:
        print(f"Error creating file {name}.txt: {e}")



"""
Basic Macros
"""

def finish_line(s=600):
    # Added to every function to end the line. Not sure what this is for....
    return f") [{s!s}]"

def add_comment(comment):
    output = "Comment: "
    output += f"{comment}"
    return output

def add_prompt(prompt):
    output = "Prompt("
    output += f"{prompt}"
    output += finish_line(s=0.1)
    return output

# The save_as condition determines whether or not
def set_instrument(
        instrument,
        sync = "True",
        exp = 0,
        num_to_sum=1,
        extension = '""', # Don't include '.' for extension
        live = 'false',
        dark_image = '""',
        sub_dark = 'false' # just don't
    ):
    instrument_list = ["CCD", "Video", None]
    if instrument not in instrument_list:
        warnings.warn(f"{instrument} is not part of instrument list. Returning blank line", UserWarning, stacklevel=2)
        return ""
    if instrument is None:
        instrument = '""'

    output = "Set Instrument("
    output += instrument +","
    output += "false" +"," # Always false, corresponds to 'Setup button' that cannot be opened
    output += extension +","
    output += "false" +"," # I think this is the 'Take Picture' button
    output += live + ","
    output += dark_image + ","
    output += sync + ","
    output += str(exp) + ","
    output += sub_dark + ","
    output += str(num_to_sum)
    output += finish_line(s=0.5)

    return output



def set_DIO(name, on_off, delay=0.1):
    output = "Set DIO("
    output += name + ","
    output += str(delay) + ","
    output += str(on_off)
    output += finish_line(s=0.5)
    return output

def move_motor(motor, pos, delay=0.1):
    output = "Move Motor("
    output += motor+","
    output += str(pos)+","
    output += str(delay)+","
    output += "1" + "," # A value of 1 sets the Motor_Action to 'Move Motor'
    output += "1"
    output += finish_line(s=5)
    return output

def set_motor(motor, pos, delay=0.1):
    output = "Move Motor("
    output += motor+","
    output += str(pos)+","
    output += str(delay)+","
    output += "6" + "," # A value of 6 sets the Motor_Action to 'Set Motor'
    output += "1"
    output += finish_line(s=0.5)
    return output

def jog_motor(motor, pos, delay=0.1):
    output = "Move Motor("
    output += motor+","
    output += str(pos)+","
    output += str(delay)+","
    output += "20" + "," # A value of 6 sets the Motor_Action to 'Set Motor'
    output += "1"
    output += finish_line(s=5)
    return output

# This is currently not recommended since motors CANNOT be stopped
def move_trajectory(name_of_traj, delay=0.1):
    output = "Move Trajectory("
    output += name_of_traj + ","
    output += str(delay) + ","
    output += "0"
    output += finish_line()
    return output

def save_trajectory(name, list_of_motors):
    output = "Save Trajectory("
    output += name +","
    output += "10.0" +"," # Days till it gets deleted
    for motor in list_of_motors:
        output += motor + "\t"
    output = output[:-2] + "\r\n" # Remove the last tab and finish the list of motors
    # A few other things need to go here
    output += finish_line()
    return output

def relative_photodiode_scan(motor, delta, incr,delay=0.1,count_time=0.3):
    # Order of inputs: motor, delta, "current position", incr, delay, count, other defaults"
    output = "Single Motor Relative Scan("
    output += motor+","
    output += str(delta)+",0," # Includes a temporary (current position at 0)
    output += str(incr)+","
    output += str(delay)+","
    output += str(count_time)+","
    output += "1"+"," # Number of Scans
    output += "false"+"," # Bi-directional
    output += "Photodiode"+"," # AI to run
    output += "Center FWHM"+"," # At end of scan
    output += "Max"+"," # Max or Min value
    output += "100.0"+"," # Percent of Max/Min Value
    output += "false"+"," # A few other things that I may need to figure out...
    # A few other things need to go here
    output += ")"
    output += " [600]"
    return output

def relative_generic_scan(
        motor,
        base_file_name,
        AI = "Photodiode",
        delta = 0.5,
        incr = 0.1,
        delay = 0.1,
        count_time=0.3,
        num_scans = 1,
        bidirection = "false",
        end_of_scan = "Center FWHM",
        min_or_max = "Max",
        perc_min_or_max = 50.0,
        smooth=2.0
    ):
    # Order of inputs: motor, delta, "current position", incr, delay, count, other defaults"
    output = "Single Motor Relative Scan("
    output += motor+","
    output += str(delta)+",0,"
    output += str(incr)+","
    output += str(delay)+","
    output += str(count_time)+","
    output += str(num_scans)+","
    output += bidirection+","
    output += AI +","
    output += end_of_scan+","
    output += min_or_max+","
    output += str(perc_min_or_max)+","
    output += "false"+"," # Unsure what this is for
    output += motor +","     # Unsure what this is for The default seems to be 'Sample X'
    output += str(smooth) + ","
    output += '""' + "," + base_file_name
    # A few other things need to go here to wrap up the line
    output += finish_line()
    return output

def absolute_generic_scan(
        motor,
        start,
        stop,
        incr,
        delay=0.1,
        count_time=0.5,
        bidirection='false',
        end_of_scan='Return'
    ):
    output = "Single Motor Scan("
    output += motor+","
    output += str(start)+","
    output += str(stop)+","
    output += str(incr)+","
    output += str(delay)+","
    output += str(count_time)+","
    output += "1"+","
    output += bidirection+","
    output += "EPU Polarization" + ","
    output += end_of_scan + ","
    output += "Min"
    output += finish_line()
    return output

def analog_from_file(name, path_to_scan, delay=0.1, count_time=0.5, move_sequential="false", dont_repeat="false", end_of_scan='Return', end_trajectory='""'):
    output  = "From File Scan("
    output += windows_path(path_to_scan) +","
    output += str(delay) + ","
    output += str(count_time) + ","
    output += end_of_scan + ","
    output += move_sequential + ","
    output += dont_repeat + ","
    output += end_trajectory + "," # This is a trajectory to run when complete. Keep this as open quotes
    output += '""' + ","
    output += name
    output += finish_line()

    return output


def time_scan(name, time_between_points, total_time=1, number_of_samples=1, count_time=0.1, stop_condition='Samples Taken'):
    # Make sure the stop condition is one of the two options. 'Samples Taken' or 'Time Elapsed'
    if stop_condition=='Samples Taken':
        total_time = number_of_samples/time_between_points
    elif stop_condition=='Time Elapsed':
        number_of_samples = total_time/time_between_points
    else:
        return ""

    output = "Time Scan("
    output += str(total_time) + ","
    output += str(number_of_samples) + ","
    output += count_time + ","
    output += stop_condition +","
    output += '""' + ","
    output += name
    output += finish_line()

    return output

"""
Initialization macros that start everything
"""



"""
Experiment Specific macros
"""

clear_instruments = set_instrument(None, extension="")


def begin_macro(name):
    macro = []

    macro += [add_comment(f"{name} macro")]
    macro += [clear_instruments]

    return macro


# Built in trajectories
def run_dict_trajectory(d):

    if not isinstance(d, dict):
        msg = "Input must be a dictionary containing Motor:Position Pairs"
        raise TypeError(msg)
    trajectory_name = get_dict_name(d)

    macro = []
    macro += [add_comment(f"Run Trajectory: {trajectory_name}")]
    for key, value in d.items():
        macro += [move_motor(key, value, 0.1)]

    return macro


def piezo_toggle(in_out=0):
    """
    Toggle the Piezo position by bringing the shutter in or out
    Move the shutter in, in_out = 1
    Move the shutter out, in_out = 0.
    """
    macro = []
    if(in_out):
        macro += [move_motor("PiezoShutter Trans", piezo_in, delay=0.1)]
        macro += [set_DIO("Air Shutter Output", on_off="ON", delay=0.1)]
    if(not in_out):
        macro += [set_DIO("Air Shutter Output", on_off="OFF", delay=0.1)]
        macro += [move_motor("PiezoShutter Trans", piezo_out, delay=0.1)]
    return macro


# Auto-align sample-theta given a sample position
def XRR_lineup(repeat=3):
    # Repeated Scans
    course_z_scan = relative_generic_scan("Sample Z", "XRR_lineup_scan_ZC_", AI="Photodiode", end_of_scan="Move % of Min/Max", perc_min_or_max=50.0, delta=1.5, incr=0.1)
    fine_z_scan = relative_generic_scan("Sample Z", "XRR_lineup_scan_ZF_",  AI="Photodiode", end_of_scan="Move % of Min/Max", perc_min_or_max=50.0, delta=0.5, incr=0.02)
    theta_rocking_scan = relative_generic_scan("Sample Theta", "XRR_lineup_scan_th_", AI="Photodiode", delta=2, incr=0.1)
    shutter_on =  set_DIO("Air Shutter Output", on_off = "ON", delay=0.1)
    shutter_off =  set_DIO("Air Shutter Output", on_off = "OFF", delay=0.1)

    #theta_rocking_scan = relative_photodiode_scan("Sample Theta", 2, 0.02)
    th = 1
    # Start building the scan
    macro = []
    macro += [add_comment("Begin XRR_Lineup")]
    macro += [clear_instruments] # Make sure the CCD is turned off for diode scans
    macro += piezo_toggle(in_out=0) # Move the Piezo motor out of the way to actuate the shutter
    macro += [move_motor("Higher Order Suppressor", HOS_in, 0.1)]
    macro += [move_motor("Sample Theta", 0, 0.1)] # Make sure the sample is at the current theta = 0
    macro += [move_motor("CCD Theta", 0.0, 0.1)] # Move the CCD down to the zero position
    macro += [jog_motor("Sample Z", -2, 0.1)] # Move the sample out of the way for PD scan
    macro += [shutter_on]
    macro += [relative_generic_scan("CCD Theta", "XRRLineup_CCDth_", AI="Photodiode", end_of_scan="Center FWHM", perc_min_or_max=50.0, delta=2.0, incr=0.1)]
    macro += [shutter_off]
    macro += [jog_motor("Sample Z", 2, 0.1)]
    macro += [shutter_on]
    macro += [course_z_scan]
    macro += [fine_z_scan]
    macro += [shutter_off]
    macro += [jog_motor("Sample Theta", th, 0.1)]
    macro += [jog_motor("CCD Theta", 2*th, 0.1)]
    macro += [shutter_on]
    macro += [theta_rocking_scan]
    macro += [shutter_off]
    macro += [jog_motor("Sample Theta", -th, 0.1)]
    macro += [jog_motor("CCD Theta", -2*th, 0.1)]
    for _i in range(repeat):
        #th += 0.5
        macro += [shutter_on]
        macro += [fine_z_scan]
        macro += [shutter_off]
        macro += [jog_motor("Sample Theta", th, 0.1)]
        macro += [jog_motor("CCD Theta", 2*th, 0.1)]
        macro += [shutter_on]
        macro += [theta_rocking_scan]
        macro += [shutter_off]
        macro += [jog_motor("Sample Theta", -th, 0.1)]
        macro += [jog_motor("CCD Theta", -2*th, 0.1)]

    macro += [set_motor("Sample Theta", 0)]
    macro += [set_motor('Sample Z', 0)]
    macro += [shutter_off]
    macro += piezo_toggle(in_out=1)

    return macro

# Take a photo of the sample  // will need to record pixel value of cursor
def sample_photo(name):
    macro = []
    macro += [add_comment(f"Taking photo of sample: {name}")]
    macro += [set_instrument("Video", exp=1, extension="png")] # Turn on camera as instrument
    macro += [set_DIO("Light Output", on_off="ON")] # Turn the light on
    macro += [time_scan(name, time_between_points=1)] # Take one photo based on the current motor position
    macro += [set_DIO("Light Output", on_off="OFF")] # Turn the light off before continuing
    macro += [clear_instruments] # Remove the camera from the instruments queue before continuing
    return macro

def run_I0(name, scan_file, posx, posy):
    macro = []
    macro += [add_comment(f"Running {name}")]
    macro += [clear_instruments]
    macro.extend(run_dict_trajectory(Photodiode_Far))
    macro += [move_motor('Sample X', posx, 0.1)]
    macro += [move_motor('Sample Y', posy, 0.1)]
    macro += [analog_from_file(name, scan_file)]

    return macro

"""
Experiment Specific Trajectories that included during a full run.
"""


Photodiode_Far = {
    "CCD Y": 100.000,
    "CCD Theta": 0.000,
    "CCD X": 6.000
}

XRR_init = {
    "CCD Y": 100.000,
    "CCD Theta": 0.000,
    "CCD X": 100.500,
    "Beam Stop": 15.000
}

CCDFar_LowEn = {
    "CCD X": 100.500,
    "CCD Theta": -1.305500,
    "CCD Y": 100,
    "Beam Stop": -0.750
}

CCDMid_LowEn = {
    "CCD X": 100.500,
    "CCD Theta": -1.305500,
    "CCD Y": 50,
    "Beam Stop": -0.750
}

CCDClose_LowEn = {
    "CCD X": 100.15,
    "CCD Theta": 0.8350,
    "CCD Y": 0.00,
    "Beam Stop": -0.9570
}

CCDClose_Edge_LowEn = {
    "CCD X": 89.400,
    "CCD Theta": 0.2355,
    "CCD Y": 0.000,
    "Beam Stop": 10.490
}

# Other functions...
def get_dict_name(d, namespace=locals()):
    for name, value in namespace.items():
        if value is d and isinstance(value, dict):
            return name
    return None

def windows_path(path_string):
    r"""
    Normalizes a path string by replacing all '/' and '\' separators with '\\'.

    Args:
        path_string: The input path string.

    Returns
    -------
        The normalized path string with '\\' as the separator.
    """
    # Replace forward slashes with backslashes
    path_string = path_string.replace('/', '\\')

    # Ensure double backslashes for Windows paths
    normalized_path = path_string.replace('\\', '\\\\')

    return normalized_path
