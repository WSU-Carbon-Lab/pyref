"""
Xray Reflectivity data analysis package.

This package is designed to provide a simple interface for the analysis of X-ray
reflectivity data collected at the Advanced Light Source (ALS) at Lawrence Berkeley
National Laboratory (LBNL) Beamline 11.0.1.2. The package is based on the early
PRSoXR package written by Thomas Ferron during his time at NIST. This work makes a good
faith effort to provide the same functionality as the original PRSoXR package, but with
a more modern and user-friendly interface, and rust bindings for I/O operations.

*Author*: Harlan Heilman

### NIST LICENSE
The package adhears to the nist license, and is provided as is with no warranty.

NIST-developed software is provided by NIST as a public service. You may use, copy, and
distribute copies of the software in any medium, provided that you keep intact this
entire notice. You may improve, modify, and create derivative works of the software or
any portion of the software, and you may copy and distribute such modifications or
works. Modified works should carry a notice stating that you changed the software and
should note the date and nature of any such change. Please explicitly acknowledge the
National Institute of Standards and Technology as the source of the software.

NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY
KIND, EXPRESS, IMPLIED, IN FACT, OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT
LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
NON-INFRINGEMENT, AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE
OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL
BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF
THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS,
ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

You are solely responsible for determining the appropriateness of using and distributing
the software and you assume all risks associated with its use, including but not limited
to the risks and costs of program errors, compliance with applicable laws, damage to or
loss of data, programs or equipment, and the unavailability or interruption of
operation. This software is not intended to be used in any situation where a failure
could cause risk of injury or damage to property. The software developed by NIST
employees is not subject to copyright protection within the United States.

This data/work was created by employees of the National Institute of Standards and
Technology (NIST), an agency of the Federal Government. Pursuant to title 17 United
States Code Section 105, works of NIST employees are not subject to copyright protection
in the United States.  This data/work may be subject to foreign copyright.

The data/work is provided by NIST as a public service and is expressly provided
“AS IS.” NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR STATUTORY, INCLUDING,
WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST does not warrant or make any
representations regarding the use of the data or the results thereof, including but not
limited to the correctness, accuracy, reliability or usefulness of the data. NIST SHALL
NOT BE LIABLE AND YOU HEREBY RELEASE NIST FROM LIABILITY FOR ANY INDIRECT,
CONSEQUENTIAL, SPECIAL, OR INCIDENTAL DAMAGES (INCLUDING DAMAGES FOR LOSS OF BUSINESS
PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, AND THE LIKE), WHETHER
ARISING IN TORT, CONTRACT, OR OTHERWISE, ARISING FROM OR RELATING TO THE DATA (OR THE
USE OF OR INABILITY TO USE THIS DATA), EVEN IF NIST HAS BEEN ADVISED OF THE POSSIBILITY
OF SUCH DAMAGES.

To the extent that NIST may hold copyright in countries other than the United States,
you are hereby granted the non-exclusive irrevocable and unconditional right to print,
publish, prepare derivative works and distribute the NIST data, in any medium, or
authorize others to do so on your behalf, on a royalty-free basis throughout the world.

You may improve, modify, and create derivative works of the data or any portion of the
data, and you may copy and distribute such modifications or works. Modified works should
carry a notice stating that you changed the data and should note the date and nature of
any such change. Please explicitly acknowledge the National Institute of Standards and
Technology as the source of the data:  Data citation recommendations are provided at
https://www.nist.gov/open/license.

Permission to use this data is contingent upon your acceptance of the terms of this
agreement and upon your providing appropriate acknowledgments of NIST's creation of the
data/work.
"""

__author__ = """Harlan Heilman"""
__email__ = "Harlan.Heilman@wsu.edu"

from pyref.loader import PrsoxrLoader
from pyref.masking import InteractiveImageMasker
from pyref.pyref import py_read_experiment
from pyref.utils import err_prop_div, err_prop_mult, weighted_mean, weighted_std

__all__ = [
    "InteractiveImageMasker",
    "PrsoxrLoader",
    "err_prop_div",
    "err_prop_mult",
    "py_read_experiment",
    "weighted_mean",
    "weighted_std",
]
