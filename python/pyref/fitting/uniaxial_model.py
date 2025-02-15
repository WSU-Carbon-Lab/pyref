"""
*Calculates the polarized X-ray reflectivity.

 from an anisotropic stratified
series of layers.

The refnx code is distributed under the following license:

Copyright (c) 2015 A. R. J. Nelson, ANSTO

Permission to use and redistribute the source code or binary forms of this
software and its documentation, with or without modification is hereby
granted provided that the above notice of copyright, these terms of use,
and the disclaimer of warranty below appear in the source code and
documentation, and that none of the names of above institutions or
authors appear in advertising or endorsement of works derived from this
software without specific prior written permission from all parties.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THIS SOFTWARE.

"""

import numpy as np
from numpy.linalg import LinAlgError

# from numba import njit, complex128

# some definitions for resolution smearing
_FWHM = 2 * np.sqrt(2 * np.log(2.0))
_INTLIMIT = 3.5

##fundamental constants ##may not need these if converion to Gaussian units works (c=1)
hc = 12398.4193  ##ev*Angstroms
c = 299792458.0
mu0 = 4.0 * np.pi * 1e-7
ep0 = 1.0 / (c**2 * mu0)
TINY = np.finfo(float).eps


def uniaxial_reflectivity(q, layers, tensor, energy):
    """
    EMpy implementation of the uniaxial 4x4 matrix formalism.

       for calculating reflectivity from a stratified
    medium.

    Uses the implementation developed by FSRStools -
    https://github.com/ddietze/FSRStools - written by Daniel Dietze

    ----------
    q: array_like
        the q values required for the calculation.
        Q = 4 * Pi / lambda * sin(omega).
        Units = Angstrom**-1
    layers: np.ndarray
        coefficients required for the calculation, has shape (2 + N, 4),
        where N is the number of layers
        layers[0, 1] - SLD of fronting (/1e-6 Angstrom**-2)
        layers[0, 2] - iSLD of fronting (/1e-6 Angstrom**-2)
        layers[N, 0] - thickness of layer N
        layers[N, 1] - SLD of layer N (/1e-6 Angstrom**-2)
        layers[N, 2] - iSLD of layer N (/1e-6 Angstrom**-2)
        layers[N, 3] - roughness between layer N-1/N
        layers[-1, 1] - SLD of backing (/1e-6 Angstrom**-2)
        layers[-1, 2] - iSLD of backing (/1e-6 Angstrom**-2)
        layers[-1, 3] - roughness between backing and last layer

    tensor: np.ndarray
        contains the 1x3x3 dimensions
        First dimension may change in teh fiture to account for multi-energy
        currently it will just cycle
    scale: float
        Multiply all reflectivities by this value.
    bkg: float
        Linear background to be added to all reflectivities
    threads: int, optional
        <THIS OPTION IS CURRENTLY IGNORED>


    Returns
    -------
    Reflectivity: np.ndarray
        Calculated reflectivity values for each q value.
    """
    # Plane of incidence - required to define polarization vectors
    OpticAxis = np.array([0.0, 0.0, 1.0])
    phi = 0  # This is a uniaxial calculation

    ##Organize qvals into proper order
    qvals = np.asarray(q)
    flatq = qvals.ravel()
    numpnts = flatq.size  # Number of q-points

    ##Grab the number of layers
    nlayers = layers.shape[0]

    ##hc has been converted into eV*Angstroms
    wl = (
        hc / energy
    )  ##calculate the wavelength array in Aangstroms for layer calculations
    k0 = 2 * np.pi / (wl)

    # Convert optical constants into dielectric tensor
    tensor = np.conj(np.eye(3) - 2 * tensor[:, :, :])  # From tensor[:,0,:,:]

    # freq = 2*np.pi * c/wls #Angular frequency
    theta_exp = np.zeros(numpnts, dtype=float)
    theta_exp = np.pi / 2 - np.arcsin((flatq[:] / (2 * k0)).clip(-1, 1))

    ##Generate arrays of data for calculating transfer matrix
    ##Scalar values ~~
    ## Special cases!
    ##kx is constant for each wavelength but changes with angle
    ## Dimensionality ##
    ## (angle)
    # kx = np.zeros(numpnts, dtype=complex)
    # ky = np.zeros(numpnts, dtype=complex) #Used to keep the k vectors three
    # components later on for cross / dot products
    kx = k0 * np.sin(theta_exp) * np.cos(phi)
    ky = k0 * np.sin(theta_exp) * np.sin(phi)

    ## Calculate the eigenvalues corresponding to kz ~~ Each one has 4 solutions
    ## Dimensionality ##
    ## (angle, #layer, solution)
    kz = np.zeros((numpnts, nlayers, 4), dtype=complex)

    ## Calculate the eignvectors corresponding to each kz ~~ polarization of D and H
    ## Dimensionality ##
    ## (angle, #layers, solution, vector)
    Dpol = np.zeros(
        (numpnts, nlayers, 4, 3), dtype=complex
    )  ##The polarization of the displacement field
    Hpol = np.zeros(
        (numpnts, nlayers, 4, 3), dtype=complex
    )  ##The polarization of the magnetic field

    # Cycle through the layers and calculate kz
    for j, epsilon in enumerate(
        tensor
    ):  # Each layer will have a different epsilon and subsequent kz
        kz[:, j, :] = calculate_kz_uni(epsilon, kx, ky, k0, opticaxis=OpticAxis)
        Dpol[:, j, :, :], Hpol[:, j, :, :] = calculate_Dpol_uni(
            epsilon, kx, ky, kz[:, j, :], k0, opticaxis=OpticAxis
        )

    ##Make matrices for the transfer matrix calculation
    ##Dimensionality ##
    ##(angles, #layers, Matrix (4,4)

    ## Propogation Matrix
    P = calculate_P(
        numpnts, nlayers, kz[:, :, :], layers[:, 0]
    )  ##layers[k,0] is the thicknes of layer k
    ##Nevot-Croche roughness matrix
    W = calculate_W(numpnts, nlayers, kz[:, :, :], kz[:, :, :], layers[:, 3])
    ##Dynamic Matrix and inverse
    D, Di = calculate_D(numpnts, nlayers, Dpol[:, :, :, :], Hpol[:, :, :, :])

    ##Calculate the full system transfer matrix
    ##Dimensionality ##
    ##(angles, Matrix (4,4))
    M = np.ones((numpnts, 4, 4), dtype=complex)
    # Make a (numpnts x 4x4) identity matrix for the TMM -
    M = np.einsum("...ij,ij->...ij", M, np.identity(4))
    M = calculate_TMM(numpnts, nlayers, M, D, Di, P, W)
    ##Calculate the final outputs and organize into the appropriate waves for later
    refl, tran = calculate_output(numpnts, M)

    return refl, tran, kx, ky, kz, Dpol, Hpol, D, Di, P, W, M


"""
The following functions were adapted from PyATMM copyright Pavel Dmitriev
"""


def calculate_kz_uni(ep, kx, ky, k0, opticaxis=(None)):
    """Calculate the z-component of the wavevector for uniaxial media.

    This function calculates the z-components of both ordinary and extraordinary waves
    in a uniaxial medium, given the permittivity tensor and the x,y components of the
    wavevector.

    Parameters
    ----------
    ep : numpy.ndarray
        3x3 permittivity tensor of the material
    kx : numpy.ndarray
        x-component of the wavevector
    ky : numpy.ndarray
        y-component of the wavevector
    k0 : float
        Free space wavevector magnitude
    opticaxis : tuple, optional
        Unit vector defining the optical axis direction. Defaults to [0.0, 1.0, 0.0]

    Returns
    -------
    numpy.ndarray
        Array of shape (len(kx), 4) containing the z-components of the wavevector:
        - [:, 0]: forward extraordinary wave
        - [:, 1]: backward extraordinary wave
        - [:, 2]: forward ordinary wave
        - [:, 3]: backward ordinary wave

    Notes
    -----
    The calculation assumes the material is uniaxial with the extraordinary axis
    aligned with the optic axis. The ordinary and extraordinary components are
    calculated using the standard dispersion relations for uniaxial media.
    """
    # Calculate ordinary and extraordinary components from the tensor

    if opticaxis is None:
        opticaxis = [0.0, 1.0, 0.0]
    e_o = ep[0, 0]
    e_e = ep[2, 2]
    nu = (e_e - e_o) / e_o  # intermediate birefringence from reference
    k_par = np.sqrt(kx**2 + ky**2)  # Magnitude of parallel component
    # l = [kx/k_par, ky/k_par, 0]

    kz_ord = np.zeros(len(kx), dtype=np.complex128)
    kz_extraord = np.zeros(len(kx), dtype=np.complex128)
    kz_out = np.zeros((len(kx), 4), dtype=np.complex128)

    # n = [0, 0, 1] #Normal vector
    # if not numpy.isclose(k_par, 0):
    #    l = [kx/k_par, ky/k_par, 0]
    #    assert numpy.isclose(numpy.dot(l, l), 1)
    # else:
    #    l = [0, 0, 0]

    # Dot product between optical axis and vector normal and perpindicular component
    na = 1  # numpy.dot(n, opticAxis)
    la = 0  # numpy.dot(l, opticAxis)

    kz_ord = np.sqrt(e_o * k0**2 - k_par[:] ** 2)  # , dtype=np.complex128)

    kz_extraord = (1 / (1 + nu * na**2)) * (
        -nu * k_par[:] * na * la
        + np.sqrt(
            e_o * k0**2 * (1 + nu) * (1 + nu * na**2)
            - k_par[:] ** 2 * (1 + nu * (la**2 + na**2))
        )
    )

    kz_out[:, 2] = kz_ord
    kz_out[:, 3] = -kz_ord
    kz_out[:, 0] = kz_extraord
    kz_out[:, 1] = -kz_extraord
    return kz_out


def calculate_Dpol_uni(ep, kx, ky, kz, k0, opticaxis=(None)):
    """Calculate electric and magnetic dipole polarizations for uniaxial materials.

    This function computes the electric and magnetic dipole polarization vectors for
    a uniaxial anisotropic material, given the permittivity tensor and wavevector
    components.

    Args:
        ep (numpy.ndarray): 3x3 permittivity tensor for the uniaxial material
        kx (numpy.ndarray): x-component of the wavevector
        ky (numpy.ndarray): y-component of the wavevector
        kz (numpy.ndarray): z-component of the wavevector
        k0 (float): Free space wavevector magnitude
        opticaxis (list, optional): Unit vector defining optical axis direction.
        Defaults to [0.0, 1.0, 0.0]

    Returns
    -------
        tuple: A tuple containing:
            - dpol_temp (numpy.ndarray): Normalized electric dipole polarization vectors
            - hpol_temp (numpy.ndarray): Normalized magnetic dipole polarization vectors

    Notes
    -----
        - The optical axis should not be collinear with the k-vector
        - The optical axis should be a unit vector
        - The function handles both ordinary and extraordinary waves in the uniaxial
        material
    """
    if opticaxis is None:
        opticaxis = [0.0, 1.0, 0.0]
    e_o = ep[0, 0]
    e_e = ep[2, 2]
    nu = (e_e - e_o) / e_o  # intermediate birefringence from reference

    kvec = np.zeros((len(kx), 4, 3), dtype=np.complex128)
    kdiv = np.zeros((len(kx), 4), dtype=np.complex128)
    dpol_temp = np.zeros((len(kx), 4, 3), dtype=np.complex128)
    hpol_temp = np.zeros((len(kx), 4, 3), dtype=np.complex128)

    # create k-vector
    kvec[:, :, 0] = kx[:, None]
    kvec[:, :, 1] = ky[:, None]
    kvec[:, :, 2] = kz

    kdiv = np.sqrt(
        np.einsum("ijk,ijk->ij", kvec, kvec)
    )  # Performs the commented out dot product calculation

    knorm = kvec / kdiv[:, :, None]  # (np.linalg.norm(kvec,axis=-1)[:,:,None])

    # calc propogation of k along optical axis
    kpol = np.dot(knorm, opticaxis)

    dpol_temp[:, 2, :] = np.cross(opticaxis[None, :], knorm[:, 2, :])
    dpol_temp[:, 3, :] = np.cross(opticaxis[None, :], knorm[:, 3, :])
    dpol_temp[:, 0, :] = np.subtract(
        opticaxis[None, :],
        ((1 + nu) / (1 + nu * kpol[:, 0, None] ** 2))
        * kpol[:, 0, None]
        * knorm[:, 0, :],
    )
    dpol_temp[:, 1, :] = np.subtract(
        opticaxis[None, :],
        ((1 + nu) / (1 + nu * kpol[:, 1, None] ** 2))
        * kpol[:, 1, None]
        * knorm[:, 1, :],
    )

    dpol_norm = np.linalg.norm(dpol_temp, axis=-1)
    dpol_temp /= dpol_norm[:, :, None] + TINY
    hpol_temp = np.cross(kvec, dpol_temp) * (1 / k0)

    return dpol_temp, hpol_temp


# calculate the dynamic matrix and its inverse
def calculate_D(numpnts, nlayers, Dpol, Hpol):
    """
    Transfer matrices D and its inverse Di for a multilayer optical system.

    This function constructs the transfer matrix D and calculates its inverse Di for
    each point
    and layer in the system using polarization components from D-polarized and
    H-polarized fields.

    Parameters
    ----------
    numpnts : int
        Number of points (typically wavelengths or angles) to calculate for
    nlayers : int
        Number of layers in the optical system
    Dpol : ndarray
        D-polarization transfer matrix components with shape (numpnts, nlayers, 4, 4)
    Hpol : ndarray
        H-polarization transfer matrix components with shape (numpnts, nlayers, 4, 4)

    Returns
    -------
    list
        A list containing two elements:
        - D_Temp : ndarray of shape (numpnts, nlayers, 4, 4)
            The constructed transfer matrix D
        - Di_Temp : ndarray of shape (numpnts, nlayers, 4, 4)
            The inverse of D_Temp, calculated using np.linalg.inv() or np.linalg.pinv()
            as fallback

    Notes
    -----
    The function attempts to use numpy.linalg.inv() for matrix inversion first, and
    falls back
    to numpy.linalg.pinv() (pseudo-inverse) if the regular inversion fails.
    """
    D_Temp = np.zeros((numpnts, nlayers, 4, 4), dtype=np.complex128)
    Di_Temp = np.zeros((numpnts, nlayers, 4, 4), dtype=np.complex128)

    D_Temp[:, :, 0, 0] = Dpol[:, :, 0, 0]
    D_Temp[:, :, 0, 1] = Dpol[:, :, 1, 0]
    D_Temp[:, :, 0, 2] = Dpol[:, :, 2, 0]
    D_Temp[:, :, 0, 3] = Dpol[:, :, 3, 0]
    D_Temp[:, :, 1, 0] = Hpol[:, :, 0, 1]
    D_Temp[:, :, 1, 1] = Hpol[:, :, 1, 1]
    D_Temp[:, :, 1, 2] = Hpol[:, :, 2, 1]
    D_Temp[:, :, 1, 3] = Hpol[:, :, 3, 1]
    D_Temp[:, :, 2, 0] = Dpol[:, :, 0, 1]
    D_Temp[:, :, 2, 1] = Dpol[:, :, 1, 1]
    D_Temp[:, :, 2, 2] = Dpol[:, :, 2, 1]
    D_Temp[:, :, 2, 3] = Dpol[:, :, 3, 1]
    D_Temp[:, :, 3, 0] = Hpol[:, :, 0, 0]
    D_Temp[:, :, 3, 1] = Hpol[:, :, 1, 0]
    D_Temp[:, :, 3, 2] = Hpol[:, :, 2, 0]
    D_Temp[:, :, 3, 3] = Hpol[:, :, 3, 0]

    """
    for i in range(numpnts):
        for j in range(nlayers):
            Di_Temp[i,j,:,:] = np.linalg.pinv(D_Temp[i,j,:,:])
    """
    # Try running an a matrix inversion for the tranfer matrix.
    # If it fails, run a pseudo-inverse
    # Update 07/07/2021: I don't think the uniaxial calculation will error...changing
    # pinv to inv
    #                   for default calculation
    try:
        Di_Temp = np.linalg.inv(D_Temp)  # Broadcasted along the 'numpnts' dimension
    except LinAlgError:
        Di_Temp = np.linalg.pinv(D_Temp)

    return [D_Temp, Di_Temp]


def calculate_P(numpnts, nlayers, kz, d):
    """
    Calculate the propagation matrix using the previously calculated values for kz.

        :param complex 4-entry kz: Eigenvalues for solving characteristic equation, 4
        potentially degenerate inputs
        :param float d: thickness of the layer in question. (units: Angstroms)
        returns: :math:`P`

    .. important:: Requires prior execution of :py:func:`calculate_kz`.
    """
    # Create the diagonal components in the propogation matrix
    # Cast into a 4x4 version through redundent broadcasting
    diagonal_components = np.exp(-1j * kz[:, :, :, None] * d[None, :, None, None])
    # Element by element multiplication with the identity over each q-point
    P_temp = np.einsum("...jk,jk->...jk", diagonal_components, np.identity(4))

    return P_temp


def calculate_W(numpnts, nlayers, kz1, kz2, r):
    """Calculate matrix W for uniaxial model fitting.

    This function calculates the W matrix used in uniaxial model fitting, which involves
    exponential terms of wavevectors for different layers.

    Parameters
    ----------
    numpnts : int
        Number of points in the calculation grid
    nlayers : int
        Number of layers in the model
    kz1 : ndarray
        Array of shape (numpnts, nlayers, 4) containing z-component of wavevector 1
    kz2 : ndarray
        Array of shape (numpnts, nlayers, 4) containing z-component of wavevector 2
    r : ndarray
        Array of shape (nlayers,) containing roughness parameters for each layer

    Returns
    -------
    ndarray
        4D array of shape (numpnts, nlayers, 4, 4) containing the W matrix elements
        with exponential terms for interface roughness

    Notes
    -----
    The function calculates exponential terms for interface roughness using the
    sum and difference of wavevectors, and arranges them in a 4x4 matrix format
    for each point and layer.
    """
    W_temp = np.zeros((numpnts, nlayers, 4, 4), dtype=np.complex128)
    eplus = np.zeros((numpnts, nlayers, 4), dtype=np.complex128)
    eminus = np.zeros((numpnts, nlayers, 4), dtype=np.complex128)

    kz2 = np.roll(
        kz2, 1, axis=1
    )  # Reindex to allow broadcasting in the next step....see commented loop
    # for j in range(nlayers):
    eplus[:, :, :] = np.exp(
        -((kz1[:, :, :] + kz2[:, :, :]) ** 2) * r[None, :, None] ** 2 / 2
    )
    eminus[:, :, :] = np.exp(
        -((kz1[:, :, :] - kz2[:, :, :]) ** 2) * r[None, :, None] ** 2 / 2
    )

    W_temp[:, :, 0, 0] = eminus[:, :, 0]
    W_temp[:, :, 0, 1] = eplus[:, :, 1]
    W_temp[:, :, 0, 2] = eminus[:, :, 2]
    W_temp[:, :, 0, 3] = eplus[:, :, 3]
    W_temp[:, :, 1, 0] = eplus[:, :, 0]
    W_temp[:, :, 1, 1] = eminus[:, :, 1]
    W_temp[:, :, 1, 2] = eplus[:, :, 2]
    W_temp[:, :, 1, 3] = eminus[:, :, 3]
    W_temp[:, :, 2, 0] = eminus[:, :, 0]
    W_temp[:, :, 2, 1] = eplus[:, :, 1]
    W_temp[:, :, 2, 2] = eminus[:, :, 2]
    W_temp[:, :, 2, 3] = eplus[:, :, 3]
    W_temp[:, :, 3, 0] = eplus[:, :, 0]
    W_temp[:, :, 3, 1] = eminus[:, :, 1]
    W_temp[:, :, 3, 2] = eplus[:, :, 2]
    W_temp[:, :, 3, 3] = eminus[:, :, 3]

    return W_temp


def calculate_TMM(numpnts, nlayers, M, D, Di, P, W):
    """Calculate Transfer Matrix Method (TMM) for multilayered optical system.

    This function computes the total transfer matrix for a multilayer optical system
    using
    the Transfer Matrix Method (TMM). It iteratively multiplies matrices representing
    different
    layers to obtain the overall system response.

    Parameters
    ----------
    numpnts : int
        Number of points in the calculation (typically wavelength or frequency points).
    nlayers : int
        Number of layers in the optical system.
    M : ndarray
        Initial transfer matrix, shape (numpnts, 2, 2).
    D : ndarray
        Dynamic matrices for each layer, shape (numpnts, nlayers, 2, 2).
    Di : ndarray
        Inverse dynamic matrices for each layer, shape (numpnts, nlayers-1, 2, 2).
    P : ndarray
        Propagation matrices for each layer, shape (numpnts, nlayers-1, 2, 2).
    W : ndarray
        Interface matrices between layers, shape (numpnts, nlayers, 2, 2).

    Returns
    -------
    ndarray
        Final transfer matrix M after considering all layers, shape (numpnts, 2, 2).

    Notes
    -----
    The function implements the TMM calculation following the sequence:
    1. Iterates through internal layers
    2. Multiplies appropriate matrices using Einstein summation
    3. Includes final layer calculation separately
    """
    for j in range(1, nlayers - 1):
        A = np.einsum("...ij,...jk ->...ik", Di[:, j - 1, :, :], D[:, j, :, :])
        B = A * W[:, j, :, :]
        C = np.einsum("...ij,...jk ->...ik", B, P[:, j, :, :])
        M[:, :, :] = np.einsum("...ij,...jk ->...ik", M[:, :, :], C)
    AA = np.einsum("...ij,...jk ->...ik", Di[:, -2, :, :], D[:, -1, :, :])
    BB = AA * W[:, -1, :, :]
    M[:, :, :] = np.einsum("...ij,...jk ->...ik", M[:, :, :], BB)
    return M


def calculate_output(numpnts, M_full):
    """Calculate reflection and transmission coefficients from a transfer matrix.

    This function computes the reflection and transmission coefficients for s- and
    p-polarized light
    from a given transfer matrix using the Berreman 4x4 matrix method.

    Parameters
    ----------
    numpnts : int
        Number of points/wavelengths being calculated
    M_full : ndarray
        4x4 transfer matrix with shape (numpnts, 4, 4) containing the optical system
        information

    Returns
    -------
    refl : ndarray
        Array of shape (numpnts, 2, 2) containing reflection coefficients:
        refl[:,0,0] = r_ss (s to s reflection)
        refl[:,0,1] = r_sp (p to s reflection)
        refl[:,1,0] = r_ps (s to p reflection)
        refl[:,1,1] = r_pp (p to p reflection)
    tran : ndarray
        Complex array of shape (numpnts, 2, 2) containing transmission coefficients:
        tran[:,0,0] = t_ss (s to s transmission)
        tran[:,0,1] = t_sp (p to s transmission)
        tran[:,1,0] = t_ps (s to p transmission)
        tran[:,1,1] = t_pp (p to p transmission)

    Notes
    -----
    The coefficients are calculated using the standard transfer matrix method formalism
    where the coefficients are obtained from the elements of the 4x4 transfer matrix.
    """
    refl = np.zeros((numpnts, 2, 2), dtype=np.float64)
    tran = np.zeros((numpnts, 2, 2), dtype=np.complex128)

    M = M_full

    # Calculate denominator with numerical stability check
    denom = M[:, 0, 0] * M[:, 2, 2] - M[:, 0, 2] * M[:, 2, 0]
    # Add tiny constant to avoid division by zero
    denom = np.where(denom < TINY, denom + TINY, denom)

    # Calculate reflection coefficients
    r_ss = (M[:, 1, 0] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 0]) / denom
    r_sp = (M[:, 3, 0] * M[:, 2, 2] - M[:, 3, 2] * M[:, 2, 0]) / denom
    r_ps = (M[:, 0, 0] * M[:, 1, 2] - M[:, 1, 0] * M[:, 0, 2]) / denom
    r_pp = (M[:, 0, 0] * M[:, 3, 2] - M[:, 3, 0] * M[:, 0, 2]) / denom

    # Calculate transmission coefficients
    t_ss = M[:, 2, 2] / denom
    t_sp = -M[:, 2, 0] / denom
    t_ps = -M[:, 0, 2] / denom
    t_pp = M[:, 0, 0] / denom

    # Clip reflection coefficients to physical values
    refl[:, 0, 0] = np.clip(np.real(np.multiply(r_ss, np.conj(r_ss))), 0, 1)
    refl[:, 0, 1] = np.clip(np.real(np.multiply(r_sp, np.conj(r_sp))), 0, 1)
    refl[:, 1, 0] = np.clip(np.real(np.multiply(r_ps, np.conj(r_ps))), 0, 1)
    refl[:, 1, 1] = np.clip(np.real(np.multiply(r_pp, np.conj(r_pp))), 0, 1)

    # Store transmission coefficients
    tran[:, 0, 0] = t_ss
    tran[:, 0, 1] = t_sp
    tran[:, 1, 0] = t_ps
    tran[:, 1, 1] = t_pp

    return refl, tran
