r"""
Matrix method algorithm to calculate X-ray reflectivity and transmittivity.

The algorithms in this file are written in numba for maximum speed, and are
therefore sometimes more difficult to follow than the pure python implementations.

Matrix method algorithm for x-ray reflectivity and transmittivity as described by A.
Gibaud and G. Vignaud in J. Daillant, A. Gibaud (Eds.), "X-ray and
Neutron Reflectivity: Principles and Applications", Lect. Notes Phys. 770
(Springler, Berlin Heidelberg 2009), DOI 10.1007/978-3-540-88588-7, chapter 3.2.1
"The Matrix Method".

Conventions used:
I'm following the conventions of A. Gibaud and G. Vignaud cited above:
There is a stack of j=0..N media on a substrate S, with j=0 and S being infinite.
The interface between j and j+1 is Z_{j+1}, so Z_1 is the interface between the topmost
layer (i.e. usually air or vacuum) and the first sample layer. Electromagnetic waves are
 represented by their electric field \vec{E}, which is divided in one part travelling
downwards, \vec{E}^- and one travelling upwards, \vec{E}^+.
"""

import cmath
import enum
import math

import numba
import numpy as np
from numba import prange

jit = numba.jit(nopython=True, cache=False, fastmath=True, nogil=True)
pjit = numba.jit(nopython=True, cache=False, fastmath=True, nogil=True, parallel=True)


# type alias
Array = np.ndarray


class Polarization(enum.IntEnum):
    """Polarization of the light."""

    P = 0
    S = 1


@jit
def _p_m_s_pol(k_z: Array, n2: Array, j: int, s2h: Array) -> tuple[complex, complex]:
    """
    Matrix elements of the refraction matrices in s polarization.

    p[j] is p_{j, j+1}
    p_{j, j+1} = (k_{z, j} + k_{z, j+1}) / (2 * k_{z, j}) * exp(-(k_{z,j} - k_{z,j+1})**2 sigma_j**2/2) for all j=0..N-1
    m_{j, j+1} = (k_{z, j} - k_{z, j+1}) / (2 * k_{z, j}) * exp(-(k_{z,j} + k_{z,j+1})**2 sigma_j**2/2) for all j=0..N-1
    """  # noqa: E501, W505
    p = k_z[j] + k_z[j + 1]
    m = k_z[j] - k_z[j + 1]
    rp = cmath.exp(-np.square(m) * s2h[j])
    rm = cmath.exp(-np.square(p) * s2h[j])
    o = 2 * k_z[j]
    return p * rp / o, m * rm / o


@jit
def _p_m_p_pol(k_z: Array, n2: Array, j: int, s2h: Array) -> tuple[complex, complex]:
    """
    Matrix elements of the refraction matrices in p polarization.

    p[j] is p_{j, j+1}
    p_{j, j+1} = (n_{j+1}**2 k_{z, j} + n_j**2 k_{z, j+1}) / (2 n_{n+1}**2 * k_{z, j}) * exp(-(k_{z,j} - k_{z,j+1})**2 sigma_j**2/2) for all j=0..N-1
    m_{j, j+1} = (n_{j+1}**2 k_{z, j} - n_j**2 k_{z, j+1}) / (2 n_{n+1}**2 * k_{z, j}) * exp(-(k_{z,j} + k_{z,j+1})**2 sigma_j**2/2) for all j=0..N-1
    """  # noqa: E501, W505
    n2lkzp = n2[j] * k_z[j + 1]
    n2lpkz = n2[j + 1] * k_z[j]
    p = n2lpkz + n2lkzp
    m = n2lpkz - n2lkzp
    rp = cmath.exp(-np.square(k_z[j] - k_z[j + 1]) * s2h[j])
    rm = cmath.exp(-np.square(k_z[j] + k_z[j + 1]) * s2h[j])
    o = 2 * n2lpkz
    return p * rp / o, m * rm / o


@jit
def _p_m(
    k_z: Array, n2: Array, j: int, s2h: Array, pol: Polarization
) -> tuple[complex, complex]:
    if pol:  # pol=1 is s
        return _p_m_s_pol(k_z=k_z, n2=n2, j=j, s2h=s2h)
    else:
        return _p_m_p_pol(k_z=k_z, n2=n2, j=j, s2h=s2h)


@jit
def _reflec_and_trans_inner(
    k2: float,
    n2: Array,
    k2n2: Array,
    theta: float,
    thick: Array,
    s2h: Array,
    N: int,
    p_m,
) -> tuple[complex, complex]:
    # wavevectors in the different layers
    k2_x = k2 * np.square(math.cos(theta))  # k_x is conserved due to snell's law
    k_z = -np.sqrt(k2n2 - k2_x)  # k_z is different for each layer.

    pS, mS = p_m(k_z=k_z, n2=n2, j=N, s2h=s2h)
    # RR over interface to substrate
    mm12 = mS
    mm22 = pS
    for l in range(N):  # noqa: E741
        j = N - l - 1  # j = N-1 .. 0
        # transition through layer j
        vj = cmath.exp(1j * k_z[j + 1] * thick[j])
        # transition through interface between j-1 an j
        pj, mj = p_m(k_z=k_z, n2=n2, j=j, s2h=s2h)
        m11 = pj / vj
        m12 = mj * vj
        m21 = mj / vj
        m22 = pj * vj

        mm12, mm22 = m11 * mm12 + m12 * mm22, m21 * mm12 + m22 * mm22

    # reflection coefficient
    r = mm12 / mm22

    # transmission coefficient
    t = 1 / mm22

    return r, t


@jit
def _precompute(
    n: Array, lam: float, thetas: Array, thick: Array, rough: Array, pol: Polarization
) -> tuple[float, Array, Array, Array, int, int]:
    N = len(thick)
    T = len(thetas)

    if not len(n) == len(rough) + 1 == N + 2:
        e = "array lengths do not match: len(n) == len(rough) + 1 == len(thick) + 2 does not hold."  # noqa: E501
        raise ValueError(e)

    k2 = np.square(2 * math.pi / lam)  # k is conserved
    n2 = np.square(n)
    k2n2 = k2 * n2
    s2h = np.square(rough) / 2

    if pol != 1 and pol != 0:
        e = "pol has to be either 1 for s-polarization or 0 for p-polarization."
        raise ValueError(e)
    return k2, n2, k2n2, s2h, N, T


@jit
def _reflec_and_trans(
    thetas: Array,
    thick: Array,
    n2: Array,
    k2: float,
    k2n2: Array,
    s2h: Array,
    N: int,
    T: int,
    p_m,
) -> tuple[Array, Array]:
    rs = np.empty(T, np.complex128)
    ts = np.empty(T, np.complex128)
    for i in range(T):
        r, t = _reflec_and_trans_inner(
            k2=k2, n2=n2, k2n2=k2n2, theta=thetas[i], thick=thick, s2h=s2h, N=N, p_m=p_m
        )
        rs[i] = r
        ts[i] = t

    return rs, ts


def reflec_and_trans(
    n: Array, lam: float, thetas: Array, thick: Array, rough: Array, pol=Polarization.S
) -> tuple[Array, Array]:
    r"""
    Reflection coefficient and the transmission coefficient.

    for a stack of N layers, with the incident
    wave coming from layer 0, which is reflected into layer 0 and transmitted into layer
    N.
    Note that N=len(n) is the total number of layers, including the substrate.
    That is the only point where the notation
    differs from Gibaud & Vignaud.
    :param n: array of refractive indices n = 1 - \\delta + i \beta of all layers,
    so n[0] is usually 1.
    :param lam: x-ray wavelength in nm
    :param thetas: array of incidence angles in rad
    :param thick: thicknesses in nm, len(thick) = N-2, since layer 0 and layer N are
    assumed infinite
    :param rough: rms roughness in nm, len(rough) = N-1 (number of interfaces)
    :param pol: polarization (either 1 for s-polarization or 0 for p-polarization)
    :return: (reflec, trans)
    """
    k2, n2, k2n2, s2h, N, T = _precompute(
        n=n, lam=lam, thetas=thetas, thick=thick, rough=rough, pol=pol
    )

    if pol:  # pol=1 is s
        return _reflec_and_trans(
            thetas=thetas,
            thick=thick,
            n2=n2,
            k2=k2,
            k2n2=k2n2,
            s2h=s2h,
            N=N,
            T=T,
            p_m=_p_m_s_pol,
        )
    else:
        return _reflec_and_trans(
            thetas=thetas,
            thick=thick,
            n2=n2,
            k2=k2,
            k2n2=k2n2,
            s2h=s2h,
            N=N,
            T=T,
            p_m=_p_m_p_pol,
        )


@pjit
def _reflec_and_trans_parallel(
    thetas: Array,
    thick: Array,
    n2: Array,
    k2: float,
    k2n2: Array,
    s2h: Array,
    N: int,
    T: int,
    p_m,
) -> tuple[Array, Array]:
    rs = np.empty(T, np.complex128)
    ts = np.empty(T, np.complex128)
    for i in prange(T):
        r, t = _reflec_and_trans_inner(
            k2=k2, n2=n2, k2n2=k2n2, theta=thetas[i], thick=thick, s2h=s2h, N=N, p_m=p_m
        )
        rs[i] = r
        ts[i] = t

    return rs, ts


def reflec_and_trans_parallel(
    n: Array, lam: float, thetas: Array, thick: Array, rough: Array, pol=Polarization.S
) -> tuple[Array, Array]:
    r"""
    Calculate the reflection coefficient and the transmission coefficient for a stack.

     of N layers, with the incident
    wave coming from layer 0, which is reflected into layer 0 and transmitted into
    layer N.

    This function calculates the thetas in parallel using numba, which can be faster,
    especially if you have large
    stacks; however, if you have to call this function a lot of times (e.g. for a fit),
      it is usually faster to
    parallelize function calls of reflec_and_trans instead of using
    reflec_and_trans_parallel.

    Note that N=len(n) is the total number of layers, including the substrate. That is
    the only point where the notation
    differs from Gibaud & Vignaud.
    :param n: array of refractive indices n = 1 - \\delta + i \beta of all layers, so
    n[0] is usually 1.
    :param lam: x-ray wavelength in nm
    :param thetas: array of incidence angles in rad
    :param thick: thicknesses in nm, len(thick) = N-2, since layer 0 and layer N are
      assumed infinite
    :param rough: rms roughness in nm, len(rough) = N-1 (number of interfaces)
    :param pol: polarization (either 1 for s-polarization or 0 for p-polarization)
    :return: (reflec, trans)
    """
    k2, n2, k2n2, s2h, N, T = _precompute(n, lam, thetas, thick, rough, pol)

    if pol:  # pol=1 is s
        return _reflec_and_trans_parallel(
            thetas=thetas,
            thick=thick,
            n2=n2,
            k2=k2,
            k2n2=k2n2,
            s2h=s2h,
            N=N,
            T=T,
            p_m=_p_m_s_pol,
        )
    else:
        return _reflec_and_trans_parallel(
            thetas=thetas,
            thick=thick,
            n2=n2,
            k2=k2,
            k2n2=k2n2,
            s2h=s2h,
            N=N,
            T=T,
            p_m=_p_m_p_pol,
        )


@jit
def _fields_inner(
    n2: Array,
    k2: float,
    k2n2: Array,
    theta: float,
    thick: Array,
    s2h: Array,
    mm12: Array,
    mm22: Array,
    rs: Array,
    ts: Array,
    kt: int,
    N: int,
    pol: Polarization,
) -> None:
    # wavevectors in the different layers
    k2_x = k2 * np.square(math.cos(theta))  # k_x is conserved due to snell's law
    k_z = -np.sqrt(k2n2 - k2_x)  # k_z is different for each layer.

    pS, mS = _p_m(k_z, n2, N, s2h, pol)

    # RR over interface to substrate
    mm12[N] = mS
    mm22[N] = pS
    for l in range(N):  # noqa: E741
        j = N - l - 1  # j = N-1 .. 0
        # transition through layer j
        vj = cmath.exp(1j * k_z[j + 1] * thick[j])
        # transition through interface between j-1 an j
        pj, mj = _p_m(k_z, n2, j, s2h, pol)
        m11 = pj / vj
        m12 = mj * vj
        m21 = mj / vj
        m22 = pj * vj

        mm12[j] = m11 * mm12[j + 1] + m12 * mm22[j + 1]
        mm22[j] = m21 * mm12[j + 1] + m22 * mm22[j + 1]

    # reflection coefficient
    r = mm12[0] / mm22[0]

    # transmission coefficient
    t = 1 / mm22[0]

    ts[kt][0] = 1  # in the vacuum layer
    rs[kt][0] = r  # in the vacuum layer
    for j in range(1, N + 1):  # j = 1 .. N
        ts[kt][j] = mm22[j] * t
        rs[kt][j] = mm12[j] * t
    ts[kt][N + 1] = t  # in the substrate
    rs[kt][N + 1] = 0


@jit
def fields(
    n: Array,
    lam: float,
    thetas: Array,
    thick: Array,
    rough: Array,
    pol: Polarization = 1,
) -> tuple[Array, Array]:
    r"""Calculate the reflection coefficient and the transmission coefficient for a stack of N layers, with the incident
    wave coming from layer 0, which is reflected into layer 0 and transmitted into layer N.
    Note that N=len(n) is the total number of layers, including the substrate. That is the only point where the notation
    differs from Gibaud & Vignaud.
    :param n: array of refractive indices n = 1 - \\delta + i \beta of all layers, so n[0] is usually 1.
    :param lam: x-ray wavelength in nm
    :param thetas: array of incidence angles in rad
    :param thick: thicknesses in nm, len(thick) = N-2, since layer 0 and layer N are assumed infinite
    :param rough: rms roughness in nm, len(rough) = N-1 (number of interfaces)
    :param pol: polarization (either 1 for s-polarization or 0 for p-polarization)
    :return: (reflec, trans).
    """  # noqa: D205, E501, W505
    k2, n2, k2n2, s2h, N, T = _precompute(
        n=n, lam=lam, thetas=thetas, thick=thick, rough=rough, pol=pol
    )

    # preallocate temporary arrays
    mm12 = np.empty(N + 1, dtype=np.complex128)
    mm22 = np.empty(N + 1, dtype=np.complex128)
    # preallocate whole result arrays
    rs = np.empty((T, N + 2), dtype=np.complex128)
    ts = np.empty((T, N + 2), dtype=np.complex128)

    for kt, theta in enumerate(thetas):
        _fields_inner(
            n2=n2,
            k2=k2,
            k2n2=k2n2,
            theta=theta,
            thick=thick,
            s2h=s2h,
            mm12=mm12,
            mm22=mm22,
            rs=rs,
            ts=ts,
            kt=kt,
            N=N,
            pol=pol,
        )
    return rs, ts


@pjit
def fields_parallel(
    n: Array,
    lam: float,
    thetas: Array,
    thick: Array,
    rough: Array,
    pol: Polarization = 1,
) -> tuple[Array, Array]:
    r"""Calculate the reflection coefficient and the transmission coefficient for a stack of N layers, with the incident
    wave coming from layer 0, which is reflected into layer 0 and transmitted into layer N.

    This function calculates the thetas in parallel using numba, which can be faster, especially if you have large
    stacks. It uses more memory and has to allocate and deallocate more memory than the non-parallel version, though.

    Note that N=len(n) is the total number of layers, including the substrate. That is the only point where the notation
    differs from Gibaud & Vignaud.
    :param n: array of refractive indices n = 1 - \\delta + i \beta of all layers, so n[0] is usually 1.
    :param lam: x-ray wavelength in nm
    :param thetas: array of incidence angles in rad
    :param thick: thicknesses in nm, len(thick) = N-2, since layer 0 and layer N are assumed infinite
    :param rough: rms roughness in nm, len(rough) = N-1 (number of interfaces)
    :param pol: polarization (either 1 for s-polarization or 0 for p-polarization)
    :return: (reflec, trans)
    """  # noqa: D205, E501, W505
    k2, n2, k2n2, s2h, N, T = _precompute(
        n=n, lam=lam, thetas=thetas, thick=thick, rough=rough, pol=pol
    )

    # preallocate whole result arrays
    rs = np.empty((T, N + 2), dtype=np.complex128)
    ts = np.empty((T, N + 2), dtype=np.complex128)

    for kt in prange(T):
        # preallocate temporary arrays
        mm12 = np.empty(N + 1, dtype=np.complex128)
        mm22 = np.empty(N + 1, dtype=np.complex128)
        _fields_inner(
            n2=n2,
            k2=k2,
            k2n2=k2n2,
            theta=thetas[kt],
            thick=thick,
            s2h=s2h,
            mm12=mm12,
            mm22=mm22,
            rs=rs,
            ts=ts,
            kt=kt,
            N=N,
            pol=pol,
        )
    return rs, ts


@jit
def _fields_positions_inner(
    thick: Array,
    n2: Array,
    s2h: Array,
    kt: int,
    rs: Array,
    ts: Array,
    mm12: Array,
    mm22: Array,
    N: int,
    k_z: Array,
    pol: Polarization,
) -> None:
    pS, mS = _p_m(k_z=k_z[kt], n2=n2, j=N, s2h=s2h, pol=pol)
    # entries of the transition matrix MM
    # mm11, mm12, mm21, mm22
    # RR over interface to substrate
    mm12[N] = mS
    mm22[N] = pS
    for l in range(N):  # noqa: E741
        j = N - l - 1  # j = N-1 .. 0
        # transition through layer j
        vj = cmath.exp(1j * k_z[kt][j + 1] * thick[j])
        # transition through interface between j-1 an j
        pj, mj = _p_m(k_z[kt], n2, j, s2h, pol)
        m11 = pj / vj
        m12 = mj * vj
        m21 = mj / vj
        m22 = pj * vj

        mm12[j] = m11 * mm12[j + 1] + m12 * mm22[j + 1]
        mm22[j] = m21 * mm12[j + 1] + m22 * mm22[j + 1]

    # reflection coefficient
    r = mm12[0] / mm22[0]

    # transmission coefficient
    t = 1 / mm22[0]

    ts[kt][0] = 1  # in the vacuum layer
    rs[kt][0] = r  # in the vacuum layer
    for l in range(N):  # noqa: E741
        j = l + 1  # j = 1 .. N
        ts[kt][j] = mm22[j] * t
        rs[kt][j] = mm12[j] * t
    ts[kt][N + 1] = t  # in the substrate
    rs[kt][N + 1] = 0


@jit
def _fields_positions_inner_positions(
    kp: int,
    pos: int,
    Z: Array,
    pos_rs: Array,
    pos_ts: Array,
    k_z: Array,
    ts: Array,
    rs: Array,
    T: int,
):
    # MM_j * (0, t) is the field at the interface between the layer j and the layer j+1.
    # thus, if pos is within layer j, we need to use the translation matrix
    # TT = exp(-ik_{z,j} h), 0 \\ 0, exp(ik_{z,j} h)
    # with h the distance between the interface between the layer j and the layer j+1
    # (the "bottom" interface if
    # the vacuum is at the top and the z-axis is pointing upwards) and pos.

    # first find out within which layer pos lies
    for j, zj in enumerate(Z):  # checking from the top  # noqa: B007
        if pos > zj:
            break
    else:  # within the substrate
        # need to special-case the substrate since we have to propagate "down" from the
        # substrate interface
        # all other cases are propagated "up" from their respective interfaces
        dist_j = 1j * (
            pos - Z[-1]
        )  # common for all thetas: distance from interface to evaluation_position
        for ko in range(T):  # iterate through all thetas
            pos_rs[ko][kp] = 0.0
            pos_ts[ko][kp] = ts[ko][-1] * cmath.exp(k_z[ko][-1] * dist_j)
        return

    # now zj = Z[j] is the layer in which pos lies
    dist_j = 1j * (
        pos - zj
    )  # common for all thetas: distance from interface to evaluation_position
    for ko in range(T):
        # now propagate the fields through the layer
        vj = cmath.exp(k_z[ko][j] * dist_j)

        # fields at position

        pos_ts[ko][kp] = ts[ko][j] * vj
        pos_rs[ko][kp] = rs[ko][j] / vj


@jit
def fields_at_positions(
    n: Array,
    lam: float,
    thetas: Array,
    thick: Array,
    rough: Array,
    evaluation_positions: Array,
    pol: Polarization = 1,
) -> tuple[Array, Array, Array, Array]:
    r"""Calculate the electric field intensities in a stack of N layers, with the incident
    wave coming from layer 0, which is reflected into layer 0 and transmitted into layer N.

    Note that N=len(n) is the total number of layers, including the substrate. That is the only point where the notation
    differs from Gibaud & Vignaud.
    :param n: array of refractive indices n = 1 - \\delta + i \beta of all layers, so n[0] is usually 1.
    :param lam: x-ray wavelength in nm
    :param thetas: array of incidence angles in rad
    :param thick: thicknesses in nm, len(thick) = N-2, since layer 0 and layer N are assumed infinite
    :param rough: rms roughness in nm, len(rough) = N-1 (number of interfaces)
    :param evaluation_positions: positions (in nm) at which the electric field should be evaluated. Given in distance
           from the surface, with the axis pointing away from the layer (i.e. negative positions are within the stack)
    :param pol: polarization (either 1 for s-polarization or 0 for p-polarization)
    :return: (reflec, trans, pos_reflec, pos_trans)
    """  # noqa: D205, E501, W505
    P = len(evaluation_positions)
    k2, n2, k2n2, s2h, N, T = _precompute(
        n=n, lam=lam, thetas=thetas, thick=thick, rough=rough, pol=pol
    )
    k2_x = k2 * np.square(
        np.cos(thetas)
    )  # k_x is conserved due to snell's law, i.e. only dependent on theta

    k_z = np.empty((T, N + 2), dtype=np.complex128)

    for kt in range(T):
        for kl in range(N + 2):
            k_z[kt][kl] = -np.sqrt(
                k2n2[kl] - k2_x[kt]
            )  # k_z is different for each layer.

    # calculate absolute interface positions from thicknesses
    Z = np.empty(N + 1, dtype=np.float64)
    Z[0] = 0.0
    cs = -np.cumsum(thick)
    for i in range(N):
        Z[i + 1] = cs[i]

    # preallocate temporary arrays
    mm12 = np.empty(N + 1, dtype=np.complex128)
    mm22 = np.empty(N + 1, dtype=np.complex128)
    # preallocate whole result arrays
    rs = np.empty((T, N + 2), dtype=np.complex128)
    ts = np.empty((T, N + 2), dtype=np.complex128)
    pos_rs = np.empty((T, P), dtype=np.complex128)
    pos_ts = np.empty((T, P), dtype=np.complex128)

    # first calculate the fields at the interfaces
    for kt in range(T):
        _fields_positions_inner(thick, n2, s2h, kt, rs, ts, mm12, mm22, N, k_z, pol)

    # now calculate the fields at the given evaluation positions
    for kp, pos in enumerate(evaluation_positions):
        _fields_positions_inner_positions(kp, pos, Z, pos_rs, pos_ts, k_z, ts, rs, T)

    return rs, ts, pos_rs, pos_ts


@pjit
def fields_at_positions_parallel(
    n: Array,
    lam: float,
    thetas: Array,
    thick: Array,
    rough: Array,
    evaluation_positions: Array,
    pol: Polarization = 1,
) -> tuple[Array, Array, Array, Array]:
    r"""
    Calculate the electric field intensities in a stack of N layers.

    This function calculates the thetas and evaluation positions in parallel using numba
    , which can be faster,
    especially if you have large stacks or many evaluation positions. It uses more
    memory and has to allocate and deallocate more memory than the non-parallel version,
      though.

    Note that N=len(n) is the total number of layers, including the substrate. That is
    the only point where the notation
    differs from Gibaud & Vignaud.
    :param n: array of refractive indices n = 1 - \\delta + i \beta of all layers, so
    n[0] is usually 1.
    :param lam: x-ray wavelength in nm
    :param thetas: array of incidence angles in rad
    :param thick: thicknesses in nm, len(thick) = N-2, since layer 0 and layer N are
    assumed infinite
    :param rough: rms roughness in nm, len(rough) = N-1 (number of interfaces)
    :param evaluation_positions: positions (in nm) at which the electric field should
    be evaluated. Given in distance
           from the surface, with the axis pointing away from the layer (i.e. negative
           positions are within the stack)
    :param pol: polarization (either 1 for s-polarization or 0 for p-polarization)
    :return: (reflec, trans, pos_reflec, pos_trans)
    """
    P = len(evaluation_positions)
    k2, n2, k2n2, s2h, N, T = _precompute(n, lam, thetas, thick, rough, pol)
    k2_x = k2 * np.square(
        np.cos(thetas)
    )  # k_x is conserved due to snell's law, i.e. only dependent on theta

    k_z = np.empty((T, N + 2), dtype=np.complex128)

    for kt in range(T):
        for kl in range(N + 2):
            k_z[kt][kl] = -np.sqrt(
                k2n2[kl] - k2_x[kt]
            )  # k_z is different for each layer.

    # calculate absolute interface positions from thicknesses
    Z = np.empty(N + 1, dtype=np.float64)
    Z[0] = 0.0
    cs = -np.cumsum(thick)
    for i in range(N):
        Z[i + 1] = cs[i]

    # preallocate whole result arrays
    rs = np.empty((T, N + 2), dtype=np.complex128)
    ts = np.empty((T, N + 2), dtype=np.complex128)

    # first calculate the fields at the interfaces
    for kt in prange(T):
        # preallocate temporary arrays
        mm12 = np.empty(N + 1, dtype=np.complex128)
        mm22 = np.empty(N + 1, dtype=np.complex128)
        _fields_positions_inner(
            thick=thick,
            n2=n2,
            s2h=s2h,
            kt=kt,
            rs=rs,
            ts=ts,
            mm12=mm12,
            mm22=mm22,
            N=N,
            k_z=k_z,
            pol=pol,
        )

    pos_rs = np.empty((T, P), dtype=np.complex128)
    pos_ts = np.empty((T, P), dtype=np.complex128)
    # now calculate the fields at the given evaluation positions
    for kp in prange(P):
        _fields_positions_inner_positions(
            kp=kp,
            pos=evaluation_positions[kp],
            Z=Z,
            pos_rs=pos_rs,
            pos_ts=pos_ts,
            k_z=k_z,
            ts=ts,
            rs=rs,
            T=T,
        )

    return rs, ts, pos_rs, pos_ts


def _simple_test():
    n_layers = 1001
    n = np.array([1] + [1 - 1e-5 + 1e-6j, 1 - 2e-5 + 2e-6j] * int((n_layers - 1) / 2))
    thick = np.array([0.1] * (n_layers - 2))
    rough = np.array([0.02] * (n_layers - 1))
    wl = 0.15
    ang_deg = np.linspace(0.1, 2.0, 10001)
    ang = np.deg2rad(ang_deg)
    r, t = reflec_and_trans(n, wl, ang, thick, rough)
    rp, tp = reflec_and_trans_parallel(n, wl, ang, thick, rough)
    assert np.all(r == rp)
    assert np.all(t == tp)

    rs, ts = fields(n, wl, ang, thick, rough, pol=0)
    rsp, tsp = fields(n, wl, ang, thick, rough, pol=0)
    assert np.all(rs == rsp)
    assert np.all(ts == tsp)

    evaluation_positions = np.linspace(-20, 0, 101)
    rs, ts, pos_rs, pos_ts = fields_at_positions(
        n, wl, ang, thick, rough, evaluation_positions
    )
    rsp, tsp, pos_rsp, pos_tsp = fields_at_positions_parallel(
        n, wl, ang, thick, rough, evaluation_positions
    )
    assert np.all(rs == rsp)
    assert np.all(ts == tsp)
    assert np.all(pos_rs == pos_rsp)
    assert np.all(pos_ts == pos_tsp)


if __name__ == "__main__":
    _simple_test()
