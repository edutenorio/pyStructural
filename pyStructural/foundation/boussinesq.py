import numpy as np
from scipy.interpolate import interpn


def Ip(hor_dist: float, depth: float) -> float:
    """
    Influence factor for the vertical stress calculation at a given depth in the
    soil due to a point load applied at a specific horizontal distance on the
    surface.

    :param hor_dist: float
        The horizontal distance from the point load to the point where the
        stress is being calculated (in units of length).
    :param depth: float
        The depth at which the vertical stress is to be calculated
        (in units of length).
    :return: float
        The Influence factor Ip, calculated so that the vertical stress is:
        σ = Q / a^2 * Ip
    """
    if np.isclose(depth, 0):
        if np.isclose(hor_dist, 0):
            return 0.477464829275686
        return 0.0
    return 0.477464829275686 * (1 + (hor_dist / depth) ** 2) ** -2.5


def boussinesq_pnt(point_load: float, hor_dist: float, depth: float) -> float:
    """
    Calculates the vertical stress at a given depth in the soil due to a
    point load applied at a specific horizontal distance.

    This function uses the Boussinesq equation for a point load, which is
    a fundamental solution in soil mechanics to determine the stress
    distribution beneath a point load.

    :param point_load: float
        The magnitude of the applied point load (in units of force).
    :param hor_dist: float
        The horizontal distance from the point load to the point where the
        stress is being calculated (in units of length).
    :param depth: float
        The depth at which the vertical stress is to be calculated
        (in units of length).

    :return: float
        The vertical stress at the specified depth and horizontal distance
        (in units of force per unit area).

    The Boussinesq equation for the vertical stress σ due to a point load Q is:
        σ = (3 * Q * a^3) / (2 * π * (r^2 + a^2)^(5/2))
    where:
        - Q is the point load.
        - a is the depth.
        - r is the horizontal distance, calculated as sqrt(x^2 + z^2).
    """
    # return 0.477464829275686 * point_load * depth ** 3 * (hor_dist ** 2 + depth ** 2) ** -2.5
    return point_load * Ip(hor_dist=hor_dist, depth=depth)


def Ic(radius: float, depth: float) -> float:
    """
    Influence factor for the vertical stress calculation at the center of a
    circular foundation due to a uniformly distributes load (UDL) at a given
    depth.

    Ic = 1 - (1 + (radius/depth)^2)^(-3/2)

    :param radius: float
        The radius of the circular foundation (in units of length).
    :param depth: float
        The depth at which the vertical stress is to be calculated
        (in units of length).
    :return: float
        The Influence factor Ic, calculated so that the vertical stress is:
        σ = q * Ic
    """
    if np.isclose(depth, 0.0):
        return 1.0
    return 1 - (1 + (radius / depth) ** 2) ** -1.5


def boussinesq_udl_circle(udl: float, radius: float, depth: float) -> float:
    """
    Calculates the vertical stress at the center of a circular foundation due to
    a uniformly distributed load (UDL) at a given depth.

    The function computes the stress using an analytical solution derived from
    the Boussinesq equation for a uniformly distributed load over a circular
    area.

    :param udl: float
        The uniformly distributed load applied to the foundation
        (in units of force per area).
    :param radius: float
        The radius of the circular foundation (in units of length).
    :param depth: float
        The depth at which the vertical stress is to be calculated
        (in units of length).

    :return: float
        The vertical stress at the specified depth directly beneath the center
        of the circular foundation (in units of force per area).

    The analytical solution for the vertical stress σ at the center of a
    circular foundation due to a uniformly distributed load q is given by:
        σ = q * (1 - (1 + (radius/depth)^2)^(-3/2))

    This formula is derived from the integration of the Boussinesq equation over
    the area of the circular foundation.
    """
    return udl * Ic(radius=radius, depth=depth)


def Ir_calc(m: float, n: float, n_pts: int) -> float:
    """
    Calculate the Influence factor Ir with the Gauss-Legendre quadrature.

    Used to generate and store a table of values. For performance, the functions
    don't calculate the Ir values every time, but just look up the
    pre-calculated ones.
    """
    # Take m as the largest for the calculations
    m, n = (m, n) if m > n else (n, m)

    # The quarter of circle with radius equal to the smaller dimension n has an analytical solution and can be
    # calculated with the Ic function. We only need to numerically integrate the remainder of the area, the
    # (rectangle - quarter of circle).

    # Take the Gauss-Legendre weights and coordinates
    gauleg_x, gauleg_w = np.polynomial.legendre.leggauss(n_pts)

    # The coordinates xsi and eta below correspond to (length/depth) and (width/depth) coordinates, respectively
    eta_m = n / 2
    eta_r = n / 2
    integrand = 0
    for i in range(n_pts):
        eta_i = eta_m + eta_r * gauleg_x[i]
        xsi_0 = np.sqrt(n * n - eta_i * eta_i)
        xsi_f = m
        xsi_m = (xsi_f + xsi_0) / 2
        xsi_r = (xsi_f - xsi_0) / 2
        for j in range(n_pts):
            xsi_j = xsi_m + xsi_r * gauleg_x[j]
            integrand += eta_r * xsi_r * gauleg_w[i] * gauleg_w[j] * ((1 + xsi_j * xsi_j + eta_i * eta_i) ** -2.5)

    return 0.25 * Ic(radius=n, depth=1) + 0.477464829275686 * integrand


def Ir_lookup_single(m: float, n: float) -> float:
    m_vec, n_vec, ir_table = np.load('Ir_table.npz').values()
    return interpn((m_vec, n_vec), ir_table, (m, n), method='linear', bounds_error=False, fill_value=None)[0]


def Ir_lookup_array(mn: np.array) -> np.array:
    m_vec, n_vec, ir_table = np.load('Ir_table.npz').values()
    return interpn((m_vec, n_vec), ir_table, mn, method='linear', bounds_error=False, fill_value=None)


def Ir(*args) -> float:
    """
    Influence factor for the vertical stress calculation at the corner of a
    rectangular foundation due to a uniformly distributed load (UDL) at a given
    depth.

    :param m: float or array
        The ratio length / depth (dimensionless) or an
        array of pairs (length/depth, width/depth)
    :param n: float
        The ratio width / depth (dimensionless). Only used if the first argument
        is a float

    :return: float or array
        The result of the integral:
        Ir = ∫∫ [3 / (2 * π) * (1 + ξ^2 + η^2)^(-5/2)] dξ dη
    """
    if len(args) == 1:
        # Single argument shall be an array of (m, n) values
        return Ir_lookup_array(args[0])
    if len(args) == 2:
        # Two arguments shall be a single pair (m, n)
        return Ir_lookup_single(*args)
    raise TypeError(f'Invalid argument for the Ir function: {args}')


def boussinesq_udl_rect_corner(udl: float, width_x: float, width_z: float, depth):
    """
    Calculates the vertical stress at the corner of a rectangular foundation due
    to a uniformly distributed load (UDL).

    :param udl: float
        The uniformly distributed load applied to the foundation
        (in units of force per area).
    :param width_x: float
        The length of the rectangular foundation along the x-axis
        (in units of length).
    :param width_z: float
        The length of the rectangular foundation along the z-axis
        (in units of length).
    :param depth: float or array
        The depth at which the vertical stress is to be calculated
        (in units of length).

    :return: float or array
        The vertical stress at the point (in units of force per area).

    The Boussinesq equation for a point load Q is:
        σ = (3 * a^3) / (2 * π * (r^2 + a^2)^(5/2)) * Q
    where r is the horizontal distance to the point Q, r = sqrt(x^2 + z^2) and
    a is the depth.

    For a uniformly distributed load q over a rectangular area:
        σ = ∫∫_Area [(3 * a^3) / (2 * π * (x^2 + z^2 + a^2)^(5/2)) * q] dx dz

    To solve the integral, the domain is divided into a quarter of circle, with
    the centre located in the corner of the foundation and a radius equal to the
    shorter side. This part has an analytical solution. The remainder is
    calculated numerically using Gauss-Legendre quadrature.
    """
    if np.isclose(width_x, 0) or np.isclose(width_z, 0):
        return 0.0
    if np.isscalar(depth) and np.isclose(depth, 0):
        return udl / 4
    if np.isscalar(depth):
        return udl * Ir(width_x/depth, width_z/depth)
    if np.isclose(depth[0], 0):
        depth[0] = max([width_x, width_z]) / 100
    return udl * Ir(np.fromiter(((width_x/a, width_z/a) for a in depth), dtype=(float, 2)))


def boussinesq_udl_rect_cent(udl, width_x, width_z, depth):
    """
    Calculates the vertical stress at the centre of a rectangular foundation due
    to a uniformly distributed load (UDL).

    :param udl: float
        The uniformly distributed load applied to the foundation
        (in units of force per area).
    :param width_x: float
        The length of the rectangular foundation along the x-axis
        (in units of length).
    :param width_z: float
        The length of the rectangular foundation along the z-axis
        (in units of length).
    :param depth: float or array
        The depth at which the vertical stress is to be calculated
        (in units of length).

    :return: float or array
        The vertical stress at the point (in units of force per area).

    The Boussinesq equation for a point load Q is:
        σ = (3 * a^3) / (2 * π * (r^2 + a^2)^(5/2)) * Q
    where r is the horizontal distance to the point Q, r = sqrt(x^2 + z^2) and
    a is the depth.

    For a uniformly distributed load q over a rectangular area:
        σ = ∫∫_Area [(3 * a^3) / (2 * π * (x^2 + z^2 + a^2)^(5/2)) * q] dx dz

    This functions uses the 'boussinesq_udl_rect_corner' function, solve it for
    one quarter of the rectangle and multiply the result by 4x.
    """
    return 4 * boussinesq_udl_rect_corner(udl, 0.5*width_x, 0.5*width_z, depth)
