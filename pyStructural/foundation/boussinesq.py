import numpy as np


def boussinesq_pnt(point_load: float, depth: float, hor_dist: float) -> float:
    """
    Calculates the vertical stress at a given depth in the soil due to a
    point load applied at a specific horizontal distance.

    This function uses the Boussinesq equation for a point load, which is
    a fundamental solution in soil mechanics to determine the stress
    distribution beneath a point load.

    :param point_load: float
        The magnitude of the applied point load (in units of force).
    :param depth: float
        The depth at which the vertical stress is to be calculated
        (in units of length).
    :param hor_dist: float
        The horizontal distance from the point load to the point where the
        stress is being calculated (in units of length).

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
    return 0.477464829275686 * point_load * depth ** 3 * (hor_dist ** 2 + depth ** 2) ** -2.5


def boussinesq_udl_circle(udl: float, depth: float, radius: float) -> float:
    """
    Calculates the vertical stress at the center of a circular foundation due to
    a uniformly distributed load (UDL) at a given depth.

    The function computes the stress using an analytical solution derived from
    the Boussinesq equation for a uniformly distributed load over a circular
    area.

    :param udl: float
        The uniformly distributed load applied to the foundation
        (in units of force per area).
    :param depth: float
        The depth at which the vertical stress is to be calculated
        (in units of length).
    :param radius: float
        The radius of the circular foundation (in units of length).

    :return: float
        The vertical stress at the specified depth directly beneath the center
        of the circular foundation (in units of force per area).

    The analytical solution for the vertical stress σ at the center of a
    circular foundation due to a uniformly distributed load q is given by:
        σ = q * (1 - (1 + (radius/depth)^2)^(-3/2))

    This formula is derived from the integration of the Boussinesq equation over
    the area of the circular foundation.
    """
    if np.isclose(depth, 0):
        return udl
    return udl * (1 - (1 + (radius / depth) ** 2) ** -1.5)


def boussinesq_udl_rect_corner(udl, depth, width_x, width_z, n=6):
    """
    Calculates the vertical stress at the corner of a rectangular foundation due
    to a uniformly distributed load (UDL).

    :param udl: float
        The uniformly distributed load applied to the foundation
        (in units of force per area).
    :param depth: float
        The depth at which the vertical stress is to be calculated
        (in units of length).
    :param width_x: float
        The length of the rectangular foundation along the x-axis
        (in units of length).
    :param width_z: float
        The length of the rectangular foundation along the z-axis
        (in units of length).
    :param n: int, optional (default=6)
        The number of Gauss-Legendre quadrature points for the numerical
        integration.

    :return: float
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
    if np.isclose(depth, 0):
        return udl / 4

    # Take the quarter of the circle up to the smaller width, as there is an analytical solution
    radius = width_x if width_x < width_z else width_z
    sigma1 = 0.25 * udl * (1 - (1 + (radius / depth) ** 2) ** -1.5)

    # Calculate the integral of the remainder (rectangle - circle)
    width, length = (width_x, width_z) if width_x < width_z else (width_z, width_x)

    # Using pre-calculated Gauss-Legendre points for n = 6 (default) and n = 10 to speed up the process
    if n == 6:
        xsi = [-0.932469514203152, -0.661209386466265, -0.238619186083197,
               0.238619186083197, 0.661209386466265, 0.932469514203152]
        w = [0.171324492379163, 0.360761573048139, 0.467913934572689,
             0.467913934572689, 0.360761573048139, 0.171324492379163]
    elif n == 10:
        xsi = [-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.43339539412924716,
               -0.14887433898163122, 0.14887433898163122, 0.43339539412924716, 0.6794095682990244, 0.8650633666889845,
               0.9739065285171717]
        w = [0.06667134430868371, 0.14945134915058053, 0.21908636251598218, 0.26926671930999174, 0.2955242247147529,
             0.2955242247147529, 0.26926671930999174, 0.21908636251598218, 0.14945134915058053, 0.06667134430868371]
    else:
        xsi, w = np.polynomial.legendre.leggauss(n)

    # The coordinates x and z below correspond to the length and width coordinates, respectively
    zm = width / 2
    zr = width / 2
    integ = 0
    for i in range(n):
        zi = zm + zr * xsi[i]
        x0 = np.sqrt((width ** 2 - zi ** 2))
        xf = length
        xm = (xf + x0) / 2
        xr = (xf - x0) / 2
        for j in range(n):
            xj = xm + xr * xsi[j]
            integ += zr * xr * w[i] * w[j] * ((xj ** 2 + zi ** 2 + depth ** 2) ** -2.5)

    sigma2 = 0.477464829275686 * depth ** 3 * udl * integ
    return sigma1 + sigma2


def boussinesq_udl_rect_cent(udl, depth, width_x, width_z, n=6):
    """
    Calculates the vertical stress at the centre of a rectangular foundation due
    to a uniformly distributed load (UDL).

    :param udl: float
        The uniformly distributed load applied to the foundation
        (in units of force per area).
    :param depth: float
        The depth at which the vertical stress is to be calculated
        (in units of length).
    :param width_x: float
        The length of the rectangular foundation along the x-axis
        (in units of length).
    :param width_z: float
        The length of the rectangular foundation along the z-axis
        (in units of length).
    :param n: int, optional (default=6)
        The number of Gauss-Legendre quadrature points for the numerical
        integration.

    :return: float
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
    return 4 * boussinesq_udl_rect_corner(udl, depth, width_x/2, width_z/2, n)
