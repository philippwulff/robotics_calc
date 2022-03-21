import numpy as np
import sympy as sy
from sympy import N, Matrix, Symbol, sin, cos, pprint, latex, init_printing, simplify
import math
from IPython.display import display, Math


def rad(degrees):
    """Convert degrees to radians."""
    return degrees/180 * sy.pi


def homo_transf(alpha, a, d, theta):
    """Returns the homogeneous transform matrix. Input are Denavit-Hartenberg parameters of one link.
    All angles are in degrees.

    Parameters
    ----------
    alpha: float
        The angle from Zi to Zi+1 measured about Xi
    a: float
        The distance from Zi to Zi+1 measured along Xi
    d: float
        The distance from Xi-1 to Xi measured along Zi
    theta: float
        The angle from Xi-1 to Xi measured about Zi
    Returns
    -------
    sympy.Matrix
    """
    alpha = rad(alpha)
    if isinstance(theta, str):
        theta = Symbol(theta)
    else:
        theta = rad(theta)

    transf = Matrix(
        [
            [cos(theta), -sin(theta), 0, a],
            [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
            [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), cos(alpha)*d],
            [0, 0, 0, 1]
        ]
    )
    return simplify(transf)


def build_transf(dh_params, verbose=True):
    """Computes the homogeneous transforms for all links.

    Parameters
    ----------
    dh_params: list of lists
        Each sublist are the D-H-parameters for a given link (alpha, a, d, theta).
        Example: [[0, 1, 0, sympy.Symbol("theta_1")], [0, 2, 2, sympy.Symbol("theta_2")]]
    verbose: bool
        Wether to print the outputs.

    Returns
    -------
        List of transform matrices.
    """
    transforms = [homo_transf(*dhp) for dhp in dh_params]

    if verbose:
        for i, transf in enumerate(transforms):
            index = "{}" + f"^{i}_{i + 1}"
            display(Math(f"{index}T = {latex(transf)}"))

    return transforms


def full_homo_transf(transforms, verbose=True, simple=True):
    """Multiplies out homogeneous transforms from the left.

    Parameters
    ----------
    transforms: List of homogeneous transforms.
    verbose: Whether to print the intermediary and final outputs.
    simple: Whether to simplify the results.

    Returns
    -------
        sympy.Matrix
    """
    left_mat = transforms[0]
    for i in range(1, len(transforms)):
        left_mat = left_mat @ transforms[i]
        if verbose:
            index = "{}" + f"^0_{i + 1}"
            if simple:
                left_mat = simplify(left_mat)
            display(Math(f"{index}T = {latex(left_mat)}"))

    return left_mat


def prop_velo(dh_params, joint_points, verbose=True, simple=True):
    """Propagate velocities from the base to the end-effector of the manipulator.

    Parameters
    ----------
    dh_params: list of lists with D-H-parameters (alpha, a, d, theta)
    joint_points: list of sympy.Matrix vectors
        The joint points measured in the coordinate frame of the previous joint. Ordered from base to end-effector.
    verbose: Whether to print the intermediary results.
    simple: Whether to simplify results.

    Returns
    -------
        Final linear and angular velocity vectors
        joint_params: list of derivatives of joint parameters
    """
    transforms = [homo_transf(*dhp) for dhp in dh_params]
    rot_mats = [t[:3, :3] for t in transforms]

    joint_params = []
    for joint in dh_params:
        for param in joint:
            if "d" in str(param):
                joint_params.append(Symbol(fr"\dot\{param}"))
            elif "theta" in str(param):
                joint_params.append(Symbol(fr"\dot\{param}"))

    # base is fixed
    omega = Matrix([0, 0, 0])
    v = Matrix([0, 0, 0])
    for i in range(len(rot_mats)):
        # If joint is neither revolute not prismatic
        if i >= len(joint_params):
            # Transpose inverts rotation
            v = rot_mats[i].T @ (v + omega.cross(joint_points[i]))
            omega = rot_mats[i].T @ omega
        # Revolute
        elif "theta" in str(joint_params[i]):
            v = rot_mats[i].T @ (v + omega.cross(joint_points[i]))
            omega = rot_mats[i].T @ omega + joint_params[i] * Matrix([0, 0, 1])
        # Prismatic
        elif "d" in str(joint_params[i]):
            v = rot_mats[i].T @ (v + omega.cross(joint_points[i])) + joint_params[i] * Matrix([0, 0, 1])
            omega = rot_mats[i].T @ omega

        if verbose:
            if simple:
                omega = simplify(omega)
                v = simplify(v)
            display(Math("{}" f"^{i + 1}v_{i + 1} = {latex(v)}"))
            display(Math("{}" f"^{i+1}{latex(Symbol('omega'))}_{i+1} = {latex(omega)}"))

    return v, omega, joint_params


def comp_jacobian(dh_params, joint_points, verbose=True, simple=True):
    """Computes the Jacobian matrix.

    Parameters
    ----------
    dh_params: list of lists with D-H-parameters (alpha, a, d, theta)
    joint_points: list of sympy.Matrix vectors
        The joint points measured in the coordinate frame of the previous joint. Ordered from base to end-effector.
    verbose: Whether to print the intermediary results.
    simple: Whether to simplify results.

    Returns
    -------
        sympy.Matrix
        The Jacobian matrix of shape (6, number of joint parameters).
    """

    v, omega, joint_params = prop_velo(dh_params, joint_points, verbose=verbose)

    jacobian_rows = []
    for vec in (v, omega):
        for i in range(3):
            # Collect factors; expanding and not simplifying makes this more robust
            expr = sy.collect(sy.expand(vec[i, 0]), joint_params)
            coeff = sy.Matrix([expr.coeff(v) for v in joint_params]).T
            jacobian_rows.append(coeff)

    jacobian = Matrix.vstack(*jacobian_rows)
    display(Math("{}" f"^{len(dh_params)}J = {latex(jacobian)}"))
    if simple:
        jacobian = sy.simplify(jacobian)
    return jacobian


def prop_force_torque(dh_params, joint_points, end_force_torque, verbose=True, simple=True):
    """Propagates forces and torques from the end-effector to the base of the manipulator.

    Parameters
    ----------
    dh_params: list of lists with D-H-parameters (alpha, a, d, theta)
    joint_points: list of sympy.Matrix vectors
        The joint points measured in the coordinate frame of the previous joint. Ordered from base to end-effector.
    end_force_torque: sympy.Matrix of shape 6x1
        The force-torque vector at the end-effector/ last link.
    verbose: Whether to print the intermediary results.
    simple: Whether to simplify results.

    Returns
    -------
        None
    """
    transforms = [homo_transf(*dhp) for dhp in dh_params]
    rot_mats = [t[:3, :3] for t in transforms]

    force = end_force_torque[:3, :]
    torque = end_force_torque[3:, :]

    if verbose:
        n = len(rot_mats)
        display(Math("{}" f"^{n}f_{n} = {latex(force)}"))
        display(Math("{}" f"^{n}n_{n} = {latex(torque)}"))

    for i in reversed(range(1, len(rot_mats))):

        force = rot_mats[i] @ force
        torque = rot_mats[i] @ torque + joint_points[i].cross(force)

        if verbose:
            if simple:
                force = simplify(force)
                torque = simplify(torque)
            display(Math("{}" f"^{i}f_{i} = {latex(force)}"))
            display(Math("{}" f"^{i}n_{i} = {latex(torque)}"))

