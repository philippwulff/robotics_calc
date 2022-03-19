import numpy as np
import sympy
from sympy import N, Matrix, Symbol, sin, cos, pprint, latex, init_printing, simplify
import math
from IPython.display import display, Math


def rad(degrees):
    return degrees/180 * sympy.pi


def homo_transf(alpha, a, d, theta):
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
    transforms = [homo_transf(*dhp) for dhp in dh_params]

    if verbose:
        for i, transf in enumerate(transforms):
            index = "{}" + f"^{i}_{i + 1}"
            display(Math(f"{index}T = {latex(transf)}"))

    return transforms


def full_homo_transf(transforms, verbose=True, simple=True):
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
    for i in range(len(rot_mats)):
        if "theta" in str(joint_params[i]):
            v = rot_mats[i] @ (omega.cross(joint_points[i]))
            omega = rot_mats[i] @ omega + joint_params[i] * Matrix([0, 0, 1])
        elif "d" in str(joint_params[i]):
            v = rot_mats[i] @ (omega.cross(joint_points[i])) + joint_params[i] * Matrix([0, 0, 1])
            omega = rot_mats[i] @ omega

        if verbose:
            if simple:
                omega = simplify(omega)
                v = simplify(v)
            display(Math("{}" f"^{i + 1}v_{i + 1} = {latex(v)}"))
            display(Math("{}" f"^{i+1}{latex(Symbol('omega'))}_{i+1} = {latex(omega)}"))

