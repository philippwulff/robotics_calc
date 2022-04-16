import sympy as sy
from sympy import Matrix, Symbol, sin, cos, latex, simplify
from IPython.display import display, Math
from sympy.physics.mechanics import dynamicsymbols, init_vprinting


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
    if isinstance(d, str):
        d = Symbol(d)

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
                joint_params.append(Symbol(fr"\dot {param}"))
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
            display(Math("{}" f"^{i+1}{latex(Symbol('omega'))}_{i+1} = {latex(omega)}"))
            display(Math("{}" f"^{i + 1}v_{i + 1} = {latex(v)}"))

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
    if simple:
        jacobian = sy.simplify(jacobian)
    display(Math("{}" f"^{len(dh_params)}J = {latex(jacobian)}"))
    return jacobian


def prop_force_torque(dh_params, joint_points, end_force_torque, verbose=True, simple=True):
    """Propagates static forces and torques from the end-effector to the base of the manipulator.

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


def newton_euler(dh_params, joint_points, m_center_points, v_dot_0, link_m, link_I,
                 end_force_torque=sy.Matrix([0, 0, 0, 0, 0, 0]), verbose=True, simple=True):
    """Compute joint torques/ forces via the Newton-Euler-Method.

    Parameters
    ----------
    dh_params: list of lists with D-H-parameters (alpha, a, d, theta)
    joint_points: list of sympy.Matrix vectors
        The joint points measured in the coordinate frame of the previous joint. Ordered from base to end-effector.
    end_force_torque: sympy.Matrix of shape 6x1
        The force-torque vector at the end-effector/ last link.
    verbose: Whether to print the intermediary results.
    simple: Whether to simplify results.
    m_center_points: list of sympy.Matrix
        The center of mass of link i in the coordinate frame of link i.
    v_dot_0: sympy.Matrix (3 x 1 vector)
        The acceleration of the base frame. Usually set to -G.
    link_m: list
        The mass of each link.
    link_I: list of sympy.Matrix (3 x 3 matrices)
        The inertia tensor of each link.

    Returns
    -------
        sympy.Matrix of shape (number of joints, 1)
        The joint torques/ forces.
    """

    transforms = [homo_transf(*dhp) for dhp in dh_params]
    rot_mats = [t[:3, :3] for t in transforms]

    joint_params_fo = []            # first-order derivative
    joint_params_so = []            # second-order derivative
    for joint in dh_params:
        for param in joint:
            if "d" in str(param):
                joint_params_fo.append(Symbol(r"\dot{" + param + "}"))
                joint_params_so.append(Symbol(r"\ddot{" + param + "}"))
            elif "theta" in str(param):
                joint_params_fo.append(Symbol(fr"\dot\{param}"))
                joint_params_so.append(Symbol(fr"\ddot\{param}"))

    # base is fixed
    omega = Matrix([0, 0, 0])
    omega_dot = Matrix([0, 0, 0])
    v_dot = v_dot_0
    # Forces and torques acting on the centre of mass of each link
    link_F = []
    link_N = []

    if verbose:
        print("Forward phase:")

    # forward phase
    for i in range(len(rot_mats)):
        # If the joint is neither revolute not prismatic
        if i >= len(joint_params_fo):
            # Transpose inverts rotation
            v_dot = rot_mats[i].T @ (omega.cross(joint_points[i]) + v_dot)
            omega_dot = rot_mats[i].T @ omega_dot
            omega = rot_mats[i].T @ omega
        # Revolute
        elif "theta" in str(joint_params_fo[i]):
            v_dot = rot_mats[i].T @ (omega_dot.cross(joint_points[i]) + omega.cross(omega.cross(joint_points[i])) +
                                     v_dot)
            omega_dot = rot_mats[i].T @ omega_dot + \
                        rot_mats[i].T @ omega.cross(joint_params_fo[i] * Matrix([0, 0, 1])) + \
                        joint_params_so[i] * Matrix([0, 0, 1])
            omega = rot_mats[i].T @ omega + joint_params_fo[i] * Matrix([0, 0, 1])
        # Prismatic
        elif "d" in str(joint_params_fo[i]):
            v_dot = rot_mats[i].T @ (omega_dot.cross(joint_points[i]) + omega.cross(omega.cross(joint_points[i])) +
                                     v_dot)
            omega_dot = rot_mats[i].T @ omega_dot
            omega = rot_mats[i].T @ omega
            v_dot += 2 * omega.cross(joint_params_fo[i] * Matrix([0, 0, 1])) + joint_params_so[i] * Matrix([0, 0, 1])

        # calculate velocity of center of mass and forces and torques acting on link i+1

        v_c_dot = omega_dot.cross(m_center_points[i]) + omega.cross(omega.cross(m_center_points[i])) + v_dot
        F = link_m[i] * v_c_dot
        N = link_I[i] @ omega_dot + omega.cross(link_I[i] @ omega)
        link_F.append(F)
        link_N.append(N)

        if verbose:
            if simple:
                omega = simplify(omega)
                omega_dot = simplify(omega_dot)
                v_dot = simplify(v_dot)
            print(f"i = {i}:")
            display(Math("{}" f"^{i + 1}" + r"\dot{v}_" + f"{i + 1} = {latex(v_dot)}"))
            display(Math("{}" f"^{i+1}{latex(Symbol('omega'))}_{i+1} = {latex(omega)}"))
            display(Math("{}" f"^{i + 1}" + r"\dot{" + latex(Symbol("omega")) + "}_" + f"{i + 1} = {latex(omega_dot)}"))
            display(Math("{}" f"^{i + 1}" + r"\dot{v}_{C_" + f"{i + 1}" + "}" + f" = {latex(v_dot)}"))
            display(Math("{}" f"^{i+1}F_{i+1} = {latex(F)}"))
            display(Math("{}" f"^{i+1}N_{i+1} = {latex(N)}"))

    if verbose:
        print("Backwards phase:")

    # Backwards
    force = end_force_torque[:3, :]
    torque = end_force_torque[3:, :]
    forces = []
    torques = []

    for i in reversed(range(0, len(rot_mats))):

        torque = link_N[i] + rot_mats[i] @ torque + m_center_points[i].cross(link_F[i]) + \
                 joint_points[i].cross(rot_mats[i] @ force)
        force = rot_mats[i] @ force + link_F[i]

        if simple:
            force = simplify(force)
            torque = simplify(torque)

        forces.append(force)
        torques.append(torque)

        if verbose:
            print(f"i = {i+1}:")
            display(Math("{}" f"^{i+1}f_{i+1} = {latex(force)}"))
            display(Math("{}" f"^{i+1}n_{i+1} = {latex(torque)}"))

    print("Joint torques/ forces:")

    tau = []
    for i in range(len(joint_params_fo)):
        # Check for theta first, because e.g. "\dot\theta_1" also contains "d"
        if "theta" in str(joint_params_fo[i]):
            tau.append(torques[-i-1][2, 0])
        elif "d" in str(joint_params_fo[i]):
            tau.append(forces[-i-1][2, 0])

    tau = sy.Matrix(tau)

    display(Math(r"\tau = " + latex(sy.Matrix([sy.Symbol(f"tau_{i}") for i in range(1, len(tau)+1)])) + f" = {latex(tau)}"))

    return tau


def lagrange(dh_params, m_center_points_0, link_m, link_I, verbose=True, simple=True):

    # TODO

    # Convert thetas and ds in dh_params to dynamic symbols which allow autodiff
    joint_params = []
    dh_rows = []
    for joint in dh_params:
        row = []
        for param in joint:
            if "d" in str(param) or "theta" in str(param):
                dyn_sym = dynamicsymbols(str(param))
                row.append(dyn_sym)
                joint_params.append(dyn_sym)
            elif isinstance(param, str):
                row.append(Symbol(param))
            else:
                row.append(param)
        dh_rows.append(row)
    dh_params = dh_rows

    transforms = [homo_transf(*dhp) for dhp in dh_params]
    rot_mats = [t[:3, :3] for t in transforms]

    return dh_params, joint_params