import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy import symbols, Eq, solve, parse_expr
import scipy
from math import sin, cos

PI = np.pi
G = 9.81


class Bounds:
    def __init__(self, O, T):
        self.O = O
        self.T = T


T = 3
x_b = Bounds(3, 2)
y_b = Bounds(2, 1)
z_b = Bounds(2, 1)
xd_b = Bounds(1, 3)
yd_b = Bounds(1, 2)
zd_b = Bounds(1, 0)
th_b = Bounds(PI / 6, PI / 3)
ph_b = Bounds(PI / 3, PI / 6)
thd_b = Bounds(PI / 6, PI / 3)
phd_b = Bounds(PI / 3, PI / 6)
ksi1_b = Bounds(G, G)
ksi2 = Bounds(0, 0)

x, y, z, xd, yd, zd, ksi1, th, ph, g = symbols("x y z xd yd zd ksi1 th ph g")
h1 = parse_expr("x")
h2 = parse_expr("y")
h3 = parse_expr("z")
h1d = parse_expr("xd")
h2d = parse_expr("yd")
h3d = parse_expr("zd")
h1dd = parse_expr("-ksi1 * sin(th)")
h2dd = parse_expr("ksi1 * cos(th) * sin(ph)")
h3dd = parse_expr("ksi1 * cos(th) * cos(ph) - g")


def convert_to_bound_contitions(h, hd, hdd):
    boundary_conditions = []

    boundary_conditions.append((0, h.subs({x: x_b.O, y: y_b.O, z: z_b.O}), 0))
    boundary_conditions.append((T, h.subs({x: x_b.T, y: y_b.T, z: z_b.T}), 0))

    boundary_conditions.append((0, hd.subs({xd: xd_b.O, yd: yd_b.O, zd: zd_b.O}), 1))
    boundary_conditions.append((T, hd.subs({xd: xd_b.T, yd: yd_b.T, zd: zd_b.T}), 1))

    boundary_conditions.append((0, hdd.subs({ksi1: ksi1_b.O, th: th_b.O, ph: ph_b.O, g: G}), 2))
    boundary_conditions.append((T, hdd.subs({ksi1: ksi1_b.T, th: th_b.T, ph: ph_b.T, g: G}), 2))

    boundary_conditions.append((0, 0, 3))
    boundary_conditions.append((T, 0, 3))
    print(boundary_conditions)
    return boundary_conditions


def polynomial_coefficients(degree, boundary_conditions):
    rows = (degree + 1)  # number of variables and equations
    A = np.zeros((rows, rows))
    b = np.zeros(rows)

    for i, (t, value, order) in enumerate(boundary_conditions):
        for j in range(order, degree + 1):
            A[i, j] = np.math.factorial(j) / np.math.factorial(j - order) * t ** (j - order)

        b[i] = value

    coefficients = np.linalg.solve(A, b)
    return coefficients


def derivative_coeffs(coeffs):
    return [(i + 1) * coeff for i, coeff in enumerate(coeffs[1:])]


def make_poly(coefs):
    def poly(x):
        value = 0
        for i in range(len(coefs)):
            value += coefs[i] * x ** i
        return value

    return poly


x_coefs = polynomial_coefficients(7, convert_to_bound_contitions(h1, h1d, h1dd))
y_coefs = polynomial_coefficients(7, convert_to_bound_contitions(h2, h2d, h2dd))
z_coefs = polynomial_coefficients(7, convert_to_bound_contitions(h3, h3d, h3dd))

x_func = make_poly(x_coefs)
y_func = make_poly(y_coefs)
z_func = make_poly(z_coefs)

h1dd_func = make_poly(derivative_coeffs(derivative_coeffs(x_coefs)))
h2dd_func = make_poly(derivative_coeffs(derivative_coeffs(y_coefs)))
h3dd_func = make_poly(derivative_coeffs(derivative_coeffs(z_coefs)))

t_space = np.linspace(0, T, 200)
x_values = [x_func(t) for t in t_space]
y_values = [y_func(t) for t in t_space]
z_values = [z_func(t) for t in t_space]
phi_values = [np.arctan(h2dd_func(t) / (h3dd_func(t) + G)) for t in t_space]
theta_values = [np.cos(np.arctan(h2dd_func(t) / (h3dd_func(t) + G))) * np.arctan(h1dd_func(t) / (h3dd_func(t) + G)) for
                t in t_space]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim3d([min(x_values) - 1, max(x_values) + 1])
ax.set_ylim3d([min(y_values) - 1, max(y_values) + 1])
ax.set_zlim3d([min(z_values) - 1, max(z_values) + 1])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
line, = ax.plot(x_values[0], y_values[0], z_values[0], 'o', markersize=2, color="black")
dots_x = []
dots_y = []
dots_z = []
line2, = ax.plot(x_values[0], y_values[0], z_values[0], 'o', markersize=2, color="red", alpha=0.3)
part1, = ax.plot(x_values[0], y_values[0], z_values[0], color="blue")
part2, = ax.plot(x_values[0], y_values[0], z_values[0], color="blue")
part3, = ax.plot(x_values[0], y_values[0], z_values[0], color="red")
ax.plot(1, 0, 0)
i = 0


def euler_angles_to_rotation_matrix(euler_angles):
    r, p, y = euler_angles

    cr = cos(r)
    sr = sin(r)
    cp = cos(p)
    sp = sin(p)
    cy = cos(y)
    sy = sin(y)
    # Create rotation matrix for each axis
    R = np.array([
        [cp * cr, sr * sp, -sp],
        [cr * sp * sy - sr * cy, sr * sp * sy + cr * cy, cp * sy],
        [cr * sp * sy + sr * cy, sr * sp * cy - cr * sy, cp * cy]
    ])

    # Combine the individual rotations into a single matrix
    return R


def rotate_vector(vector, euler_angles):
    R = euler_angles_to_rotation_matrix(euler_angles)
    return R @ vector


def get_edg(vec, point, angles):
    rotated = rotate_vector(vec, angles)
    right_point = point + rotated
    left_point = point - rotated
    return left_point, right_point


def get_edges(point, height, angle):
    l = 1.5
    left_point = point - l * cos(angle), height + l * sin(angle)
    right_point = point + l * cos(angle), height - l * sin(angle)
    return left_point, right_point


def update(frame):
    global i, dots_x, dots_y, dots_z
    length = 1.5
    height = 0.5
    cur_point = np.array([x_values[frame - 1], y_values[frame - 1], z_values[frame - 1]])
    cur_angle = (0, theta_values[frame - 1], phi_values[frame - 1])
    l, r = get_edg(np.array([length, 0, 0]), cur_point, cur_angle)
    part1.set_data([l[0], r[0]], [l[1], r[1]])
    part1.set_3d_properties([l[2], r[2]])
    dots_x += [l[0], r[0]]
    dots_y += [l[1], r[1]]
    dots_z += [l[2], r[2]]

    l, r = get_edg(np.array([0, length, 0]), cur_point, cur_angle)
    part2.set_data([l[0], r[0]], [l[1], r[1]])
    part2.set_3d_properties([l[2], r[2]])
    dots_x += [l[0], r[0]]
    dots_y += [l[1], r[1]]
    dots_z += [l[2], r[2]]

    l, r = get_edg(np.array([0, 0, height]), cur_point, cur_angle)
    part3.set_data([cur_point[0], r[0]], [cur_point[1], r[1]])
    part3.set_3d_properties([cur_point[2], r[2]])

    line.set_data(x_values[:frame], y_values[:frame])
    line.set_3d_properties(z_values[:frame])
    line2.set_data(dots_x[-40:], dots_y[-40:])
    line2.set_3d_properties(dots_z[-40:])
    i += 1
    print(i)
    return line,


ani = FuncAnimation(fig, update, frames=len(x_values), blit=True)
ani.save("animation.gif", writer="imagemagick", fps=40)
