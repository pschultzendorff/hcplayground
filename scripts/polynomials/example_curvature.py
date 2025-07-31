import pathlib
import sys

src_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "src" / "polynomials"
sys.path.append(str(src_dir))

from curvature import HomotopyCurvature


def f(x):
    return x**4 - 15 * x**3 + 3 * x - 20


def f_prime(x):
    return 4 * x**3 - 45 * x**2 + 3


def f_double_prime(x):
    return 12 * x**2 - 90 * x


def g(x):
    return x**2 - 20


def g_prime(x):
    return 2 * x


def g_double_prime(x):
    return 2


hc = HomotopyCurvature(
    f=f,
    g=g,
    f_prime=f_prime,
    g_prime=g_prime,
    f_double_prime=f_double_prime,
    g_double_prime=g_double_prime,
    x0=0.5,
)
hc.plot_curvature()
