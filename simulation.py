import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh_tridiagonal


def build_grid_2d(N):
    X, Y = np.mgrid[0:1:N*1j, 0:1:N*1j]
    return X, Y


def gaussian_potential_2d(X, Y, x0=0.3, y0=0.3, sigma=0.1, amplitude=1.0):
    V = amplitude * np.exp(-(X - x0)**2 / (2 * sigma**2)) * np.exp(-(Y - y0)**2 / (2 * sigma**2))
    return V


def solve_stationary_2d(N=200, k=10, x0=0.3, y0=0.3, sigma=0.1, amplitude=1.0):
    X, Y = build_grid_2d(N)
    V = gaussian_potential_2d(X, Y, x0=x0, y0=y0, sigma=sigma, amplitude=amplitude)

    diag = np.ones(N)
    diags = np.array([diag, -2 * diag, diag])
    D = sparse.spdiags(diags, np.array([-1, 0, 1]), N, N)
    T = -0.5 * sparse.kronsum(D, D)
    U = sparse.diags(V.reshape(N**2), 0)
    H = T + U

    eigenvalues, eigenvectors = eigsh(H, k=k, which='SM')
    return X, Y, V, eigenvalues, eigenvectors


def get_mode_2d(eigenvectors, N, mode):
    psi = eigenvectors[:, mode].reshape((N, N))
    return psi


def build_grid_1d(Nx=301):
    x  = np.linspace(0, 1, Nx)
    dx = 1.0 / (Nx - 1)
    return x, dx


def gaussian_potential_1d(x, mu=0.5, sigma=0.05, amplitude=-1e4):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))


def initial_state_1d(x):
    return np.sqrt(2) * np.sin(np.pi * x)


def solve_time_basis(Nx=301, mu=0.5, sigma=0.05, amplitude=-1e4, n_modes=70):
    x, dx = build_grid_1d(Nx)
    psi0 = initial_state_1d(x)
    Vx = gaussian_potential_1d(x, mu=mu, sigma=sigma, amplitude=amplitude)

    # Matrice tridiagonale sur les points interieurs (bords nuls imposes)
    d = 1 / dx**2 + Vx[1:-1]
    e = -1 / (2 * dx**2) * np.ones(len(d) - 1)

    w, v = eigh_tridiagonal(d, e)

    E_js = w[:n_modes]
    psi_js = np.pad(v.T[:n_modes], [(0, 0), (1, 1)], mode='constant')
    cs = np.dot(psi_js, psi0)

    return x, psi0, Vx, E_js, psi_js, cs


def psi_t(x, E_js, psi_js, cs, t):
    psi = np.zeros_like(x, dtype=complex)
    for j in range(len(E_js)):
        psi += cs[j] * psi_js[j] * np.exp(-1j * E_js[j] * t)
    return psi


def density_t(x, E_js, psi_js, cs, t):
    psi = psi_t(x, E_js, psi_js, cs, t)
    return np.abs(psi)**2


def density_surface(x, E_js, psi_js, cs, t_vals):
    rho = np.zeros((len(t_vals), len(x)))
    for i, t in enumerate(t_vals):
        rho[i] = density_t(x, E_js, psi_js, cs, t)
    return rho