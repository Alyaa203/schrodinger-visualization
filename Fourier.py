import os
import numpy as np
import matplotlib.pyplot as plt

_trapz = np.trapz if hasattr(np, 'trapz') else np.trapezoid

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def construire_grille(x_min, x_max, N):
    x  = np.linspace(x_min, x_max, N, endpoint=False)
    dx = x[1] - x[0]
    L  = x_max - x_min
    k  = np.fft.fftfreq(N, d=1.0 / N) * (2 * np.pi / L)
    return x, dx, k


def paquet_onde_gaussien(x, x0, sigma, k0):
    psi = np.exp(-((x - x0)**2) / (4 * sigma**2)) * np.exp(1j * k0 * x)
    norme = np.sqrt(_trapz(np.abs(psi)**2, x))
    return psi / norme


def potentiel_barriere(x, x_centre, largeur, hauteur):
    V = np.zeros_like(x)
    masque = np.abs(x - x_centre) < largeur / 2
    V[masque] = hauteur
    return V


def potentiel_harmonique(x, omega=1.0):
    return 0.5 * omega**2 * x**2


def potentiel_double_puits(x, a=2.0, b=0.5):
    return -a * x**2 + b * x**4


def pas_split_step(psi, V, k, dt):
    psi = np.exp(-1j * V * dt / 2) * psi
    psi_k = np.fft.fft(psi)
    psi_k = np.exp(-1j * k**2 * dt) * psi_k
    psi = np.fft.ifft(psi_k)
    psi = np.exp(-1j * V * dt / 2) * psi
    return psi


def resoudre(psi0, V, k, dt, N_steps, save_every=1):
    psi = psi0.copy()
    psi_histoire = [psi.copy()]
    t_histoire   = [0.0]

    for n in range(N_steps):
        psi = pas_split_step(psi, V, k, dt)
        if (n + 1) % save_every == 0:
            psi_histoire.append(psi.copy())
            t_histoire.append((n + 1) * dt)

    return np.array(psi_histoire), np.array(t_histoire)


def norme(psi, dx):
    return _trapz(np.abs(psi)**2, dx=dx).real


def energie(psi, V, k, dx):
    psi_k  = np.fft.fft(psi) / len(psi)
    E_cin  = np.sum(k**2 * np.abs(psi_k)**2) * dx * len(psi)
    E_pot  = _trapz(V * np.abs(psi)**2, dx=dx).real
    return (E_cin + E_pot).real


def pas_split_step_dst(psi, V, k_dst, dt):
    psi = np.exp(-1j * V * dt / 2) * psi
    psi_s = np.fft.rfft(np.concatenate([psi, psi[::-1]])).imag[:len(psi)]
    psi_s = np.exp(-1j * k_dst**2 * dt) * psi_s
    N = len(psi)
    psi_s_full = np.concatenate([[0], psi_s, [0], -psi_s[::-1]])
    psi = np.fft.irfft(1j * psi_s_full)[:N]
    psi = np.exp(-1j * V * dt / 2) * psi
    return psi


def validation_croisee(Nx=301, mu=0.5, sigma=0.05, amplitude=-1e4,
                        t=0.01, n_modes=70, dt=1e-5):
    from scipy.linalg import eigh_tridiagonal

    x  = np.linspace(0, 1, Nx)
    dx = 1.0 / (Nx - 1)

    Vx   = amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))
    psi0 = np.sqrt(2) * np.sin(np.pi * x)
    psi0 = psi0 / np.sqrt(_trapz(psi0**2, x))

    d_tri  = 1 / dx**2 + Vx[1:-1]
    e_tri  = -1 / (2 * dx**2) * np.ones(len(d_tri) - 1)
    w, v   = eigh_tridiagonal(d_tri, e_tri)
    E_js   = w[:n_modes]
    psi_js = np.pad(v.T[:n_modes], [(0, 0), (1, 1)], mode='constant')
    cs     = np.dot(psi_js, psi0)

    psi_modal = np.zeros(Nx, dtype=complex)
    for j in range(n_modes):
        psi_modal += cs[j] * psi_js[j] * np.exp(-1j * E_js[j] * t)
    rho_A = np.abs(psi_modal)**2

    N_int = Nx - 2
    n_dst = np.arange(1, N_int + 1)
    k_dst = (n_dst * np.pi) ** 2 / 2.0

    psi_int = psi0[1:-1].astype(complex).copy()
    Vx_int  = Vx[1:-1]
    N_steps = max(1, int(t / dt))

    exp_cin      = np.exp(-1j * k_dst * dt)
    exp_pot_half = np.exp(-1j * Vx_int * dt / 2)

    from scipy.fft import dst, idst

    for _ in range(N_steps):
        psi_int = exp_pot_half * psi_int
        psi_s   = dst(psi_int.real, type=2) + 1j * dst(psi_int.imag, type=2)
        psi_s   = exp_cin * psi_s
        psi_int = (idst(psi_s.real, type=2) + 1j * idst(psi_s.imag, type=2))
        psi_int = exp_pot_half * psi_int

    psi_ss = np.zeros(Nx, dtype=complex)
    psi_ss[1:-1] = psi_int
    rho_B = np.abs(psi_ss)**2

    mse     = float(np.mean((rho_A - rho_B)**2))
    err_rel = float(np.mean(np.abs(rho_A - rho_B)) / (np.mean(rho_A) + 1e-10)) * 100
    norme_A = float(_trapz(rho_A, x))
    norme_B = float(_trapz(rho_B, x))

    print(f"MSE          : {mse:.2e}")
    print(f"Erreur rel.  : {err_rel:.2f}%")
    print(f"Norme A      : {norme_A:.5f}")
    print(f"Norme B      : {norme_B:.5f}")

    Vn = Vx / (np.max(np.abs(Vx)) + 1e-10) * max(rho_A.max(), rho_B.max()) * 0.4

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Validation croisee — t = {t:.4f}", fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(x, rho_A, color='royalblue', linewidth=2)
    ax.plot(x, rho_B, color='tomato', linewidth=2, linestyle='--')
    ax.fill_between(x, Vn, where=Vn < 0, alpha=0.15, color='green')
    ax.set_xlabel("Position x")
    ax.set_ylabel(r"$|\psi(x,t)|^2$")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    err_pts = np.abs(rho_A - rho_B)
    ax.fill_between(x, err_pts, alpha=0.5, color='darkorange')
    ax.plot(x, err_pts, color='darkorange', linewidth=1.2)
    ax.set_xlabel("Position x")
    ax.set_ylabel("Erreur absolue")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "validation_croisee.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure sauvegardee : {out_path}")

    return x, rho_A, rho_B


def simuler_et_afficher(scenario="barriere"):
    N       = 1024
    dt      = 0.002
    T_max   = 4.0
    N_steps = int(T_max / dt)

    if scenario == "barriere":
        x, dx, k = construire_grille(-12, 12, N)
        V    = potentiel_barriere(x, x_centre=2.0, largeur=0.8, hauteur=8.0)
        psi0 = paquet_onde_gaussien(x, x0=-4.0, sigma=1.0, k0=3.0)
        titre = "Effet tunnel"

    elif scenario == "harmonique":
        x, dx, k = construire_grille(-8, 8, N)
        V    = potentiel_harmonique(x, omega=2.0)
        psi0 = paquet_onde_gaussien(x, x0=3.0, sigma=0.7, k0=0.0)
        titre = "Oscillateur harmonique quantique"

    elif scenario == "double_puits":
        x, dx, k = construire_grille(-4, 4, N)
        V    = potentiel_double_puits(x, a=4.0, b=1.0)
        psi0 = paquet_onde_gaussien(x, x0=-1.4, sigma=0.4, k0=0.0)
        titre = "Double puits "

    else:
        raise ValueError(f"Scenario inconnu : {scenario}")

    save_every = 5
    psi_hist, t_hist = resoudre(psi0, V, k, dt, N_steps, save_every=save_every)

    normes   = [norme(p, dx) for p in psi_hist]
    energies = [energie(p, V, k, dx) for p in psi_hist]
    print(f"\n{'='*50}")
    print(f"Scenario : {titre}")
    print(f"  Variation de norme   : {max(normes) - min(normes):.2e}")
    print(f"  Variation d'energie  : {max(energies) - min(energies):.2e}")
    print(f"{'='*50}\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(titre, fontsize=14, fontweight='bold')
    indices = [0, len(t_hist)//2, -1]
    labels  = ["t = 0 (initial)", f"t = {t_hist[len(t_hist)//2]:.2f}", f"t = {t_hist[-1]:.2f} (final)"]
    V_affiche = V / (np.max(np.abs(V)) + 1e-10) * 0.5
    for ax, idx, label in zip(axes, indices, labels):
        rho = np.abs(psi_hist[idx])**2
        ax.fill_between(x, rho, alpha=0.5, color='royalblue')
        ax.plot(x, rho, color='royalblue', linewidth=1.5)
        ax.plot(x, V_affiche, color='tomato', linewidth=2, linestyle='--')
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Position x")
        ax.set_ylabel(r"$|\psi(x,t)|^2$")
        ax.set_ylim(-0.1, max(np.max(np.abs(psi_hist[0])**2), 0.5) * 1.3)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"split_step_{scenario}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure sauvegardee pour le scenario '{scenario}'.")
    return psi_hist, t_hist, x, V


if __name__ == "__main__":
    for sc in ["barriere", "harmonique", "double_puits"]:
        simuler_et_afficher(sc)

    print("\n--- Validation croisee ---")
    validation_croisee(Nx=301, mu=0.5, sigma=0.05, amplitude=-1e4, t=0.01, n_modes=70, dt=1e-5)