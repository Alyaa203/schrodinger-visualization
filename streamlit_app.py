import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import streamlit as st

from scipy.ndimage import gaussian_filter

from simulation import (
    solve_stationary_2d,
    get_mode_2d,
    solve_time_basis,
    density_t,
    density_surface,
)



st.set_page_config(page_title="Visualisation de Schrödinger", layout="wide")

st.title("Visualisation de l'équation de Schrödinger")
page = st.sidebar.radio(
    "Choix de la partie",
    ["Régime stationnaire 2D", "Régime dépendant du temps 1D"]
)



def artistic_render(psi, glow_sigma=3, glow_intensity=0.7):
    rho = np.abs(psi) ** 2
    phi = np.angle(psi)

    rho_max = rho.max()
    if rho_max > 0:
        rho = rho / rho_max

    h = (phi + np.pi) / (2 * np.pi)   # teinte
    s = np.ones_like(h)               # saturation
    v = rho                           # luminosité

    hsv = np.stack((h, s, v), axis=-1)
    rgb = colors.hsv_to_rgb(hsv)

    rho_glow = gaussian_filter(rho, sigma=glow_sigma)

    rgb_glow = rgb.copy()
    rgb_glow[..., 2] = np.clip(rgb_glow[..., 2] + glow_intensity * rho_glow, 0, 1)

    return rho, phi, rgb, rgb_glow


if page == "Régime stationnaire 2D":
    st.header("Régime stationnaire 2D")

    st.sidebar.subheader("Paramètres physiques")
    N = st.sidebar.slider("N", 80, 250, 150)
    k = st.sidebar.slider("Nombre de modes calculés", 3, 20, 10)
    mode = st.sidebar.slider("Mode à afficher", 0, k - 1, 0)

    x0 = st.sidebar.slider("x0", 0.0, 1.0, 0.3)
    y0 = st.sidebar.slider("y0", 0.0, 1.0, 0.3)
    sigma = st.sidebar.slider("sigma", 0.02, 0.30, 0.10)
    amplitude = st.sidebar.slider("Amplitude du potentiel", 0.1, 5.0, 1.0)

    st.sidebar.subheader("Paramètres artistiques")
    glow_sigma = st.sidebar.slider("Glow sigma", 1, 10, 3)
    glow_intensity = st.sidebar.slider("Intensité glow", 0.0, 2.0, 0.7)

    X, Y, V, eigenvalues, eigenvectors = solve_stationary_2d(
        N=N,
        k=k,
        x0=x0,
        y0=y0,
        sigma=sigma,
        amplitude=amplitude
    )

    psi = get_mode_2d(eigenvectors, N, mode)

    rho, phi, rgb, rgb_glow = artistic_render(
        psi,
        glow_sigma=glow_sigma,
        glow_intensity=glow_intensity
    )

    st.subheader("Visualisation scientifique")
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        im1 = ax1.imshow(V, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
        ax1.set_title("Potentiel V(x,y)")
        plt.colorbar(im1, ax=ax1)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        im2 = ax2.imshow(rho, origin="lower", extent=(0, 1, 0, 1), cmap="magma")
        ax2.contour(X, Y, rho, levels=15, linewidths=0.6, colors="white")
        ax2.set_title(f"Densité du mode {mode}")
        plt.colorbar(im2, ax=ax2)
        st.pyplot(fig2)

    st.subheader("Visualisation artistique")
    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.imshow(rgb, origin="lower", extent=(0, 1, 0, 1))
        ax3.set_title("Phase + densité")
        ax3.set_xticks([])
        ax3.set_yticks([])
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        ax4.imshow(rgb_glow, origin="lower", extent=(0, 1, 0, 1))
        ax4.set_title("Phase + densité + glow")
        ax4.set_xticks([])
        ax4.set_yticks([])
        st.pyplot(fig4)

    st.subheader("Valeurs propres calculées")
    st.write(eigenvalues)

elif page == "Régime dépendant du temps 1D":
    st.header("Régime dépendant du temps 1D")

    st.sidebar.subheader("Paramètres physiques")
    Nx = st.sidebar.slider("Nx", 101, 501, 301, step=50)
    n_modes = st.sidebar.slider("Nombre de modes utilisés", 10, 100, 70, step=10)

    mu = st.sidebar.slider("Centre du potentiel", 0.1, 0.9, 0.5)
    sigma = st.sidebar.slider("Largeur du potentiel", 0.01, 0.15, 0.05)
    amplitude = st.sidebar.slider("Amplitude du puits", -20000.0, -100.0, -10000.0, step=100.0)

    t = st.sidebar.slider("Temps t", 0.0, 0.05, 0.01)

    x, psi0, Vx, E_js, psi_js, cs = solve_time_basis(
        Nx=Nx,
        mu=mu,
        sigma=sigma,
        amplitude=amplitude,
        n_modes=n_modes
    )

    rho_t = density_t(x, E_js, psi_js, cs, t)

    col1, col2 = st.columns(2)

    with col1:
        fig5, ax5 = plt.subplots(figsize=(7, 4))
        ax5.plot(x, psi0**2, label="|psi0|^2")
        ax5.plot(x, rho_t, label=f"|psi(x,t)|^2 à t={t:.4f}")
        ax5.set_xlabel("x")
        ax5.set_ylabel("Densité")
        ax5.set_title("Évolution temporelle de la densité")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        st.pyplot(fig5)

    with col2:
        fig6, ax6 = plt.subplots(figsize=(7, 4))
        ax6.plot(x, Vx)
        ax6.set_title("Potentiel V(x)")
        ax6.set_xlabel("x")
        ax6.grid(True, alpha=0.3)
        st.pyplot(fig6)

    st.subheader("Surface 3D de la densité")

    Nt = st.slider("Nombre de temps pour la surface 3D", 30, 150, 80)
    t_max = st.slider("Temps maximal pour la surface 3D", 0.005, 0.05, 0.02)

    t_vals = np.linspace(0, t_max, Nt)
    rho_surface = density_surface(x, E_js, psi_js, cs, t_vals)

    T, Xgrid = np.meshgrid(t_vals, x)

    fig7 = plt.figure(figsize=(10, 6))
    ax7 = fig7.add_subplot(111, projection="3d")
    ax7.plot_surface(Xgrid, T, rho_surface.T, cmap="viridis")
    ax7.set_xlabel("x")
    ax7.set_ylabel("t")
    ax7.set_zlabel(r"$|\psi(x,t)|^2$")
    ax7.set_title("Surface 3D de la densité de probabilité")
    st.pyplot(fig7)