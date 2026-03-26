import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from simulation import (
    solve_stationary_2d,
    get_mode_2d,
    solve_time_basis,
    density_t,
    density_surface,
    psi_t,
)
from visualisation import rendu_artistique, generer_rosace

st.set_page_config(page_title="Visualisation de Schrödinger", layout="wide")

st.title("Visualisation de l'équation de Schrödinger")
page = st.sidebar.radio(
    "Choix de la partie",
    ["Régime stationnaire 2D", "Régime dépendant du temps 1D", "Art quantique"]
)

NX = 301       # Nombre de points de grille 1D : impair pour avoir un point exactement au centre
NT = 80        # Nombre d'instants pour la surface 3D 
T_MAX = 0.02    #Si T_MAX est trop grand, les oscillations deviennent trop rapides à résoudre
TAILLE_ART = 700  # Résolution en pixels de la rosace, just for art 


# Initialisation du stockage session
if "donnees_art" not in st.session_state:
    st.session_state["donnees_art"] = {}


if page == "Régime stationnaire 2D":
    st.header("Régime stationnaire 2D")

    st.sidebar.subheader("Paramètres physiques")
    N = st.sidebar.slider("Résolution de la grille", 80, 250, 150)
    k = st.sidebar.slider("Nombre de modes calculés", 3, 20, 10)
    mode = st.sidebar.slider("Mode à afficher", 0, k - 1, 0)

    x0 = st.sidebar.slider("Position x du potentiel", 0.0, 1.0, 0.3)
    y0 = st.sidebar.slider("Position y du potentiel", 0.0, 1.0, 0.3)
    sigma = st.sidebar.slider("Largeur du potentiel", 0.02, 0.30, 0.10)
    amplitude = st.sidebar.slider("Amplitude du potentiel", 0.1, 5.0, 1.0)

    X, Y, V, valeurs_propres, vecteurs_propres = solve_stationary_2d(
        N=N,
        k=k,
        x0=x0,
        y0=y0,
        sigma=sigma,
        amplitude=amplitude
    )

    psi = get_mode_2d(vecteurs_propres, N, mode)
    rho, phi, rvb = rendu_artistique(psi)

    st.session_state["donnees_art"]["stationnaire_2d"] = {
        "X": X,
        "Y": Y,
        "V": V,
        "valeurs_propres": valeurs_propres,
        "vecteurs_propres": vecteurs_propres,
        "psi": psi,
        "rho": rho,
        "phi": phi,
        "N": N,
        "k": k
    }

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
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.imshow(rvb, origin="lower", extent=(0, 1, 0, 1))
    ax3.set_title("Phase + densité")
    ax3.set_xticks([])
    ax3.set_yticks([])
    st.pyplot(fig3)

    st.subheader("Valeurs propres calculées")
    st.write(valeurs_propres)


elif page == "Régime dépendant du temps 1D":
    st.header("Régime dépendant du temps 1D")

    st.sidebar.subheader("Paramètres physiques")
    n_modes = st.sidebar.slider("Nombre de modes utilisés", 10, 100, 70, step=10)

    mu = st.sidebar.slider("Centre du potentiel", 0.1, 0.9, 0.5)
    sigma = st.sidebar.slider("Largeur du potentiel", 0.01, 0.15, 0.05)
    amplitude = st.sidebar.slider("Amplitude du puits", -20000.0, -100.0, -10000.0, step=100.0)

    t = st.sidebar.slider("Temps", 0.0, 0.05, 0.01)

    x, psi0, Vx, E_js, psi_js, cs = solve_time_basis(
        Nx=NX,
        mu=mu,
        sigma=sigma,
        amplitude=amplitude,
        n_modes=n_modes
    )

    rho_t = density_t(x, E_js, psi_js, cs, t)

    st.session_state["donnees_art"]["temporel_1d"] = {
        "x": x,
        "psi0": psi0,
        "Vx": Vx,
        "E_js": E_js,
        "psi_js": psi_js,
        "cs": cs,
        "t": t,
        "rho_t": rho_t
    }

    col1, col2 = st.columns(2)

    with col1:
        fig5, ax5 = plt.subplots(figsize=(7, 4))
        ax5.plot(x, psi0**2, label=r"$|\psi_0|^2$")
        ax5.plot(x, rho_t, label=rf"$|\psi(x,t)|^2$ à $t={t:.4f}$")
        ax5.set_xlabel("Position")
        ax5.set_ylabel("Densité de probabilité")
        ax5.set_title("Évolution temporelle de la densité")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        st.pyplot(fig5)

    with col2:
        fig6, ax6 = plt.subplots(figsize=(7, 4))
        ax6.plot(x, Vx)
        ax6.set_title("Potentiel V(x)")
        ax6.set_xlabel("Position")
        ax6.set_ylabel("Énergie potentielle")
        ax6.grid(True, alpha=0.3)
        st.pyplot(fig6)

    st.subheader("Surface 3D de la densité")

    vals_temps = np.linspace(0, T_MAX, NT)
    rho_surface = density_surface(x, E_js, psi_js, cs, vals_temps)

    Tgrid, Xgrid = np.meshgrid(vals_temps, x)

    fig7 = plt.figure(figsize=(10, 6))
    ax7 = fig7.add_subplot(111, projection="3d")
    ax7.plot_surface(Xgrid, Tgrid, rho_surface.T, cmap="viridis")
    ax7.set_xlabel("Position")
    ax7.set_ylabel("Temps")
    ax7.set_zlabel(r"$|\psi(x,t)|^2$")
    ax7.set_title("Surface 3D de la densité de probabilité")
    st.pyplot(fig7)


elif page == "Art quantique":
    st.header("Art quantique")

    st.write(
        "Cette page utilise les valeurs propres calculées pour générer une rosace artistique."
    )

    donnees_art = st.session_state["donnees_art"]

    if not donnees_art:
        st.warning("Aucune donnée disponible. Exécute d'abord une simulation 2D ou 1D.")
        st.stop()

    labels_sources = {
        "stationnaire_2d": "Régime stationnaire 2D",
        "temporel_1d": "Régime temporel 1D"
    }

    source = st.selectbox(
        "Source des données",
        [k for k in donnees_art.keys()],
        format_func=lambda k: labels_sources.get(k, k)
    )

    if source == "stationnaire_2d":
        valeurs_propres = donnees_art[source]["valeurs_propres"]
    elif source == "temporel_1d":
        valeurs_propres = donnees_art[source]["E_js"]

    rosace = generer_rosace(valeurs_propres, taille=TAILLE_ART)

    figA, axA = plt.subplots(figsize=(8, 8))
    axA.imshow(rosace, origin="lower")
    axA.set_title("Rosace issue des valeurs propres")
    axA.set_xticks([])
    axA.set_yticks([])
    st.pyplot(figA)

    st.subheader("Valeurs propres utilisées")
    st.write(valeurs_propres)
