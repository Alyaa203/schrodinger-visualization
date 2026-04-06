import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from simulation import (
    solve_stationary_2d,
    get_mode_2d,
    solve_time_basis,
    density_t,
    density_surface,
)
from visualisation import (
    rendu_artistique, generer_rosace, generer_nebuleuse, generer_cristal,
    generer_mandala, generer_galaxie, 
)
from Fourier import (
    construire_grille, paquet_onde_gaussien, resoudre,
    potentiel_barriere, potentiel_harmonique, potentiel_double_puits,
    norme, energie, validation_croisee,
)

st.set_page_config(page_title="Visualisation de Schrödinger", layout="wide")
st.title("Visualisation de l'équation de Schrödinger")

page = st.sidebar.radio(
    "Choix de la partie",
    [
        "Régime stationnaire 2D",
        "Régime dépendant du temps 1D",
        "Split-Step Fourier",
        "Art quantique",
        "Validation croisée",
    ]
)

NX = 301
NT = 80
T_MAX = 0.02

if "donnees_art" not in st.session_state:
    st.session_state["donnees_art"] = {}

# =============================================================================
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
        N=N, k=k, x0=x0, y0=y0, sigma=sigma, amplitude=amplitude,
    )
    psi = get_mode_2d(vecteurs_propres, N, mode)
    rho, phi, rvb = rendu_artistique(psi)

    st.session_state["donnees_art"]["stationnaire_2d"] = {
        "X": X, "Y": Y, "V": V,
        "valeurs_propres": valeurs_propres,
        "vecteurs_propres": vecteurs_propres,
        "psi": psi, "rho": rho, "phi": phi,
        "N": N, "k": k,
        "x0": x0, "y0": y0, "sigma": sigma, "amplitude": amplitude,
    }

    st.subheader("Visualisation scientifique")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(7, 7))
        im1 = ax1.imshow(V, origin="lower", extent=(0, 1, 0, 1), cmap="viridis", aspect="equal")
        ax1.set_title("Potentiel V(x,y)")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        im2 = ax2.imshow(rho, origin="lower", extent=(0, 1, 0, 1), cmap="magma", aspect="equal")
        ax2.contour(X, Y, rho, levels=15, linewidths=0.6, colors="white")
        ax2.set_title(f"Densité du mode {mode}")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    st.subheader("Valeurs propres calculées")
    st.write(valeurs_propres)

# =============================================================================
elif page == "Régime dépendant du temps 1D":
    st.header("Régime dépendant du temps 1D")

    st.sidebar.subheader("Paramètres physiques")
    n_modes   = st.sidebar.slider("Nombre de modes utilisés", 10, 100, 70, step=10)
    mu        = st.sidebar.slider("Centre du potentiel", 0.1, 0.9, 0.5)
    sigma     = st.sidebar.slider("Largeur du potentiel", 0.01, 0.15, 0.05)
    amplitude = st.sidebar.slider("Amplitude du puits", -20000.0, -100.0, -10000.0, step=100.0)
    t         = st.sidebar.slider("Temps", 0.0, 0.05, 0.01)

    x, psi0, Vx, E_js, psi_js, cs = solve_time_basis(
        Nx=NX, mu=mu, sigma=sigma, amplitude=amplitude, n_modes=n_modes,
    )
    rho_t = density_t(x, E_js, psi_js, cs, t)

    st.session_state["donnees_art"]["temporel_1d"] = {
        "x": x, "psi0": psi0, "Vx": Vx,
        "E_js": E_js, "psi_js": psi_js, "cs": cs,
        "t": t, "rho_t": rho_t,
    }

    col1, col2 = st.columns(2)
    with col1:
        fig5, ax5 = plt.subplots(figsize=(7, 7))
        ax5.plot(x, psi0**2, label=r"$|\psi_0|^2$")
        ax5.plot(x, rho_t, label=rf"$|\psi(x,t)|^2$ à $t={t:.4f}$")
        ax5.set_xlabel("Position")
        ax5.set_ylabel("Densité de probabilité")
        ax5.set_title("Évolution temporelle de la densité")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        fig5.tight_layout()
        st.pyplot(fig5, use_container_width=True)
    with col2:
        fig6, ax6 = plt.subplots(figsize=(7, 7))
        ax6.plot(x, Vx)
        ax6.set_title("Potentiel V(x)")
        ax6.set_xlabel("Position")
        ax6.set_ylabel("Énergie potentielle")
        ax6.grid(True, alpha=0.3)
        fig6.tight_layout()
        st.pyplot(fig6, use_container_width=True)

    st.subheader("Surface 3D de la densité")
    vals_temps = np.linspace(0, T_MAX, NT)
    rho_surface = density_surface(x, E_js, psi_js, cs, vals_temps)
    Tgrid, Xgrid = np.meshgrid(vals_temps, x)
    col3, col4 = st.columns(2)
    with col3:
        fig7 = plt.figure(figsize=(7, 7))
        ax7 = fig7.add_subplot(111, projection="3d")
        ax7.plot_surface(Xgrid, Tgrid, rho_surface.T, cmap="viridis")
        ax7.set_xlabel("Position")
        ax7.set_ylabel("Temps")
        ax7.set_zlabel(r"$|\psi(x,t)|^2$")
        ax7.set_title("Surface 3D de la densité de probabilité")
        fig7.tight_layout()
        st.pyplot(fig7, use_container_width=True)

# =============================================================================
elif page == "Split-Step Fourier":
    st.header("Split-Step Fourier — Dynamique quantique")

    st.markdown("""
    Résolution de l'équation de Schrödinger par la méthode **Split-Step Fourier** (schéma de Strang).
    Chaque pas de temps est décomposé en 3 étapes :
    1. Demi-pas potentiel dans l'espace réel
    2. Pas cinétique complet via FFT dans l'espace de Fourier
    3. Demi-pas potentiel dans l'espace réel
    """)

    st.sidebar.subheader("Scénario")
    scenario = st.sidebar.radio(
        "Choisir un scénario",
        ["Barrière tunnel", "Oscillateur harmonique", "Double puits"],
    )

    if scenario == "Barrière tunnel":
        st.sidebar.subheader("Paramètres")
        x_centre = st.sidebar.slider("Position de la barrière", -4.0, 4.0, 2.0)
        largeur  = st.sidebar.slider("Largeur de la barrière", 0.2, 2.0, 0.8)
        hauteur  = st.sidebar.slider("Hauteur de la barrière", 1.0, 20.0, 8.0)
        k0       = st.sidebar.slider("Impulsion initiale k0", 1.0, 6.0, 3.0)
        x0_psi   = st.sidebar.slider("Position initiale du paquet", -10.0, -1.0, -4.0)
        x, dx, k_fft = construire_grille(-12, 12, 1024)
        V = potentiel_barriere(x, x_centre=x_centre, largeur=largeur, hauteur=hauteur)
        psi0 = paquet_onde_gaussien(x, x0=x0_psi, sigma=1.0, k0=k0)
        titre_scenario = "Effet tunnel — Barrière de potentiel"
        dt, N_steps, save_every = 0.002, 2000, 20

    elif scenario == "Oscillateur harmonique":
        st.sidebar.subheader("Paramètres")
        omega  = st.sidebar.slider("Fréquence omega", 0.5, 4.0, 2.0)
        x0_psi = st.sidebar.slider("Position initiale du paquet", -5.0, 5.0, 3.0)
        x, dx, k_fft = construire_grille(-8, 8, 1024)
        V = potentiel_harmonique(x, omega=omega)
        psi0 = paquet_onde_gaussien(x, x0=x0_psi, sigma=0.7, k0=0.0)
        titre_scenario = "Oscillateur harmonique quantique"
        dt, N_steps, save_every = 0.002, 2000, 20

    else:  # Double puits
        st.sidebar.subheader("Paramètres")
        a      = st.sidebar.slider("Paramètre a", 1.0, 6.0, 4.0)
        b      = st.sidebar.slider("Paramètre b", 0.2, 2.0, 1.0)
        x0_psi = st.sidebar.slider("Position initiale du paquet", -3.0, -0.5, -1.4)
        x, dx, k_fft = construire_grille(-4, 4, 1024)
        V = potentiel_double_puits(x, a=a, b=b)
        psi0 = paquet_onde_gaussien(x, x0=x0_psi, sigma=0.4, k0=0.0)
        titre_scenario = "Double puits — Tunneling quantique"
        dt, N_steps, save_every = 0.002, 2000, 20

    if st.button("Lancer la simulation", type="primary"):
        with st.spinner("Simulation en cours…"):
            psi_hist, t_hist = resoudre(psi0, V, k_fft, dt, N_steps, save_every)

        normes   = [norme(p, dx) for p in psi_hist]
        energies = [energie(p, V, k_fft, dx) for p in psi_hist]

       

        st.subheader("Densité à 3 instants")
        V_aff = V / (np.max(np.abs(V)) + 1e-10) * 0.5
        fig_snap, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig_snap.suptitle(titre_scenario, fontsize=13, fontweight="bold")
        indices = [0, len(t_hist) // 2, -1]
        labels  = ["t = 0 (initial)", f"t = {t_hist[len(t_hist)//2]:.2f}", f"t = {t_hist[-1]:.2f} (final)"]
        for ax, idx, label in zip(axes, indices, labels):
            rho_snap = np.abs(psi_hist[idx])**2
            ax.fill_between(x, rho_snap, alpha=0.5, color="royalblue")
            ax.plot(x, rho_snap, color="royalblue", linewidth=1.5, label=r"$|\psi|^2$")
            ax.plot(x, V_aff, color="tomato", linewidth=2, linestyle="--", label="V(x)")
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("Position x")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig_snap.tight_layout()
        st.pyplot(fig_snap, use_container_width=True)


# =============================================================================
elif page == "Art quantique":
    st.header("Art quantique")
    st.write("Cette page utilise les valeurs propres calculées pour générer plusieurs rendus artistiques.")

    donnees_art = st.session_state["donnees_art"]
    if not donnees_art:
        st.warning("Aucune donnée disponible. Exécute d'abord une simulation 2D ou 1D.")
        st.stop()

    labels_sources = {
        "stationnaire_2d": "Régime stationnaire 2D",
        "temporel_1d": "Régime temporel 1D",
    }
    source = st.selectbox(
        "Source des données",
        [k for k in donnees_art.keys()],
        format_func=lambda k: labels_sources.get(k, k),
    )
    style_art = st.radio(
        "Style artistique",
        ["Rosace", "Nébuleuse quantique", "Cristal quantique", "Mandala fractal", "Galaxie spirale"],
        horizontal=True,
    )

    TAILLE_ART = 700
    if source == "stationnaire_2d":
        valeurs_propres = donnees_art[source]["valeurs_propres"]
    else:
        valeurs_propres = donnees_art[source]["E_js"]

    styles = {
        "Rosace":              (generer_rosace,         "Rosace issue des valeurs propres"),
        "Nébuleuse quantique": (generer_nebuleuse,      "Nébuleuse quantique issue des valeurs propres"),
        "Cristal quantique":   (generer_cristal,        "Cristal quantique issu des valeurs propres"),
        "Mandala fractal":     (generer_mandala,        "Mandala fractal issu des valeurs propres"),
        "Galaxie spirale":     (generer_galaxie,        "Galaxie spirale issue des valeurs propres"),
    }
    fn, titre = styles[style_art]
    image_art = fn(valeurs_propres, taille=TAILLE_ART)

    col1, col2 = st.columns(2)
    with col1:
        figA, axA = plt.subplots(figsize=(7, 7))
        axA.imshow(image_art, origin="lower", aspect="equal")
        axA.set_title(titre)
        axA.set_xticks([])
        axA.set_yticks([])
        figA.tight_layout()
        st.pyplot(figA, use_container_width=True)

    with col2:
        st.markdown("**Spectre des valeurs propres**")
        st.markdown(
            "Chaque barre est un niveau d'énergie $E_j$. "
            "L'espacement entre les barres influence directement les fréquences "
            "des patterns générés."
        )
        fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
        vp = valeurs_propres[:20] if len(valeurs_propres) > 20 else valeurs_propres
        colors_hist = plt.cm.plasma(np.linspace(0.1, 0.9, len(vp)))
        bars = ax_hist.bar(range(len(vp)), vp, color=colors_hist, edgecolor="white", linewidth=0.5)
        ax_hist.set_xlabel("Indice $j$", fontsize=11)
        ax_hist.set_ylabel("Énergie $E_j$", fontsize=11)
        ax_hist.set_title("Spectre des valeurs propres", fontsize=12)
        ax_hist.grid(True, alpha=0.3, axis="y")
        for i, (bar, val) in enumerate(zip(bars, vp)):
            ax_hist.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)
        fig_hist.tight_layout()
        st.pyplot(fig_hist, use_container_width=True)

    st.subheader("Valeurs propres utilisées")
    st.write(valeurs_propres)

# =============================================================================
elif page == "Validation croisée":
    st.header("Validation croisée — Décomposition modale vs Split-Step Fourier")

    st.markdown("""
    Compare $|\\psi(x,t)|^2$ calculée par les deux méthodes sur la même grille
    périodique $[0, 1[$ avec le même potentiel gaussien et le même état initial.
    Si les deux courbes coïncident, les deux implémentations sont correctes.
    """)

    st.sidebar.subheader("Paramètres")
    mu_v    = st.sidebar.slider("Centre du potentiel", 0.1, 0.9, 0.5)
    sigma_v = st.sidebar.slider("Largeur du potentiel", 0.01, 0.15, 0.05)
    amp_v   = st.sidebar.slider("Amplitude du puits", -20000.0, -100.0, -10000.0, step=100.0)
    t_v     = st.sidebar.slider("Instant t", 0.001, 0.05, 0.01)
    modes_v = st.sidebar.slider("Modes propres (méthode A)", 10, 200, 70, step=10)
    dt_v    = st.sidebar.slider("Pas de temps dt (méthode B)", 0.000001, 0.0001, 0.00001, format="%.6f")

    if st.button("Lancer la comparaison", type="primary"):
        with st.spinner("Calcul en cours…"):
            x, rho_A, rho_B = validation_croisee(
                Nx=NX, mu=mu_v, sigma=sigma_v, amplitude=amp_v,
                t=t_v, n_modes=modes_v, dt=dt_v,
            )

        _trapz = np.trapz if hasattr(np, "trapz") else np.trapezoid
        mse     = float(np.mean((rho_A - rho_B)**2))
        err_rel = float(np.mean(np.abs(rho_A - rho_B)) / (np.mean(rho_A) + 1e-10)) * 100
        norme_A = float(_trapz(rho_A, x))
        norme_B = float(_trapz(rho_B, x))

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Erreur MSE",        f"{mse:.2e}")
        col_m2.metric("Erreur relative",   f"{err_rel:.2f}%")
        col_m3.metric("Norme (méthode A)", f"{norme_A:.5f}")
        col_m4.metric("Norme (méthode B)", f"{norme_B:.5f}")

        Vx = amp_v * np.exp(-(x - mu_v)**2 / (2 * sigma_v**2))
        Vn = Vx / (np.max(np.abs(Vx)) + 1e-10) * max(rho_A.max(), rho_B.max()) * 0.4

        st.subheader("Comparaison des densités")
        fig_cmp, ax_cmp = plt.subplots(figsize=(10, 4))
        ax_cmp.plot(x, rho_A, color="royalblue", linewidth=2,
                    label="Méthode A — Décomposition modale")
        ax_cmp.plot(x, rho_B, color="tomato", linewidth=2, linestyle="--",
                    label="Méthode B — Split-Step Fourier")
        ax_cmp.fill_between(x, Vn, where=Vn < 0, alpha=0.15, color="green", label="V(x) [normalisé]")
        ax_cmp.set_xlabel("Position x", fontsize=12)
        ax_cmp.set_ylabel(r"$|\psi(x,t)|^2$", fontsize=12)
        ax_cmp.set_title(f"Densité de probabilité — t = {t_v:.4f}", fontsize=13)
        ax_cmp.legend(fontsize=11)
        ax_cmp.grid(True, alpha=0.3)
        fig_cmp.tight_layout()
        st.pyplot(fig_cmp, use_container_width=True)

       

