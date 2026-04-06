import numpy as np
import matplotlib.colors as colors

from scipy.ndimage import gaussian_filter


def rendu_artistique(psi):
    rho = np.abs(psi) ** 2
    phi = np.angle(psi)

    rho_max = rho.max()
    if rho_max > 0:
        rho = rho / rho_max

    h = (phi + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    v = rho

    hsv = np.stack((h, s, v), axis=-1)
    rvb = colors.hsv_to_rgb(hsv)

    return rho, phi, rvb


def normaliser_champ(z):
    z = z - np.min(z)
    m = np.max(z)
    if m > 0:
        z = z / m
    return z


def _normaliser_valeurs_propres(valeurs_propres):
    vp = np.array(valeurs_propres, dtype=float)
    vp = vp - vp.min()
    if vp.max() > 0:
        vp = vp / vp.max()
    return vp


def generer_rosace(valeurs_propres, taille=700):
    y, x = np.mgrid[-1:1:taille*1j, -1:1:taille*1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    img = np.zeros_like(r)
    vp = _normaliser_valeurs_propres(valeurs_propres)

    for i, e in enumerate(vp):
        freq = 3 + i
        img += np.sin((8 + 25 * e) * np.pi * r + freq * theta) ** 2
        img += 0.5 * np.cos((5 + 20 * e) * theta - 10 * r)

    img = normaliser_champ(img)

    rvb = np.zeros((taille, taille, 3))
    rvb[..., 0] = img
    rvb[..., 1] = gaussian_filter(1 - img, sigma=2)
    rvb[..., 2] = np.sqrt(img)

    return np.clip(rvb, 0, 1)


def generer_cristal(valeurs_propres, taille=700):
    y, x = np.mgrid[-1.5:1.5:taille*1j, -1.5:1.5:taille*1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    vp = _normaliser_valeurs_propres(valeurs_propres)
    img = np.zeros_like(r)
    mask = r < 1.4

    for i, e in enumerate(vp):
        n = 3 + (i % 8)
        freq = 4 + 12 * e
        img += np.cos(n * theta + freq * r) * np.exp(-r * (0.8 + e))
        img += 0.4 * np.sin((i + 2) * np.pi * r**2 + (i % 5) * theta)

    img = img * mask
    img = gaussian_filter(img, sigma=taille / 400)
    img = normaliser_champ(img)

    # Crystal color palette: deep blue → cyan → gold
    rvb = np.zeros((taille, taille, 3))
    rvb[..., 0] = np.clip(0.15 * img + 0.7 * img**3, 0, 1)
    rvb[..., 1] = np.clip(0.6 * img**1.5 + 0.3 * gaussian_filter(img, sigma=taille / 200), 0, 1)
    rvb[..., 2] = np.clip(0.85 * np.sqrt(img) + 0.2 * (1 - img)**2, 0, 1)

    # Add sparkle highlights
    sparkle = np.clip((img - 0.82) * 6, 0, 1)
    rvb[..., 0] += sparkle * 0.9
    rvb[..., 1] += sparkle * 0.95
    rvb[..., 2] += sparkle

    return np.clip(rvb, 0, 1)


def generer_mandala(valeurs_propres, taille=700):
    """Géométrique / fractal — symétrie d'ordre N, motifs entrelacés."""
    y, x = np.mgrid[-1:1:taille*1j, -1:1:taille*1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    vp = _normaliser_valeurs_propres(valeurs_propres)
    img = np.zeros_like(r)

    for i, e in enumerate(vp):
        n = 4 + (i % 6) * 2          # symétries paires : 4, 6, 8, 10, 12, 14
        k = 6 + int(20 * e)
        phase = i * np.pi / (len(vp) + 1)
        petale = np.cos(n * theta + phase) ** 2
        anneau = np.exp(-((r - 0.12 * (i % 6 + 1)) ** 2) / 0.004)
        radial = np.sin(k * r + phase) ** 2
        img += petale * (anneau + 0.3 * radial)

    img = gaussian_filter(img, sigma=taille / 500)
    img = normaliser_champ(img)

    # Palette : noir profond → or → blanc ivoire
    rvb = np.zeros((taille, taille, 3))
    rvb[..., 0] = np.clip(img ** 0.6, 0, 1)
    rvb[..., 1] = np.clip(0.75 * img ** 0.8, 0, 1)
    rvb[..., 2] = np.clip(0.15 * img ** 1.5, 0, 1)

    # Reflets dorés sur les crêtes
    crete = np.clip((img - 0.75) * 4, 0, 1)
    rvb[..., 0] += crete * 0.4
    rvb[..., 1] += crete * 0.35
    rvb[..., 2] += crete * 0.05

    return np.clip(rvb, 0, 1)


def generer_galaxie(valeurs_propres, taille=700):
    """Cosmique / galactique — bras spiraux avec poussière d'étoiles."""
    y, x = np.mgrid[-1.3:1.3:taille*1j, -1.3:1.3:taille*1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    vp = _normaliser_valeurs_propres(valeurs_propres)

    # Bras spiraux : chaque valeur propre tord un bras
    spirale = np.zeros_like(r)
    for i, e in enumerate(vp):
        n_bras = 2 + (i % 3)
        pitch = 1.8 + 2.5 * e
        bras = np.cos(n_bras * theta - pitch * np.log1p(r * 4) + i * 0.4) ** 2
        enveloppe = np.exp(-r ** 2 / (0.18 + 0.25 * e))
        spirale += bras * enveloppe

    spirale = gaussian_filter(spirale, sigma=taille / 300)
    spirale = normaliser_champ(spirale)

    # Halo central lumineux
    noyau = np.exp(-8 * r ** 2)
    # Poussière interstellaire
    poussiere = normaliser_champ(
        np.sin(18 * theta + 22 * r) * np.exp(-1.5 * r)
        + np.cos(11 * theta - 15 * r) * np.exp(-2 * r)
    )

    rvb = np.zeros((taille, taille, 3))
    rvb[..., 0] = np.clip(0.85 * spirale + 0.6 * noyau, 0, 1)
    rvb[..., 1] = np.clip(0.5 * spirale ** 1.2 + 0.3 * poussiere + 0.4 * noyau, 0, 1)
    rvb[..., 2] = np.clip(0.3 + 0.6 * gaussian_filter(spirale, sigma=taille / 180) + 0.5 * noyau, 0, 1)

    # Étoiles ponctuelles
    rng = np.random.default_rng(seed=42)
    etoiles = rng.random((taille, taille))
    masque_etoile = etoiles > 0.997
    for c in range(3):
        rvb[..., c] = np.where(masque_etoile, 1.0, rvb[..., c])

    return np.clip(rvb, 0, 1)




def generer_nebuleuse(valeurs_propres, taille=700):
    y, x = np.mgrid[-1.2:1.2:taille*1j, -1.2:1.2:taille*1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    img = np.zeros_like(r)
    vp = _normaliser_valeurs_propres(valeurs_propres)

    for i, e in enumerate(vp):
        freq = 2 + (i % 7)
        phase = (i + 1) * 0.35 + 3.0 * e
        img += np.exp(-((r - 0.18 * (i % 5 + 1)) ** 2) / (0.012 + 0.03 * e))
        img += 0.8 * np.cos(10 * r - freq * theta + phase)
        img += 0.6 * np.sin((6 + 10 * e) * x + (4 + freq) * y + phase)

    img = gaussian_filter(img, sigma=taille / 250)
    img = normaliser_champ(img)

    halo = np.exp(-2.8 * r**2)
    filament = normaliser_champ(np.sin(12 * theta + 18 * r) + np.cos(7 * theta - 14 * r))

    rvb = np.zeros((taille, taille, 3))
    rvb[..., 0] = np.clip(0.95 * img + 0.35 * halo, 0, 1)
    rvb[..., 1] = np.clip(0.45 * filament + 0.55 * np.sqrt(img), 0, 1)
    rvb[..., 2] = np.clip(0.25 + 0.95 * gaussian_filter(1 - img, sigma=taille / 300), 0, 1)

    return np.clip(rvb, 0, 1)
