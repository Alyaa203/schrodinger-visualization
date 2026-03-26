import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import streamlit as st

from scipy.ndimage import gaussian_filter
from scipy.fft import fft, fftfreq


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


def generer_rosace(valeurs_propres, taille=700):
    y, x = np.mgrid[-1:1:taille*1j, -1:1:taille*1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    img = np.zeros_like(r)

    vp = np.array(valeurs_propres)
    vp = vp - vp.min()
    if vp.max() > 0:
        vp = vp / vp.max()

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
