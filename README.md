<div align="center">

# Schrödinger Visualization

**An interactive Streamlit app that solves the Schrödinger equation with two numerical methods, compares them, and turns the results into art.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

</div>

> **Note:** this repository is a development version of **SchrödArt**. The latest version, with its live demo, is in **[Alyaa203/P2i](https://github.com/Alyaa203/P2i)**.

---

## Overview

This app simulates how a quantum particle behaves (the Schrödinger equation) and lets you explore the results with interactive plots.

**Why it exists:** it was built for an individual engineering project at ENSC (Bordeaux INP), to implement two independent solvers, check that they agree, and make quantum physics visual.

**At a glance:**
- Two solvers built from scratch: **modal decomposition** (sparse diagonalisation) and **Split-Step Fourier** (FFT)
- A **cross-validation** tab that measures the error between the two methods
- A **DST-based variant** of Split-Step Fourier that matches the zero boundary conditions of the modal method
- Energy spectra turned into **5 styles of generative art**

---

## Features

- **2D stationary regime:** computes the lowest-energy eigenstates of a 2D Gaussian potential and shows each mode
- **1D time-dependent regime:** probability density over time, with a 3D surface view
- **Split-Step Fourier:** wave-packet simulations for **tunnelling**, the **harmonic oscillator** and the **double well**, with snapshots at 3 instants
- **Quantum art:** rosette, nebula, crystal, mandala and galaxy images generated from the computed eigenvalues
- **Cross-validation:** both methods run on the same problem, with plots and the relative error between them
- All parameters adjustable live with sliders

---

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Numerical computing | NumPy (FFT), SciPy (sparse matrices, `eigsh` / ARPACK, `eigh_tridiagonal`) |
| Visualisation | Matplotlib, Pillow |
| Web interface | Streamlit |

---

## Getting started

Requires Python 3.9 or later.

```bash
git clone https://github.com/Alyaa203/schrodinger-visualization.git
cd schrodinger-visualization
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open http://localhost:8501.

### Project structure

```
├── streamlit_app.py   # Web interface (5 tabs)
├── simulation.py      # Modal method (2D stationary + 1D time-dependent)
├── Fourier.py         # Split-Step Fourier method (FFT and DST variants)
├── visualisation.py   # Generative art
└── requirements.txt
```

---

**Author:** Alyaa Saab, engineering student at ENSC (Bordeaux INP)
