# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib>=3.10,<4",
#   "numpy>=2,<3",
# ]
# ///
"""f_NL of KMS trajectory 2 after inflation, and the resonance of phi.

Run from the repository root:

    uv run notes/overview/kms_after_inflation.py

It reads three CSV files next to this script and writes
``kms_after_inflation.{pdf,png}``. With ``KMS_BACKGROUND_TOLERANCE=1e-13``:

- ``kms_after_inflation.csv``: the split read off ``t`` e-folds after the
  end, ``kms_transport_probe -- after 1 4 <t,...>`` (3 for ``t <= 2.5``;
  the two agree to all printed digits).
- ``kms_after_inflation_orbit.csv``: the background,
  ``orbit 1 4 -0.2 3.8 0.0005``, and the restarts at ``t = 1`` with ``phi``
  scaled by 0.01, 0.1 and 10, ``orbit 1 4 1.0 3.8 0.0005 1.0 <scale>``.
- ``kms_floquet.csv``: ``floquet 1 <xi_scale> <A,...>`` for ``xi_scale``
  1, 0.5 and 0 (A from 0.001 to 0.1, every 0.0025 between 0.05 and 0.1).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
EDGE = 0.026  # chi amplitude below which phi grows (kms_floquet.csv)


def load(name):
    return np.genfromtxt(HERE / name, delimiter=",", names=True)


def half_oscillations(t, chi, phi):
    """(start, max|chi|, max|phi|) between consecutive zeros of chi."""
    zeros = np.nonzero(np.sign(chi[1:]) != np.sign(chi[:-1]))[0] + 1
    out = []
    for a, b in zip(zeros[:-1], zeros[1:]):
        out.append((t[a], np.abs(chi[a:b]).max(), np.abs(phi[a:b]).max()))
    return np.array(out)


def main():
    after = load("kms_after_inflation.csv")
    orbit = load("kms_after_inflation_orbit.csv")
    floquet = load("kms_floquet.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # Left: f_NL and P_zeta read off after the end.
    ax = axes[0, 0]
    spike = after["epsilon_h"] < 0.02
    ax.plot(after["t"][~spike], after["f_nl"][~spike], "o-", color="C0", ms=3)
    ax.plot(after["t"][spike], after["f_nl"][spike], "x", color="0.5",
            label=r"$\epsilon_H<0.02$")
    ax.set_yscale("symlog", linthresh=100)
    ax.set_ylabel(r"$f_{\rm NL}$")
    ax.axvspan(2.6, after["t"].max(), color="C3", alpha=0.1)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("read off on uniform-$H$ slices", fontsize=10)

    ax = axes[1, 0]
    p_end = after["p_zeta"][np.argmin(np.abs(after["t"]))]
    ax.plot(after["t"][~spike], after["p_zeta"][~spike] / p_end, "o-",
            color="C1", ms=3)
    ax.plot(after["t"][spike], after["p_zeta"][spike] / p_end, "x",
            color="0.5")
    ax.set_yscale("log")
    ax.set_ylabel(r"$P_\zeta / P_\zeta(t=0)$")
    ax.set_xlabel("e-folds after the end of inflation")
    ax.axvspan(2.6, after["t"].max(), color="C3", alpha=0.1)

    # Top right: the envelope of phi for four seeds, and of chi.
    ax = axes[0, 1]
    t = orbit["t"]
    for column, label, color in [
        ("phi_10", r"$\phi\times10$ at $t=1$", "C2"),
        ("phi", r"$\phi$", "C0"),
        ("phi_01", r"$\phi\times0.1$", "C4"),
        ("phi_001", r"$\phi\times0.01$", "C5"),
    ]:
        values = orbit[column]
        keep = np.isfinite(values)
        envelope = half_oscillations(t[keep], orbit["chi"][keep], values[keep])
        ax.plot(envelope[:, 0], envelope[:, 2], color=color, label=label)
    envelope = half_oscillations(t, orbit["chi"], orbit["phi"])
    ax.plot(envelope[:, 0], envelope[:, 1], "k--", label=r"$\chi$")
    inside = envelope[envelope[:, 1] < EDGE, 0].min()
    ax.axvspan(inside, t.max(), color="C3", alpha=0.1)
    ax.set_yscale("log")
    ax.set_ylabel("amplitude per half oscillation of $\\chi$")
    ax.set_xlabel("e-folds after the end of inflation")
    ax.legend(fontsize=7, loc="lower left")

    # Bottom right: growth per half period, Floquet against the background.
    ax = axes[1, 1]
    a = floquet["amplitude"]
    ax.plot(a, floquet["growth_xi10"], "-", color="C0", label=r"Floquet, $\xi$ (KMS)")
    ax.plot(a, floquet["growth_xi5"], "--", color="C0", label=r"Floquet, $\xi/2$")
    ax.plot(a, floquet["growth_xi0"], ":", color="k", label=r"Floquet, $\xi=0$")
    seed = orbit["phi_001"]
    keep = np.isfinite(seed)
    halves = half_oscillations(t[keep], orbit["chi"][keep], seed[keep])
    ratio = halves[:, 2] / halves[:, 1]
    growth = ratio[1:] / ratio[:-1]
    amplitude = 0.5 * (halves[1:, 1] + halves[:-1, 1])
    linear = (halves[:-1, 0] > 2.2) & (halves[:-1, 0] < 2.86)
    ax.plot(amplitude[linear], growth[linear], "o", color="C3", ms=3,
            label=r"expanding background, $|\phi|/|\chi|$")
    ax.set_xscale("log")
    ax.set_xlabel(r"amplitude $A$ of $\chi$ ($M_{\rm pl}$)")
    ax.set_ylabel("growth of $\\phi$ per half period")
    ax.legend(fontsize=7, loc="lower left")

    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"kms_after_inflation.{suffix}", dpi=150)


if __name__ == "__main__":
    main()
