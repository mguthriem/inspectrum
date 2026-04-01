"""Quick audit: is our F(hkl) calculation correct?"""
import numpy as np

# Scattering lengths (fm)
b_O = 5.803
b_D = 6.671

# Ice VII, Pn-3m Origin 2
# 2a site: O at (3/4,3/4,3/4) -> generates 2 positions
o_positions = [(0.75, 0.75, 0.75), (0.25, 0.25, 0.25)]

# 8e site: D at (x,x,x) x=0.91012 -> generates 8 positions
x = 0.91012
d_positions = [
    (x, x, x),
    (1 - x, 1 - x, x),
    (1 - x, x, 1 - x),
    (x, 1 - x, 1 - x),
    (0.5 + x, 0.5 + x, 0.5 + x),
    (0.5 - x, 0.5 - x, 0.5 + x),
    (0.5 - x, 0.5 + x, 0.5 - x),
    (0.5 + x, 0.5 - x, 0.5 - x),
]

print("=== F-squared comparison for several (hkl) ===")
print(f"{'hkl':>10s}  {'F2_full':>10s}  {'F2_ours':>10s}  {'ratio':>8s}")

for (h, k, l) in [(0, 1, 1), (1, 1, 1), (0, 0, 2), (1, 2, 1), (0, 2, 2)]:
    # Full cell
    F_full = 0.0 + 0.0j
    for (xp, yp, zp) in o_positions:
        phase = 2 * np.pi * (h * xp + k * yp + l * zp)
        F_full += b_O * 1.0 * np.exp(1j * phase)
    for (xp, yp, zp) in d_positions:
        phase = 2 * np.pi * (h * xp + k * yp + l * zp)
        F_full += b_D * 0.5 * np.exp(1j * phase)
    Fsq_full = abs(F_full) ** 2

    # Our code: only 2 CIF sites, no symmetry expansion
    F_ours = 0.0 + 0.0j
    for (xp, yp, zp), b_val, occ in [
        (o_positions[0], b_O, 1.0),
        ((x, x, x), b_D, 0.5),
    ]:
        phase = 2 * np.pi * (h * xp + k * yp + l * zp)
        F_ours += b_val * occ * np.exp(1j * phase)
    Fsq_ours = abs(F_ours) ** 2

    ratio = Fsq_full / Fsq_ours if Fsq_ours > 1e-10 else float("inf")
    print(f"({h},{k},{l}):  {Fsq_full:10.4f}  {Fsq_ours:10.4f}  {ratio:8.2f}")

print()
print("If ratio != 1.0, our code is wrong.")
print("Our code only sums over CIF asymmetric unit + centering translations.")
print("It must expand by full space group operations to get correct F-squared.")
