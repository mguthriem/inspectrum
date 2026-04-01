"""Validate: parse CIF symops, expand atoms, compute F² for ice-VII."""

import re
import numpy as np


def parse_symop(symop_str: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a CIF symmetry operation string like '-y, x+1/2, z+1/2'.

    Returns (3x3 rotation matrix, 3-vector translation).
    """
    rot = np.zeros((3, 3))
    trans = np.zeros(3)
    # Map variable names to column indices
    var_map = {"x": 0, "y": 1, "z": 2}

    parts = [p.strip() for p in symop_str.split(",")]
    for row, part in enumerate(parts):
        # Remove spaces
        part = part.replace(" ", "")
        # Extract fractional translations like +1/2, -1/4, etc.
        # and variable terms like +x, -y, z
        # We process character by character
        i = 0
        sign = 1
        while i < len(part):
            ch = part[i]
            if ch == "+":
                sign = 1
                i += 1
            elif ch == "-":
                sign = -1
                i += 1
            elif ch in var_map:
                rot[row, var_map[ch]] = sign
                sign = 1  # reset
                i += 1
            elif ch.isdigit():
                # Parse a fraction like 1/2 or 3/4 or integer
                num_str = ch
                i += 1
                while i < len(part) and (part[i].isdigit() or part[i] == "/"):
                    num_str += part[i]
                    i += 1
                if "/" in num_str:
                    num, den = num_str.split("/")
                    trans[row] += sign * int(num) / int(den)
                else:
                    trans[row] += sign * int(num_str)
                sign = 1
            else:
                i += 1
    return rot, trans


def expand_position(
    pos: np.ndarray, symops: list[tuple[np.ndarray, np.ndarray]], tol: float = 1e-5
) -> np.ndarray:
    """Apply all symops to a position, return unique positions in [0, 1)."""
    all_pos = []
    for rot, trans in symops:
        new_pos = rot @ pos + trans
        new_pos = new_pos % 1.0
        # Handle numerical issues near 1.0
        new_pos[new_pos > 1.0 - tol] = 0.0
        all_pos.append(new_pos)

    # Remove duplicates
    unique = [all_pos[0]]
    for p in all_pos[1:]:
        is_dup = False
        for u in unique:
            diff = np.abs(p - u)
            # Also check periodic boundary
            diff = np.minimum(diff, 1.0 - diff)
            if np.all(diff < tol):
                is_dup = True
                break
        if not is_dup:
            unique.append(p)
    return np.array(unique)


# ---- Parse ice-VII CIF symops ----
cif_symops_str = [
    "-y, x+1/2, z+1/2",
    "y+1/2, x+1/2, -z",
    "y+1/2, -x, z+1/2",
    "-y, -x, -z",
    "-z, x+1/2, y+1/2",
    "z+1/2, x+1/2, -y",
    "z+1/2, -x, y+1/2",
    "-z, -x, -y",
    "z+1/2, -y, x+1/2",
    "-z, y+1/2, x+1/2",
    "z+1/2, y+1/2, -x",
    "-z, -y, -x",
    "y+1/2, -z, x+1/2",
    "-y, z+1/2, x+1/2",
    "y+1/2, z+1/2, -x",
    "-y, -z, -x",
    "x+1/2, z+1/2, -y",
    "x+1/2, -z, y+1/2",
    "-x, z+1/2, y+1/2",
    "-x, -z, -y",
    "x+1/2, y+1/2, -z",
    "x+1/2, -y, z+1/2",
    "-x, y+1/2, z+1/2",
    "-x, -y, -z",
    "y, -x+1/2, -z+1/2",
    "-y+1/2, -x+1/2, z",
    "-y+1/2, x, -z+1/2",
    "y, x, z",
    "z, -x+1/2, -y+1/2",
    "-z+1/2, -x+1/2, y",
    "-z+1/2, x, -y+1/2",
    "z, x, y",
    "-z+1/2, y, -x+1/2",
    "z, -y+1/2, -x+1/2",
    "-z+1/2, -y+1/2, x",
    "z, y, x",
    "-y+1/2, z, -x+1/2",
    "y, -z+1/2, -x+1/2",
    "-y+1/2, -z+1/2, x",
    "y, z, x",
    "-x+1/2, -z+1/2, y",
    "-x+1/2, z, -y+1/2",
    "x, -z+1/2, -y+1/2",
    "x, z, y",
    "-x+1/2, -y+1/2, z",
    "-x+1/2, y, -z+1/2",
    "x, -y+1/2, -z+1/2",
    "x, y, z",
]

symops = [parse_symop(s) for s in cif_symops_str]
print(f"Parsed {len(symops)} symmetry operations")

# Expand O at (3/4, 3/4, 3/4) - Wyckoff 2a, expect 2 positions
pos_O = np.array([0.75, 0.75, 0.75])
O_positions = expand_position(pos_O, symops)
print(f"\nO at (3/4,3/4,3/4): {len(O_positions)} unique positions (expect 2)")
for p in O_positions:
    print(f"  ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")

# Expand D at (0.91012, 0.91012, 0.91012) - Wyckoff 8e, expect 8 positions
pos_D = np.array([0.91012, 0.91012, 0.91012])
D_positions = expand_position(pos_D, symops)
print(f"\nD at (0.91012,0.91012,0.91012): {len(D_positions)} unique positions (expect 8)")
for p in D_positions:
    print(f"  ({p[0]:.5f}, {p[1]:.5f}, {p[2]:.5f})")

# Neutron coherent scattering lengths (fm)
b_O = 5.803  # O-16
b_D = 6.671  # D (deuterium)

# Compute F^2 for reflections
reflections = [(0, 1, 1), (1, 1, 1), (0, 0, 2), (1, 2, 1), (0, 2, 2), (2, 2, 2)]
print(f"\nStructure factor calculations (full CIF symop expansion):")
print(f"{'hkl':>10s}  {'F_real':>10s}  {'F_imag':>10s}  {'F_sq':>10s}")

for hkl in reflections:
    h = np.array(hkl)
    F_real = 0.0
    F_imag = 0.0

    # O contribution (occ=1.0)
    for p in O_positions:
        phase = 2 * np.pi * np.dot(h, p)
        F_real += b_O * 1.0 * np.cos(phase)
        F_imag += b_O * 1.0 * np.sin(phase)

    # D contribution (occ=0.5)
    for p in D_positions:
        phase = 2 * np.pi * np.dot(h, p)
        F_real += b_D * 0.5 * np.cos(phase)
        F_imag += b_D * 0.5 * np.sin(phase)

    F_sq = F_real**2 + F_imag**2
    print(f"{str(hkl):>10s}  {F_real:10.4f}  {F_imag:10.4f}  {F_sq:10.4f}")

# Reference values from audit_fsq.py (full-cell computation)
print("\n--- Expected from audit script: ---")
print("(0,1,1): F_sq ~ 135.3 (from ratio * our old value)")
print("(1,1,1): F_sq ~ 0.0   (extinction)")
print("(0,0,2): F_sq ~ 0.0   (extinction)")
print("(1,2,1): F_sq ~ ?")
print("(0,2,2): F_sq ~ 135.3")
print("(2,2,2): F_sq ~ 582.6")
