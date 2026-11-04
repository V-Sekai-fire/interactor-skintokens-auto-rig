#!/usr/bin/env python3
"""Gate a UsdSkel rig layer against the rig contract.

The contract comes from the retired browser studio's `[API-Contract] PASS`
check, carried over to the layer this service writes: the skeleton stands the
right way up, its feet sit on the ground, and its hips are not floating. Each
check measures a position in the stage's own up axis and units, read from the
layer rather than assumed.

    python check_rig_contract.py rig.usda        # exit 0 on PASS, 1 on FAIL
    python check_rig_contract.py --self-test     # positive and negative controls

Checks, in the order they run (an unmet precondition is a FAIL, never a skip):

1. skeleton   exactly one UsdSkel.Skeleton with a non-empty joints array and a
              restTransforms array of the same length
2. names      a hips joint and a head joint are present (humanoid map from the
              layer's `weftspun:humanoid` dictionary when it exists, else the
              joint path's last component, case-insensitive)
3. upright    head is above hips along the up axis by at least HEAD_ABOVE_HIPS
4. grounded   the lowest joint sits within GROUND_TOLERANCE of 0 along the up axis
              (a nickel is 21.2 mm; the tolerance is one nickel)
5. hips       hips sit at least HIPS_MIN_FRACTION of the head height above the
              ground (a floating rig has its hips near the head or above it)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

GROUND_TOLERANCE = 0.0212  # metres, one nickel
HEAD_ABOVE_HIPS = 0.10  # metres, about the long side of a credit card plus a soda can
HIPS_MIN_FRACTION = 0.30

HIPS_NAMES = {"hips", "pelvis", "root_hips"}
HEAD_NAMES = {"head"}


def _world_rest_positions(skel, up_index, units):
    """Joint rest positions in world space along every axis, in metres."""
    from pxr import UsdSkel

    joints = list(skel.GetJointsAttr().Get() or [])
    rest = list(skel.GetRestTransformsAttr().Get() or [])
    if len(joints) != len(rest):
        return None, f"joints {len(joints)} != restTransforms {len(rest)}"
    topo = UsdSkel.Topology(joints)
    ok, why = topo.Validate()
    if not ok:
        return None, f"topology: {why}"
    world = UsdSkel.ConcatJointTransforms(topo, rest)
    positions = {}
    for path, xf in zip(joints, world):
        t = xf.ExtractTranslation()
        positions[path] = tuple(float(c) * units for c in (t[0], t[1], t[2]))
    return positions, None


def _humanoid(skel_prim, positions):
    """Map the contract's two humanoid joints to joint paths."""
    attr = skel_prim.GetAttribute("weftspun:humanoid")
    mapping = dict(attr.Get()) if attr and attr.Get() else {}
    hips = mapping.get("hips")
    head = mapping.get("head")
    if hips is None or head is None:
        for path in positions:
            leaf = path.split("/")[-1].lower()
            if hips is None and leaf in HIPS_NAMES:
                hips = path
            if head is None and leaf in HEAD_NAMES:
                head = path
    return hips, head


def check(layer_path: str) -> list[tuple[str, bool, str]]:
    from pxr import Usd, UsdGeom, UsdSkel

    results = []
    stage = Usd.Stage.Open(layer_path)
    if stage is None:
        return [("skeleton", False, f"cannot open {layer_path}")]
    up = UsdGeom.GetStageUpAxis(stage)
    up_index = {"Y": 1, "Z": 2}.get(str(up), None)
    units = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    if up_index is None:
        return [("skeleton", False, f"up axis {up!r} is neither Y nor Z")]

    skels = [p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)]
    if len(skels) != 1:
        return [("skeleton", False, f"{len(skels)} skeletons, contract wants 1")]
    skel = UsdSkel.Skeleton(skels[0])
    positions, err = _world_rest_positions(skel, up_index, units)
    if positions is None or not positions:
        return [("skeleton", False, err or "no joints")]
    results.append(("skeleton", True, f"{len(positions)} joints, up={up}, {units} m/unit"))

    hips, head = _humanoid(skels[0], positions)
    names_ok = hips is not None and head is not None
    results.append(("names", names_ok, f"hips={hips} head={head}"))
    if not names_ok:
        return results

    h_hips = positions[hips][up_index]
    h_head = positions[head][up_index]
    lowest = min(p[up_index] for p in positions.values())

    results.append(("upright", h_head - h_hips >= HEAD_ABOVE_HIPS,
                    f"head {h_head:.3f} m, hips {h_hips:.3f} m, head-hips {h_head - h_hips:.3f} m (>= {HEAD_ABOVE_HIPS})"))
    results.append(("grounded", abs(lowest) <= GROUND_TOLERANCE,
                    f"lowest joint {lowest:.4f} m from ground (|x| <= {GROUND_TOLERANCE}, one nickel)"))
    height = h_head - lowest
    results.append(("hips", height > 0 and (h_hips - lowest) >= HIPS_MIN_FRACTION * height,
                    f"hips at {(h_hips - lowest) / height if height else 0:.2f} of head height (>= {HIPS_MIN_FRACTION})"))
    return results


def report(layer_path: str) -> bool:
    results = check(layer_path)
    passed = all(ok for _, ok, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'} {name:9s} {detail}")
    print(f"{'PASS' if passed else 'FAIL'} rig contract: {sum(ok for _, ok, _ in results)} of {len(results)} checks, {layer_path}")
    return passed


def _write_rig(path: Path, heights: dict, up: str = "Y", units: float = 1.0, drop_head: bool = False) -> None:
    """A five-joint chain with the given joint heights along the up axis."""
    from pxr import Gf, Usd, UsdGeom, UsdSkel

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y if up == "Y" else UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, units)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    skel = UsdSkel.Skeleton.Define(stage, "/Asset/Rig/Skeleton")
    order = ["hips", "spine", "chest", "neck", "head", "foot_l", "foot_r"]
    if drop_head:
        order.remove("head")
    joints = [f"{j}" for j in order]  # a flat hierarchy: every joint is a root, so rest == world
    rests = []
    for j in order:
        v = [0.0, 0.0, 0.0]
        v[1 if up == "Y" else 2] = heights[j] / units
        rests.append(Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(*v)))
    skel.CreateJointsAttr(joints)
    skel.CreateRestTransformsAttr(rests)
    stage.GetRootLayer().Save()


def self_test() -> bool:
    good = {"hips": 0.95, "spine": 1.10, "chest": 1.30, "neck": 1.55, "head": 1.65, "foot_l": 0.0, "foot_r": 0.0}
    cases = [
        ("positive upright grounded Y", dict(heights=good), True),
        ("positive Z-up centimetres", dict(heights=good, up="Z", units=0.01), True),
        ("negative inverted", dict(heights={**good, "hips": 1.65, "head": 0.95}), False),
        ("negative floating 0.5 m", dict(heights={k: v + 0.5 for k, v in good.items()}), False),
        ("negative hips at the head", dict(heights={**good, "hips": 1.60}), False),
        ("negative no head joint", dict(heights=good, drop_head=True), False),
    ]
    all_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for name, kwargs, expect in cases:
            p = Path(tmp) / (name.replace(" ", "_") + ".usda")
            _write_rig(p, **kwargs)
            got = all(ok for _, ok, _ in check(str(p)))
            ok = got == expect
            all_ok &= ok
            print(f"{'PASS' if ok else 'FAIL'} self-test {name}: gate said {'PASS' if got else 'FAIL'}, expected {'PASS' if expect else 'FAIL'}")
    print(f"{'PASS' if all_ok else 'FAIL'} self-test: {len(cases)} cases, {sum(1 for _, _, e in cases if not e)} negative controls")
    return all_ok


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        sys.exit(0 if self_test() else 1)
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(0 if report(sys.argv[1]) else 1)
