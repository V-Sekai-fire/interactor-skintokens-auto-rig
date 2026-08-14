"""SkinTokens auto rig. RFD 0046.

The rig is an opinion about a mesh that already exists, thus it goes
into its own USD layer over the mesh layer (RFD 0053). A GLB skin
binds to one mesh; when retopology later changes the mesh, that skin
breaks. A UsdSkel binding in its own layer survives a new mesh layer
below it, and only the weights need a rebind.
"""

import base64
import os
import tempfile
from pathlib import Path

STUB = os.environ.get("WEFTSPUN_STUB") == "1"
_READY = {"loaded": False}

# SkinTokens rejects template mode. RFD 0035 records that UniRig is
# the only backend for it -- this model image never accepts "template".
RIG_MODES = ("skeleton", "skin", "full")


class InputError(ValueError):
    """The request is wrong. This is the caller's fault, and not ours."""


def _validate(job_input: dict) -> dict:
    if not job_input.get("mesh"):
        raise InputError("mesh is required: a URL, a data URI, or base64 GLB/USD bytes")
    rig_mode = job_input.get("rig_mode", "full")
    if rig_mode not in RIG_MODES:
        raise InputError(f"rig_mode must be one of {RIG_MODES} (not 'template' -- see UniRig)")
    return {
        "mesh": job_input["mesh"],
        "rig_mode": rig_mode,
        "seed": int(job_input.get("seed", -1)),
    }


def _run_upstream(args: dict, work: Path) -> tuple:
    """Not yet wired -- the SkinTokens forward pass against the real
    checkpoint is not yet verified against the upstream repo."""
    raise NotImplementedError(
        "Port the SkinTokens forward pass here -- see VAST-AI-Research/SkinTokens "
        "and README's Status"
    )


def _write_joint_map(stage, joints: list) -> None:
    """Record the VRM humanoid mapping as layer metadata. USD keeps an
    ordered joint array with no humanoid meaning; VRM names its
    joints. An exporter that infers the mapping from joint names fails
    on any rig that names a joint differently -- RFD 0046's joint
    order trap -- so the mapping is written here and read later, not
    inferred downstream."""
    raise NotImplementedError("Port the VRM humanoid joint-map write here -- see README's Status")


def _to_usd(mesh: Path, joints: list, work: Path):
    """RFD 0053: the rig is a sublayer over the mesh layer, not baked
    into it -- a new mesh layer below it keeps the joint hierarchy."""
    from pxr import Sdf, Usd, UsdGeom, UsdSkel

    layer = work / "rig.usda"
    stage = Usd.Stage.CreateNew(str(layer))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    rig = stage.DefinePrim("/Asset/Rig")
    rig.CreateAttribute("weftspun:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(mesh.name))
    rig.CreateAttribute("weftspun:stage", Sdf.ValueTypeNames.Token).Set("skintokens_auto_rig")
    if STUB:
        skeleton = UsdSkel.Skeleton.Define(stage, "/Asset/Rig/Skeleton")
        skeleton.CreateJointsAttr(joints)
    else:
        _write_joint_map(stage, joints)
    stage.GetRootLayer().Save()
    return layer


def _encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def predict(job_input: dict) -> dict:
    args = _validate(job_input)
    work = Path(tempfile.mkdtemp())

    if STUB:
        joints = ["hips", "spine", "chest", "neck", "head"]
        mesh = work / "input.mesh"
        mesh.write_bytes(base64.b64decode(args["mesh"]) if not args["mesh"].startswith(
            ("http://", "https://")
        ) else b"stub")
    else:
        mesh, joints = _run_upstream(args, work)

    layer = _to_usd(mesh, joints, work)

    vrm = work / ("stub.vrm" if STUB else "output.vrm")
    if STUB:
        vrm.write_bytes(b"glTFstub")

    return {
        "layer": _encode(layer),
        "vrm": _encode(vrm),
        "joint_count": len(joints),
        "seed": args["seed"],
        "stub": STUB,
    }


def load() -> None:
    _READY["loaded"] = True


def build_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="skintokens_auto_rig", version="0.1.0")

    class PredictRequest(BaseModel):
        mesh: str
        rig_mode: str = "full"
        seed: int = -1

    @app.get("/health")
    def health():
        return {"status": "ok", "ready": _READY["loaded"], "stub": STUB}

    @app.post("/predict")
    def run(request: PredictRequest):
        try:
            return predict(request.model_dump())
        except InputError as error:
            return JSONResponse(status_code=400, content={"error": str(error)})

    return app


if __name__ == "__main__":
    import uvicorn

    load()
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
