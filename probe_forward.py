"""First light: the upstream TokenRig forward pass on one mesh, no bpy anywhere.

Loads the example mesh with trimesh, feeds it through the npz loader path, and prints
what comes back: joint count, skin shape, weight-row sums. A wrong-mesh negative
control (a cube) must still return a structurally valid rig, so the assertion here is
shape and normalization, not quality.
"""
import argparse
import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

UPSTREAM = os.environ.get(
    "SKINTOKENS_UPSTREAM",
    r"C:\Users\ernes\AppData\Local\Temp\claude\C--weftspun-keypoints"
    r"\42df4a07-57d8-4d08-89ba-742cfb7f729c\scratchpad\skintokens")
sys.path.insert(0, UPSTREAM)

# No flash-attn on this desk: force sdpa on the Qwen backbone, and keep the removed
# sdp_kernel context manager from crashing the mesh encoder on torch >= 2.13.
if not hasattr(torch.backends.cuda, "sdp_kernel"):
    torch.backends.cuda.sdp_kernel = lambda **kw: contextlib.nullcontext()

# Stand-in flash_attn_interface: same [B, S, H, D] layout and (out, lse) return,
# computed through sdpa.
import types


def _sdpa_flash_attn_func(q, k, v, *args, **kwargs):
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2), None


_fai = types.ModuleType("flash_attn_interface")
_fai.flash_attn_func = _sdpa_flash_attn_func
sys.modules["flash_attn_interface"] = _fai

import src.model.tokenrig as tokenrig_mod
import src.model.skin_vae.attention_processor  # binds flash_attn_func at import

# Gone again so transformers' find_spec-based availability probe sees no flash-attn.
del sys.modules["flash_attn_interface"]

_real = tokenrig_mod.AutoModelForCausalLM


class _SdpaAuto:
    @staticmethod
    def from_config(config, **kw):
        kw["attn_implementation"] = "sdpa"
        return _real.from_config(config, **kw)


tokenrig_mod.AutoModelForCausalLM = _SdpaAuto

from src.data.dataset import DatasetConfig, RigDatasetModule
from src.data.transform import Transform
from src.server.spec import get_model
from src.tokenizer.parse import get_tokenizer


def mesh_to_npz(mesh_path: str, npz_path: str):
    scene = trimesh.load(mesh_path, force="scene")
    m = scene.to_geometry() if hasattr(scene, "to_geometry") else scene
    np.savez(npz_path,
             vertices=np.asarray(m.vertices, dtype=np.float32),
             faces=np.asarray(m.faces, dtype=np.int64))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=os.path.join(UPSTREAM, "examples", "giraffe.glb"))
    ap.add_argument("--ckpt", default=os.path.join(
        UPSTREAM, "experiments", "articulation_xl_quantization_256_token_4", "grpo_1400.ckpt"))
    ap.add_argument("--out", default="probe_asset.npz")
    ap.add_argument("--seed", type=int, default=1171)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    os.chdir(UPSTREAM)  # relative paths inside the ckpt configs resolve from the repo root

    model = get_model(a.ckpt, hf_path=None)
    tokenizer = get_tokenizer(**model.tokenizer_config)
    transform = Transform.parse(**model.transform_config["predict_transform"])
    print("model loaded: tokens_per_skin=%s" % model.tokens_per_skin)

    npz_path = os.path.abspath("probe_mesh.npz")
    m = mesh_to_npz(a.mesh, npz_path)
    print("mesh: %d vertices, %d faces (%s)" % (len(m.vertices), len(m.faces),
                                                os.path.basename(a.mesh)))

    cfg = DatasetConfig.parse(
        shuffle=False, batch_size=1, num_workers=0, pin_memory=False,
        persistent_workers=False,
        datapath={"data_name": None, "loader": "npz",
                  "filepaths": {"articulation": [npz_path]}},
    ).split_by_cls()
    module = RigDatasetModule(predict_dataset_config=cfg, predict_transform=transform,
                              tokenizer=tokenizer, process_fn=model._process_fn)
    batch = next(iter(module.predict_dataloader()["articulation"]))
    batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch.pop("skeleton_tokens", None)
    batch.pop("skeleton_mask", None)
    batch["generate_kwargs"] = dict(max_length=2048, top_k=50, top_p=0.95,
                                    temperature=0.6, repetition_penalty=1.0,
                                    num_return_sequences=1, num_beams=1, do_sample=True)

    preds = model.predict_step(batch, make_asset=True)["results"]
    asset = preds[0].asset
    assert asset is not None, "predict_step returned no asset"

    J = asset.joints.shape[0]
    skin = asset.skin
    row = skin.sum(axis=1)
    print("")
    print("joints: %d   skin: %s   parents: %s" % (J, skin.shape, asset.parents.shape))
    print("skin row sums: min %.4f  max %.4f  (1.0 = normalized)" % (row.min(), row.max()))
    print("joints per vertex >1%%: mean %.2f" % (skin > 0.01).sum(axis=1).mean())
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.out)
    np.savez(out, joints=asset.joints, parents=asset.parents, skin=skin,
             vertices=asset.vertices, faces=asset.faces,
             names=np.array(asset.joint_names if asset.joint_names is not None else []))
    print("asset saved: %s" % out)


if __name__ == "__main__":
    main()
