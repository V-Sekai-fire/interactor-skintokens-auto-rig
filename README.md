# interactor-skintokens-auto-rig

Model image for `skintokens_auto_rig`, per
[weftspun's RFD 0036](https://github.com/weftspun/request-for-discussion/tree/main/0036-packaging-convention)
packaging convention. Facts from
[RFD 0046](https://github.com/weftspun/request-for-discussion/tree/main/0046-skintokens-auto-rig).

## Model

| Property   | Value                                                                                                                                                                                                                                                                                                        |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Upstream   | [VAST-AI-Research/SkinTokens](https://github.com/VAST-AI-Research/SkinTokens)                                                                                                                                                                                                                                |
| License    | **MIT** — RFD 0046 marked this "review pending"; it isn't ambiguous. Confirmed by reading the real `LICENSE` file at `VAST-AI-Research/SkinTokens` directly (`gh api repos/VAST-AI-Research/SkinTokens/contents/LICENSE`), not inferred. Flagging for whoever owns RFD 0046 to close out the pending status. |
| Parameters | 0.5 B, estimated                                                                                                                                                                                                                                                                                             |
| bf16       | 1.0 GB — the ship format (no Q4_K_M at this size)                                                                                                                                                                                                                                                            |

## Interface

`POST /predict`:

| Input      | Type            | Default  | Note                                                                                                                                          |
| ---------- | --------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `mesh`     | Path/URL/base64 | required | GLB or USD                                                                                                                                    |
| `rig_mode` | str             | full     | `skeleton`, `skin`, or `full` — **not** `template`; SkinTokens rejects it, RFD 0035 records that UniRig is the only backend for template mode |
| `seed`     | int             | -1       |                                                                                                                                               |

Returns `{layer, vrm, joint_count, seed, stub}`.

## Why UsdSkel, not a GLB skin

A GLB skin binds to one mesh. When a later retopology stage changes the mesh, the skin breaks and
the rig stage has to re-run. A `UsdSkel` binding sits in its own layer instead (RFD 0053); a new
mesh layer below it keeps the joint hierarchy, so only the weights need a rebind.

## The joint order trap (RFD 0046)

VRM names its humanoid joints; USD keeps an ordered array with no humanoid meaning. The mapping
between them has to live in the layer as metadata, written at rig time — an exporter that tries to
infer the mapping from joint names later fails on any rig that names a joint differently.

## Build

```sh
docker build --target contract -t interactor-skintokens-auto-rig:contract .
docker run --rm -p 8000:8000 interactor-skintokens-auto-rig:contract
curl -X POST localhost:8000/predict -d @test_input.json -H 'Content-Type: application/json'
```

## Status

**Scaffolded from the RFD, not yet built or run.** `_run_upstream()` and `_write_joint_map()`
raise `NotImplementedError` outside stub mode — SkinTokens' real forward pass and the VRM humanoid
joint-map write are not yet ported from the upstream repo. Confirm the exact HF checkpoint repo id
before trusting the worker stage's weight fetch.
