# interactor-skintokens-auto-rig

A model image that rigs a mesh with the SkinTokens auto-rigger and writes the rig as a UsdSkel layer over the mesh layer.

## What it is for

The rig sits in its own USD layer, so a later retopology stage replaces the mesh layer below it and only the weights need a rebind. The service takes a mesh and a rig mode over HTTP and returns the rigged avatar. In stub mode it answers without the model; outside it, the upstream forward pass and the humanoid joint-map write are not yet ported into the server. RFD 1046 owns the design.

## Build and run

```sh
docker build --target contract -t interactor-skintokens-auto-rig:contract .
docker run --rm -p 8000:8000 interactor-skintokens-auto-rig:contract
curl -X POST localhost:8000/predict -d @test_input.json -H 'Content-Type: application/json'
```

The contract target is the stub image, and it answers `POST /predict`. `docker build .` builds the GPU worker image, which runs on a GPU the operator owns.

```sh
uv run --no-project --python 3.12 --with usd-core python check_rig_contract.py rig.usda
```

`check_rig_contract.py` gates a returned `rig.usda` against the rig contract: one skeleton with rest transforms, hips and head joints named, head above hips, lowest joint on the ground within one nickel (21.2 mm), hips at least 0.3 of head height. `--self-test` runs two positive and four negative controls and is a pre-commit hook. The stub layer carries no rest transforms, so it fails the first check by design until the forward pass lands.

## Licence

MIT. See [LICENSE](LICENSE). The upstream SkinTokens model is MIT.
