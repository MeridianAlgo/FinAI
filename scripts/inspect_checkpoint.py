import torch, os

p = "checkpoints/checkpoint-50.pt"
if not os.path.exists(p):
    print("MISSING")
else:
    ckpt = torch.load(p, map_location="cpu")
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    for k in [
        "transformer.transformer.wte.weight",
        "transformer.transformer.wpe.weight",
        "transformer.lm_head.weight",
    ]:
        if k in state:
            print(k, state[k].shape)
    layers = set()
    for k in state.keys():
        if k.startswith("transformer.transformer.h."):
            parts = k.split(".")
            if len(parts) > 3:
                try:
                    layers.add(int(parts[3]))
                except:
                    pass
    layers_sorted = sorted(list(layers))
    print("n_layers found sample:", layers_sorted[:5], "... total", len(layers_sorted))
    keys = list(state.keys())
    print("sample keys:", keys[:10])
