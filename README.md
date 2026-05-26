# ComfyUI Color Correct GPU

A small ComfyUI custom node that mirrors the controls and processing order from
Art Venture's `ColorCorrect` node while doing the image math with vectorized
PyTorch ops.

## Node

`Color Correct GPU`

Category: `ArtVenture/Post Processing`

Inputs match the Art Venture node:

- `image`
- `temperature`
- `hue`
- `brightness`
- `contrast`
- `saturation`
- `gamma`

The node ID is `ColorCorrectGPU`, so it can be installed beside
`comfyui-art-venture` without colliding with its existing `ColorCorrect` node.

## Notes

- This avoids the upstream CPU path that converts each frame through NumPy, PIL,
  and OpenCV.
- The GPU path intentionally keeps the upstream 8-bit quantization steps and
  PIL/OpenCV-style color math so results stay close to the original node.
- Neutral saturation and hue settings skip their color-space conversions to
  avoid unnecessary work and tiny no-op conversion drift.
- If the incoming ComfyUI `IMAGE` tensor is on CPU, the node moves it to
  ComfyUI's torch device when one is available, then returns it to ComfyUI's
  intermediate device for compatibility.
- Results should visually match the original node, but may not be byte-identical
  because PIL/OpenCV and torch can differ slightly in floating-point edge cases.
