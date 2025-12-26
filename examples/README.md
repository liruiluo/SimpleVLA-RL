# Examples

Directory layout:

- `examples/libero/openvla_oft/`: OpenVLA-OFT RL launchers for LIBERO.
- `examples/libero/vla_adapter_token/`: VLA-Adapter token RL launchers for LIBERO.
  - `examples/libero/vla_adapter_token/1img/`: 1-image checkpoint launchers (grouped here).
- `examples/libero/vla_adapter_token/1img/*_1img_4xa100.sh`: 1-image 4-GPU launchers for `libero_spatial/libero_object/libero_goal/libero_long`.
- `examples/twin2/`: RoboTwin2.0 launcher + task list.
- `examples/utils/`: helper scripts (e.g. checkpoint code sync).
