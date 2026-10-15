# MiniMax H3 arbitrary-frame guides

This pinned backport contains the three-file implementation from
[ComfyUI PR 15439](https://github.com/Comfy-Org/ComfyUI/pull/15439), previously
validated in the local H3 video project. Its upstream code is licensed under
the GNU GPL v3; see [LICENSE](LICENSE). The patch adds `MiniMaxH3AddGuide` and
the corresponding model coordinate handling; installing only the node is insufficient.

Stop the H3 ComfyUI instance, then check the installation:

```powershell
python scripts/install_h3_guides.py I:\Projects\ComfyUI\ComfyUI-H3
python scripts/install_h3_guides.py I:\Projects\ComfyUI\ComfyUI-H3 --apply
```

The installer accepts only the recorded source or already-patched hashes,
backs up original bytes under `h3-guide-backup-*` inside ComfyUI, and verifies
all three results. An unsupported/newer installation must be inspected separately;
do not force this patch onto it. Restart ComfyUI and check
`GET /object_info/MiniMaxH3AddGuide`. For rollback, stop ComfyUI and copy the
three files from the recorded backup to the same relative paths.

The unified video API exposes still-image guides using `keyframes`, with a
`frame_index` (negative counts from the end) or nonnegative `time_seconds`
(rounded to the nearest output frame). It accepts at most 16 guides, rejects
duplicates/out-of-range positions, and requires `prompt_mode=raw`.
Native ComfyUI additionally supports clip/audio guides; these are not exposed
by the unified API. Reference videos currently supply their visual track only
and must be prepared at 24 FPS, 2–15 seconds, as required by the H3 node.
