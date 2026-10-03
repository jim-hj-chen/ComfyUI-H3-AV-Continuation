# ComfyUI-H3-AV-Continuation

**H3 Audio-Video Continuation**

[English](README.md) | [简体中文](README.zh-CN.md)

ComfyUI nodes for long MiniMax H3 workflows: segment planning, multi-image reference conditioning, paired AV continuation, audio protection during video refinement, frame-aligned deduplication, disk output and final assembly. Generation and sampling use native ComfyUI H3 nodes.

Uses 24 fps. Video windows must satisfy `17k+5`; the planner accepts 124–362 frames. Continuation context also uses `17k+5`, defaulting to 22 frames (about 0.917 s). Requested durations round up to a compatible window, so actual new duration can be longer. Context supports continuity but does not guarantee lip sync, voice or presenter identity for every generation.

## Audio modes and existing Goohaitool controls

Use the installed Goohaitool **Ignore Multiple Groups** controller for all optional stages. Set its filter to `处理 ·`, mode to `default`, and leave its three rows (second pass, upscale, AudioRefine) off initially. Keep Set/Get ports and VAE decoders outside these bypass groups. No custom switch nodes or mutual-exclusion JavaScript are required.

Use a second Goohaitool controller with filter `音频 ·` and mode `at_most_one`. Its two separate rows, Voice Reference and Audio Drive, are mutually exclusive and can both be off. Each group contains a STRING constant (`voice_reference` or `audio_drive`). Connect them to `GoohaiAnySwitch` slots 1 and 2, and an always-active `generated` constant to slot 3. Connect the result to the planner `audio_mode`, and Load Audio to `source_audio`. Loading is lazy and skipped in generated mode.

Voice Reference uses up to three seconds of loaded speech as timbre reference for new scripted dialogue. Audio Drive encodes the continuous source recording into the native audio latent and freezes it with the native stream noise mask. Connect planner `driving_audio` to both the reference encoder and the segment writer, and `kept_frames` to the writer. The writer uses source speech and skips audio decoding when driving audio is present. Retained frames are `round(seconds × 24)`; discarded H3 grid padding does not enter the soundtrack or continuation tail. The source must cover the sum of segment durations. Mouth-motion quality still depends on H3. Keep AudioRefine off when using Audio Drive, as its audio processing would be discarded.

Reference sizing defaults to **max**. Optional second pass, SeedVR2 upscale, AudioRefine, and segment AV previews are off by default in the supplied workflow. AudioRefine remains available through the external [ComfyUI-H3-AudioRefine](https://github.com/Adudeguyman/ComfyUI-H3-AudioRefine) extension. Do not mix the old custom switches with group bypass.

The latent upscaler's `force_unload` moves its own cached weights to CPU; it does not directly unload H3. Leaving those weights resident consumes GPU memory and is not a proven speed improvement on a 32GB GPU. FFmpeg retains bounded waits (600 seconds by default), interrupt handling, and start/end logging; this does not time out GPU sampling. Original node IDs and planner outputs 0–6 remain compatible. New inputs are optional and appended.

## Installation

Clone or extract this repository into `ComfyUI/custom_nodes/ComfyUI-H3-AV-Continuation/`, with `__init__.py` directly inside that directory. Install dependencies using the same Python environment that runs ComfyUI:

```bash
python -m pip install -r custom_nodes/ComfyUI-H3-AV-Continuation/requirements.txt
```

Windows portable edition (run from the portable root):

```powershell
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-H3-AV-Continuation\requirements.txt
```

Restart ComfyUI and refresh the browser. FFmpeg must provide `libx264` and `aac`; the pack uses FFmpeg on PATH, falling back to the executable bundled with imageio-ffmpeg.

Requires a MieLoop pack providing `MIE_LOOP_CTX` v3, containing `version=3`, a 24-character lowercase hexadecimal `run_id`, zero-based `index`, and a positive `count`. Loop nodes are not included.

Requires a ComfyUI build with `MiniMaxH3ReferenceToVideo`, `MiniMaxH3AddGuide` and `comfy.nested_tensor.NestedTensor`. Configure H3 model weights, text encoder, video/audio VAEs and sampling nodes separately; model weights are not included.

The reference node accepts `IMAGE_LIST` from an external upload node: `{"items": [{"path": "absolute/local/image/path"}, ...]}`, with 1–9 images. An ordinary IMAGE batch is not accepted; references are loaded separately in order.

Optionally set `H3_FFMPEG` to the full path of the FFmpeg executable.

## Language and compatibility

Plugin categories, node titles, parameter names, input/output labels, tooltips and combo choices have English and Chinese translations. They follow `Comfy > Locale > Language` (`Comfy.Locale`); `zh` uses Simplified Chinese and other unprovided languages fall back to English.

Translations use native `locales/en` and `locales/zh`. A frontend extension also updates existing canvas nodes on language changes and preserves custom titles. Requires ComfyUI with `/api/i18n`; reload the page if an older frontend leaves the node search menu stale. GitHub repository and installation directory names are fixed identifiers.

Original node IDs, socket keys, input order, defaults and output directories are preserved for existing workflows.

When migrating, move the old `h3_av_sync` folder out of custom_nodes to avoid duplicate registrations.

[ComfyUI official i18n documentation](https://docs.comfy.org/custom-nodes/i18n)

## Typical connections

1. Connect MieLoop v3 `loop_ctx` to the segment planner. Supply all scripts to `prompts` and a connected numeric value to `seconds`. Optionally select a script from a splitter list and connect `segment_prompt` for consistency validation.
2. Connect planner `prompt`, `window_frames`, `previous_frames`, `previous_audio` and `speaker_reference` to the reference conditioning node; connect `window_frames` to `length`. Supply H3 CLIP, the image list and matching VAEs. Continuation and speaker outputs are empty on the first segment.
3. Sample using native joint H3 nodes. Use the continuation guide on positive conditioning if native anchoring is needed. For video-only refinement, lock audio before the second pass and restore the first-pass audio afterward, then decode both streams.
4. Connect decoded video and audio to the synchronized writer. Connect planner `window_frames` to `expected_frames` and `skip_frames` to its matching input; keep context frames consistent. Route the returned `loop_ctx` back into MieLoop so the segment is committed before advancing.
5. Connect the final `loop_ctx` and completion flag `done` to final assembly.

## Nodes and parameters

### H3 Segment Planner

`H3AVSegmentPlan`

Plan a 24 fps H3 generation window, script timing, deterministic seed and paired continuation context.

| Input (internal key) | Label | Type / default | Purpose |
| --- | --- | --- | --- |
| `loop_ctx` | Loop Context | MIE_LOOP_CTX | Connect the MieLoop v3 context for this run and segment index. |
| `prompts` | All Segment Scripts | STRING / connection required | One STRING with exactly one nonempty script per loop segment, separated by ----------SEGMENT_SEPARATOR----------. |
| `seconds` | New Segment Duration (s) | FLOAT / connection required | Requested new content duration. The actual duration is rounded to an H3-compatible frame window at 24 fps. |
| `overlap_frames` | Continuation Context Frames | INT / `22` | Shared video/audio context. Must be 17k+5, such as 5, 22 or 39; use the same value in the planner and writer. |
| `base_seed` | Base Seed | INT / `1087183784971949` | Each segment uses (base seed + segment index) modulo 2^64. |
| `segment_prompt` | Current Segment Script | STRING / optional / connection required | The selected raw script. If connected to the planner, it must match the script at the current index. |

| Output index | Label | Type | Meaning |
| --- | --- | --- | --- |
| 0 | Generation Prompt | `STRING` | Text conditioning prompt. The planner adds timing, joint AV and continuation instructions. |
| 1 | Generation Window Frames | `INT` | Total frames to generate, including continuation context after the first segment. |
| 2 | Previous Video Context | `IMAGE` | Final context frames from the preceding segment; empty on the first segment. |
| 3 | Previous Audio Context | `AUDIO` | Audio aligned with the preceding video context; empty on the first segment. |
| 4 | Duplicate Context Frames to Remove | `INT` | Zero for the first segment; equals context frames thereafter. Remove the matching audio interval too. |
| 5 | Segment Seed | `INT` | Deterministic seed derived from the base seed and current zero-based segment index. |
| 6 | Speaker Voice Reference | `AUDIO` | Up to the first three seconds of the first segment audio, used as a speaker identity reference in later segments. |

### H3 Segment Prompt Selector

`H3AVPickPrompt`

Select the current script from a ComfyUI output list or a wrapped Python list using the MieLoop segment index.

| Input (internal key) | Label | Type / default | Purpose |
| --- | --- | --- | --- |
| `prompts` | Split Segment Scripts | STRING / connection required | Connect a list of STRING scripts, one per segment, in generation order. |
| `count` | Segment Count | INT / connection required | Number of scripts; must agree with the MieLoop context. |
| `loop_ctx` | Loop Context | MIE_LOOP_CTX | Connect the MieLoop v3 context for this run and segment index. |

| Output index | Label | Type | Meaning |
| --- | --- | --- | --- |
| 0 | Current Segment Script | `STRING` | The selected raw script. If connected to the planner, it must match the script at the current index. |

### H3 Lock Audio for Video Refinement

`H3AVFreezeAudio`

Create native per-stream noise masks so the second pass refines video while keeping audio fixed.

| Input (internal key) | Label | Type / default | Purpose |
| --- | --- | --- | --- |
| `latent` | Joint Video and Audio Latent | LATENT | Native H3 NestedTensor latent containing the video and audio streams. |

| Output index | Label | Type | Meaning |
| --- | --- | --- | --- |
| 0 | Joint Video and Audio Latent | `LATENT` | Native H3 NestedTensor latent containing the video and audio streams. |

### H3 Restore Original Audio After Refinement

`H3AVRestoreAudio`

Restore the first-pass audio stream into the refined joint latent after validating timing dimensions.

| Input (internal key) | Label | Type / default | Purpose |
| --- | --- | --- | --- |
| `refined` | Refined Joint Latent | LATENT | Second-pass output; its timing dimensions must match the original latent. |
| `original` | Original Joint Latent | LATENT | First-pass output providing the original audio stream to restore. |

| Output index | Label | Type | Meaning |
| --- | --- | --- | --- |
| 0 | Joint Video and Audio Latent | `LATENT` | Native H3 NestedTensor latent containing the video and audio streams. |

### H3 Audio-Video Continuation Guide

`H3AVContinuationGuide`

Use native MiniMaxH3AddGuide to anchor previous video and audio at frame zero; pass through on the first segment.

| Input (internal key) | Label | Type / default | Purpose |
| --- | --- | --- | --- |
| `loop_ctx` | Loop Context | MIE_LOOP_CTX | Connect the MieLoop v3 context for this run and segment index. |
| `positive` | Positive Conditioning | CONDITIONING | H3 positive conditioning; the continuation guide adds the paired context at frame zero after the first segment. |
| `latent` | Joint Video and Audio Latent | LATENT | Native H3 NestedTensor latent containing the video and audio streams. |
| `vae` | Video VAE | VAE | Native H3 video encoder/decoder required when using video guides or references. |
| `audio_vae` | Audio VAE | VAE | Native H3 audio encoder/decoder required when using audio guides or references. |
| `previous_frames` | Previous Video Context | IMAGE / optional | Final context frames from the preceding segment; empty on the first segment. |
| `previous_audio` | Previous Audio Context | AUDIO / optional | Audio aligned with the preceding video context; empty on the first segment. |

| Output index | Label | Type | Meaning |
| --- | --- | --- | --- |
| 0 | Positive Conditioning | `CONDITIONING` | H3 positive conditioning; the continuation guide adds the paired context at frame zero after the first segment. |

### H3 Trim and Write Synchronized Segment

`H3AVEncodeSegment`

Remove duplicate video/audio context together and save H.264 video, exact-length PCM audio and the next continuation tail.

| Input (internal key) | Label | Type / default | Purpose |
| --- | --- | --- | --- |
| `loop_ctx` | Loop Context | MIE_LOOP_CTX | Connect the MieLoop v3 context for this run and segment index. |
| `images` | Video Frames | IMAGE | Decoded RGB IMAGE batch shaped [frames, height, width, 3]; H.264 requires even width and height. |
| `audio` | Segment Audio | AUDIO | Decoded AUDIO with waveform [1, channels, samples] and a positive sample_rate, from the same segment as the video. |
| `expected_frames` | Planned Window Frames | INT / connection required | Connect the planner window_frames output; decoded frame count must match exactly. |
| `skip_frames` | Duplicate Context Frames to Remove | INT / connection required | Zero for the first segment; equals context frames thereafter. Remove the matching audio interval too. |
| `overlap_frames` | Continuation Context Frames | INT / `22` | Shared video/audio context. Must be 17k+5, such as 5, 22 or 39; use the same value in the planner and writer. |
| `crf` | H.264 Quality (CRF) | INT / `19` | Constant rate factor, 0–51. Smaller values improve quality and increase file size; default 19. |
| `preset` | H.264 Encoding Preset | medium/fast/veryfast/slow | medium is balanced; fast/veryfast reduce encoding time; slow spends more time on compression at the same CRF. |
| `reject_silence` | Reject Near-Silent Audio | BOOLEAN / `true` | Reject a segment below -65 dBFS RMS and preserve raw audio diagnostics; disable deliberately for silent scenes. |
| `continuation_images` | Video Frames for Next Continuation | IMAGE / optional | Optional context source with the same frame count. Useful for keeping first-pass frames while writing refined video. |
| `create_av_preview` | Create Segment Preview with Audio | BOOLEAN / `false` / optional | Create an extra preview MP4 with AAC audio; final output still uses the PCM masters. |

| Output index | Label | Type | Meaning |
| --- | --- | --- | --- |
| 0 | Loop Context | `MIE_LOOP_CTX` | Connect the MieLoop v3 context for this run and segment index. |
| 1 | Video-Only Segment Path | `STRING` | Absolute path to the video-only segment master; its paired PCM soundtrack is stored in a separate WAV. |
| 2 | Segment Diagnostics (JSON) | `STRING` | JSON containing frame counts, source audio duration error, output samples, peak, RMS and clipping fraction. |

### H3 Assemble Synchronized Final Video

`H3AVConcat`

Copy H.264 segments and encode the concatenated PCM audio to AAC once for the final MP4.

| Input (internal key) | Label | Type / default | Purpose |
| --- | --- | --- | --- |
| `loop_ctx` | Loop Context | MIE_LOOP_CTX | Connect the MieLoop v3 context for this run and segment index. |
| `done` | Loop Finished | BOOLEAN / connection required | Connect the MieLoop completion flag; concatenation runs only when true and all segments are committed. |

| Output index | Label | Type | Meaning |
| --- | --- | --- | --- |
| 0 | Final MP4 Path | `STRING` | Absolute path to the synchronized final H.264/AAC MP4. |

### H3 Independent Sampling Steps

`H3AVSamplingSteps`

Set each pass independently. The existing Goohaitool group controls whether the second pass runs. Closing it leaves the first-pass count unchanged and skips the latent upscaler and second sampler.

| Input (internal key) | Label | Type / default | Purpose |
| --- | --- | --- | --- |
| `一采步数` | First-Pass Steps | INT / `10` | Independent first-pass count, 1–10000. |
| `二采步数` | Second-Pass Steps | INT / `4` | Independent second-pass count, 1–10000; only executes when its Goohaitool group is enabled. |

| Output index | Label | Type | Meaning |
| --- | --- | --- | --- |
| 0 | First-Pass Steps | `INT` | First-pass count. |
| 1 | Second-Pass Steps | `INT` | Second-pass count. |
| 2 | Configured Step Sum | `INT` | Sum of the two configured counts, retained for socket compatibility; not the executed count when the second pass is bypassed. |

The frontend migrates old inline `[total, first]` values to independent `[first, second]` once, with a version marker. Old `14/14` becomes `14/1` and can run one pass. Legacy external widget links require manual migration because their arithmetic meaning differs. Raw API clients must replace `总步数` with `二采步数`; loading an old workflow in the frontend performs that change automatically.

### H3 Multi-Reference Video Conditioning

`H3AVReferenceListToVideo`

Load 1–9 IMAGE_LIST paths as separate native H3 references, with optional video and voice context.

| Input (internal key) | Label | Type / default | Purpose |
| --- | --- | --- | --- |
| `clip` | H3 Text Encoder | CLIP | The CLIP object loaded for the native H3 text/reference encoder. |
| `images` | Reference Image List | IMAGE_LIST | IMAGE_LIST dictionary with items: a list of 1–9 objects containing an existing local image path. This is not an IMAGE tensor batch. |
| `prompt` | Generation Prompt | STRING / connection required | Text conditioning prompt. The planner adds timing, joint AV and continuation instructions. |
| `width` | Video Width | INT / `1344` | Target video width in pixels; use multiples of 32 for this reference encoder. |
| `height` | Video Height | INT / `768` | Target video height in pixels; use multiples of 32 for this reference encoder. |
| `length` | Generation Window Frames | INT / `124` | Total generated frames, not seconds. For segmented continuation, connect the planner window_frames output. |
| `ref_image_size` | Reference Image Sizing | match/max / `max` | match uses the target dimensions; max delegates the maximum reference sizing strategy to the native H3 encoder. |
| `vae` | Video VAE | VAE / optional | Native H3 video encoder/decoder required when using video guides or references. |
| `audio_vae` | Audio VAE | VAE / optional | Native H3 audio encoder/decoder required when using audio guides or references. |
| `previous_frames` | Previous Video Context | IMAGE / optional | Final context frames from the preceding segment; empty on the first segment. |
| `previous_audio` | Previous Audio Context | AUDIO / optional | Audio aligned with the preceding video context; empty on the first segment. |
| `speaker_reference` | Speaker Voice Reference | AUDIO / optional | Up to the first three seconds of the first segment audio, used as a speaker identity reference in later segments. |
| `driving_audio` | Driving Audio Window | AUDIO / optional | Encodes the source recording into a frozen audio stream for audio-driven video. |
| `loop_ctx` | Reference Cache Loop Context | MIE_LOOP_CTX / optional | Connect the same loop context to share fixed-image VAE latents across segments in this run. |

| Output index | Label | Type | Meaning |
| --- | --- | --- | --- |
| 0 | Positive Conditioning | `CONDITIONING` | H3 positive conditioning; the continuation guide adds the paired context at frame zero after the first segment. |
| 1 | Initial Joint Latent | `LATENT` | Initial native H3 joint video/audio latent returned by the reference encoder. |

Fixed-image caching is automatic when `loop_ctx` is connected. A local VAE proxy reuses only native image encodings, keyed by the actual resized RGB pixels, shape, dtype and VAE/model/patch identity. Native resizing, reference order and `max` sizing are preserved. Changed images, resized inputs or VAE patches invalidate the corresponding entry. Each hit returns a copy on the VAE output device with the original dtype; no global Comfy model methods are patched.

The shared CPU LRU holds at most 256 MiB of latents and clears a run after its last reference call. It never retains model objects or fixed-image tensors in GPU memory. Without `loop_ctx`, the node uses normal uncached encoding. For nine square 3840×3840 inputs, current native `max` reduces each to 2048×2048; nine float32 H3 image latents occupy approximately 13.5 MiB. Image loading/resizing, Qwen text/visual conditioning, continuation video and all audio remain live for every segment. Cache hits do not reduce reference tokens in diffusion.

Logs show hit/miss counts and wall times for image reading, fixed-image VAE/cache, continuation video VAE, reference audio VAE, text/visual conditioning, driving audio VAE and native resizing/other work. The old combined reference-encoding duration is not entirely avoidable: for four segments the saving is approximately three sets of fixed-image VAE encoding, less cache lookup/copy time. Measure those new logs before estimating an end-to-end speedup. The original [native H3 reference encoder](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py) and [VAE interface](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/sd.py) remain responsible for model execution.

## Files and troubleshooting

Output directory: `ComfyUI/output/h3_av_sync/<run_id>/`, kept compatible with the original pack.

- `manifest.json`: committed segment order; `final.mp4`: final video; `*.log`: FFmpeg diagnostics.
- Keep at least 2 GiB free for writing. Assembly also requires the segment total size plus 1 GiB. Completed segments remain on disk after a low-space failure.
- For overwrite, manifest or index errors, check wiring and start a new loop run_id. Do not delete a manifest and continue using old segments. Segments for one run_id must be committed sequentially.

`segment_*.mp4` contains video only; paired `segment_*.wav` files contain 48 kHz stereo PCM. `tail_*.npz` stores continuation context, normally retaining only the latest tail. `speaker_reference.npz` stores the first voice reference. `plan_*.json` and `prompt_*.txt` record plans and actual prompts. `final_info.json` records final frame/audio information. Optional previews are `preview_*.mp4`.

Source audio/video duration mismatch above 0.05 s is rejected. There is no time stretching or silent fallback for significant mismatch; small tail differences within tolerance are padded/trimmed to exact frame boundaries and recorded. Silence rejection preserves `rejected_*.f32le` plus JSON diagnostics. Planner instructions preserve the script's speech language; switching interface language does not rewrite generated content.

## Development checks

```bash
python -m unittest discover -s tests -v
node --test tests/*.test.mjs
```

Tests cover bilingual metadata, stable socket compatibility, independent step migration, bounded cache reuse/invalidation, live per-segment conditions for all three audio modes, exception propagation and real FFmpeg timing/cancellation. Real model generation requires ComfyUI with H3 and MieLoop installed; CPU tests cannot measure GPU speedups or generated lip-sync quality.
