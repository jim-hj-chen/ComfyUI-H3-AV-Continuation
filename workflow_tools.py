"""Small UI helpers for the H3 workflow; native H3 still performs conditioning."""
from pathlib import Path
from .sampling_steps import H3AVSamplingSteps
from .reference_cache import CACHE, VideoVAEProxy, TimedProxy, image_shapes, loop_scope, new_timings


class H3AVReferenceListToVideo:
    """One IMAGE_LIST input, without tensor batching or placeholder references.

    H3 reads img[:1] at each reference input. Concatenating images into one IMAGE
    would silently retain only its first item, so feed an ordered dict instead.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'clip': ('CLIP',),
            'images': ('IMAGE_LIST',),
            'prompt': ('STRING', {'forceInput': True}),
            'width': ('INT', {'default': 1344, 'min': 32, 'max': 16384, 'step': 32}),
            'height': ('INT', {'default': 768, 'min': 32, 'max': 16384, 'step': 32}),
            'length': ('INT', {'default': 124, 'min': 5, 'max': 3600, 'step': 17}),
            'ref_image_size': (['match', 'max'], {'default': 'max'}),
        }, 'optional': {
            'vae': ('VAE',),
            'audio_vae': ('VAE',),
            'previous_frames': ('IMAGE',),
            'previous_audio': ('AUDIO',),
            'speaker_reference': ('AUDIO',),
            'driving_audio': ('AUDIO',),
            'loop_ctx': ('MIE_LOOP_CTX',),
        }}

    RETURN_TYPES = ('CONDITIONING', 'LATENT')
    RETURN_NAMES = ('positive', 'LATENT')
    FUNCTION = 'encode'
    CATEGORY = 'H3 AV Continuation/References'

    @staticmethod
    def load_references(images):
        import numpy as np
        import torch
        from PIL import Image, ImageOps
        if not isinstance(images, dict) or not isinstance(images.get('items'), (list, tuple)):
            raise ValueError('需要豆包上传节点输出的 IMAGE_LIST（items 图片列表）。')
        items = images['items']
        if not 1 <= len(items) <= 9:
            raise ValueError(f'参考图片需为 1–9 张，当前为 {len(items)} 张。')
        refs = {}
        for index, item in enumerate(items):
            path = str(item.get('path', '')).strip() if isinstance(item, dict) else ''
            if not path or not Path(path).is_file():
                raise ValueError(f'第 {index + 1} 张参考图路径无效：{path}')
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert('RGB')
                array = np.asarray(rgb, dtype=np.float32).copy() / 255.0
            refs[f'ref_image_{index}'] = torch.from_numpy(array)[None, ...]
        return refs

    def encode(self, clip, images, prompt, width, height, length,
               ref_image_size='max', vae=None, audio_vae=None,
               previous_frames=None, previous_audio=None, speaker_reference=None, driving_audio=None, loop_ctx=None):
        import time
        from importlib import import_module
        native = import_module('comfy_extras.nodes_minimax_h3')
        started = time.monotonic()
        refs = self.load_references(images)
        loaded = time.monotonic() - started
        scope, final_segment = loop_scope(loop_ctx)
        timings = new_timings()
        video_proxy = VideoVAEProxy(vae, scope, image_shapes(refs, width, height, ref_image_size,
            getattr(native, 'REF_IMAGE_SHORT_EDGE', 2048), getattr(native, 'CANVAS_MULTIPLE', 32)), timings) if vae is not None else None
        audio_proxy = TimedProxy(audio_vae, timings, {'encode': 'reference_audio'}) if audio_vae is not None else None
        clip_proxy = TimedProxy(clip, timings, {'tokenize': 'text_visual', 'encode_from_tokens_scheduled': 'text_visual'})
        print(f'[H3AVSync] 参考条件编码开始：{len(refs)} 张图片，尺寸策略 {ref_image_size}。', flush=True)
        try:
            result = native.MiniMaxH3ReferenceToVideo.execute(
                clip=clip_proxy, prompt=prompt, width=width, height=height, length=length,
                ref_image_size=ref_image_size, vae=video_proxy, audio_vae=audio_proxy,
                ref_images=refs,
                ref_videos={'ref_video_0': previous_frames} if previous_frames is not None else {},
                ref_video_audios={'ref_video_audio_0': previous_audio} if previous_audio is not None else {},
                ref_audios={'ref_audio_0': speaker_reference} if speaker_reference is not None else {},
            )
        finally:
            # The last reference call has already retrieved copies. Dropping CPU
            # entries cannot alter any conditioning tensors still in use.
            if final_segment:
                CACHE.release(scope)
        # V3 NodeOutput is tuple-like through .result; accept old tuple returns too.
        positive, latent = tuple(result.result) if hasattr(result, 'result') else tuple(result)
        if driving_audio is not None:
            if audio_vae is None:
                raise ValueError('音频驱动需要 H3 音频 VAE。')
            from comfy_extras.nodes_minimax_h3 import _encode_ref_audio
            from comfy.nested_tensor import NestedTensor
            import torch
            import torch.nn.functional as F
            driving_started = time.monotonic()
            encoded, _ = _encode_ref_audio(audio_vae, driving_audio)
            timings['driving_audio'] = time.monotonic() - driving_started
            video, empty_audio = latent['samples'].unbind()
            if encoded.shape[:-1] != empty_audio.shape[:-1] or abs(encoded.shape[-1] - empty_audio.shape[-1]) > 2:
                raise ValueError('驱动音频 VAE 的时间维度与 H3 窗口不匹配。')
            encoded = encoded[..., :empty_audio.shape[-1]]
            encoded = F.pad(encoded, (0, empty_audio.shape[-1] - encoded.shape[-1]))
            latent = latent.copy()
            latent['samples'] = NestedTensor((video, encoded.to(empty_audio)))
            latent['noise_mask'] = NestedTensor((
                torch.ones((1, 1, *video.shape[2:]), dtype=torch.float32),
                torch.zeros((1, 1, *empty_audio.shape[2:]), dtype=torch.float32)))
        if video_proxy is not None:
            detail = f'；{video_proxy.reason}' if video_proxy.reason else ''
            print(f'[H3AVSync] 固定图缓存：命中 {video_proxy.hits}/{len(refs)}，实际编码 {video_proxy.misses} 张；CPU缓存 {CACHE.usage() / 1048576:.1f} MiB{detail}。', flush=True)
        elapsed = time.monotonic() - started
        other = max(0.0, elapsed - loaded - sum(timings.values()))
        print(f'[H3AVSync] 条件分项：读取图片 {loaded:.1f}s；固定图VAE/缓存 {timings["fixed_images"]:.1f}s；续写视频VAE {timings["previous_video"]:.1f}s；参考音频VAE {timings["reference_audio"]:.1f}s；文本/视觉条件 {timings["text_visual"]:.1f}s；驱动音频VAE {timings["driving_audio"]:.1f}s；原生缩放/其他 {other:.1f}s。', flush=True)
        print(f'[H3AVSync] 参考条件编码完成：{elapsed:.1f}s，开始采样。', flush=True)
        return positive, latent


NODE_CLASS_MAPPINGS = {
    'H3AVSamplingSteps': H3AVSamplingSteps,
    'H3AVReferenceListToVideo': H3AVReferenceListToVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'H3AVSamplingSteps': '采样步数 · 一采 / 二采',
    'H3AVReferenceListToVideo': 'H3 参考图自动适配',
}
