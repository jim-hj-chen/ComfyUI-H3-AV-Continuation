"""Small UI helpers for the H3 workflow; native H3 still performs conditioning."""
from pathlib import Path


class H3AVSamplingSteps:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            '总步数': ('INT', {'default': 16, 'min': 2, 'max': 10000}),
            '一采步数': ('INT', {'default': 10, 'min': 1, 'max': 9999}),
        }}

    RETURN_TYPES = ('INT', 'INT', 'INT')
    RETURN_NAMES = ('一采步数', '二采步数', '总步数')
    FUNCTION = 'split'
    CATEGORY = 'H3 AV Continuation/Controls'

    def split(self, 总步数=16, 一采步数=10):
        total_steps, first_steps = 总步数, 一采步数
        if not 1 <= int(first_steps) < int(total_steps) <= 10000:
            raise ValueError('步数需要满足：1 ≤ 一采步数 < 总步数 ≤ 10000。')
        return int(first_steps), int(total_steps) - int(first_steps), int(total_steps)


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
            'ref_image_size': (['match', 'max'], {'default': 'match'}),
        }, 'optional': {
            'vae': ('VAE',),
            'audio_vae': ('VAE',),
            'previous_frames': ('IMAGE',),
            'previous_audio': ('AUDIO',),
            'speaker_reference': ('AUDIO',),
            'driving_audio': ('AUDIO',),
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
               ref_image_size='match', vae=None, audio_vae=None,
               previous_frames=None, previous_audio=None, speaker_reference=None, driving_audio=None):
        import time
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        refs = self.load_references(images)
        started = time.monotonic()
        print(f'[H3AVSync] 参考条件编码开始：{len(refs)} 张图片，尺寸策略 {ref_image_size}。', flush=True)
        result = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, prompt=prompt, width=width, height=height, length=length,
            ref_image_size=ref_image_size, vae=vae, audio_vae=audio_vae,
            ref_images=refs,
            ref_videos={'ref_video_0': previous_frames} if previous_frames is not None else {},
            ref_video_audios={'ref_video_audio_0': previous_audio} if previous_audio is not None else {},
            ref_audios={'ref_audio_0': speaker_reference} if speaker_reference is not None else {},
        )
        # V3 NodeOutput is tuple-like through .result; accept old tuple returns too.
        positive, latent = tuple(result.result) if hasattr(result, 'result') else tuple(result)
        if driving_audio is not None:
            if audio_vae is None:
                raise ValueError('音频驱动需要 H3 音频 VAE。')
            from comfy_extras.nodes_minimax_h3 import _encode_ref_audio
            from comfy.nested_tensor import NestedTensor
            import torch
            import torch.nn.functional as F
            encoded, _ = _encode_ref_audio(audio_vae, driving_audio)
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
        print(f'[H3AVSync] 参考条件编码完成：{time.monotonic() - started:.1f}s，开始采样。', flush=True)
        return positive, latent


NODE_CLASS_MAPPINGS = {
    'H3AVSamplingSteps': H3AVSamplingSteps,
    'H3AVReferenceListToVideo': H3AVReferenceListToVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'H3AVSamplingSteps': '采样步数 · 总数 / 一采 / 二采',
    'H3AVReferenceListToVideo': 'H3 参考图自动适配',
}
