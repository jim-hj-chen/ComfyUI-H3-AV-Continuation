"""Audio routing without changing native H3 sampling or source speech timing."""
import math

MODES = ('generated', 'voice_reference', 'audio_drive')


def validate_audio(audio):
    if not isinstance(audio, dict) or 'waveform' not in audio:
        raise ValueError('音频模式已开启，请连接加载音频节点。')
    wave, rate = audio['waveform'], int(audio.get('sample_rate', 0))
    if wave.ndim != 3 or wave.shape[0] != 1 or wave.shape[1] not in (1, 2) or wave.shape[-1] < 1 or rate < 1:
        raise ValueError('音频需要 [1, 1或2声道, samples] 和正采样率。')
    import torch
    if not torch.isfinite(wave).all():
        raise ValueError('输入音频含 NaN/Inf。')
    return wave, rate


def driving_window(audio, offset_frames, new_frames, window_frames, skip_frames):
    """Slice the continuous recording, padding only discarded H3 grid surplus."""
    import torch
    wave, rate = validate_audio(audio)
    start = round(offset_frames * rate / 24)
    end = round((offset_frames + new_frames) * rate / 24)
    # At most half a video frame of final rounding, never missing dialogue.
    if end > wave.shape[-1] + math.ceil(rate / 48):
        raise ValueError(f'驱动音频不足：此段需要到 {end / rate:.3f}s，录音只有 {wave.shape[-1] / rate:.3f}s。请调整各段时长。')
    context_start = round((offset_frames - skip_frames) * rate / 24)
    if context_start < 0:
        raise ValueError('驱动音频上下文起点无效。')
    samples = round(window_frames * rate / 24)
    selected = wave[..., context_start:min(end, wave.shape[-1])].detach().to(device='cpu', dtype=torch.float32)
    if selected.shape[-1] > samples:
        selected = selected[..., :samples]
    return {'waveform': torch.nn.functional.pad(selected, (0, samples - selected.shape[-1])), 'sample_rate': rate}


class H3AVAudioModes:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'voice_reference': ('BOOLEAN', {'default': False}),
            'audio_drive': ('BOOLEAN', {'default': False}),
        }, 'optional': {'audio': ('AUDIO', {'lazy': True})}}

    RETURN_TYPES = ('STRING', 'AUDIO')
    RETURN_NAMES = ('audio_mode', 'source_audio')
    FUNCTION = 'route'
    CATEGORY = 'H3 AV Continuation/Controls'

    @staticmethod
    def _mode(voice_reference, audio_drive):
        if voice_reference and audio_drive:
            raise ValueError('音色参考和音频驱动不能同时开启。请关闭其中一个开关。')
        return 'voice_reference' if voice_reference else 'audio_drive' if audio_drive else 'generated'

    def check_lazy_status(self, voice_reference, audio_drive, audio=None):
        mode = self._mode(voice_reference, audio_drive)
        return ['audio'] if mode != 'generated' and audio is None else []

    def route(self, voice_reference=False, audio_drive=False, audio=None):
        mode = self._mode(voice_reference, audio_drive)
        if mode != 'generated':
            validate_audio(audio)
        return mode, audio if mode != 'generated' else None


class H3AVAudioOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'audio_mode': ('STRING', {'forceInput': True})}, 'optional': {
            'generated_audio': ('AUDIO', {'lazy': True}),
            'driving_audio': ('AUDIO', {'lazy': True}),
        }}

    RETURN_TYPES = ('AUDIO',)
    RETURN_NAMES = ('audio',)
    FUNCTION = 'select'
    CATEGORY = 'H3 AV Continuation/Output'

    def check_lazy_status(self, audio_mode, generated_audio=None, driving_audio=None):
        if audio_mode not in MODES:
            raise ValueError('无效音频模式。')
        key = 'driving_audio' if audio_mode == 'audio_drive' else 'generated_audio'
        return [key] if (driving_audio if key == 'driving_audio' else generated_audio) is None else []

    def select(self, audio_mode, generated_audio=None, driving_audio=None):
        if audio_mode not in MODES:
            raise ValueError('无效音频模式。')
        selected = driving_audio if audio_mode == 'audio_drive' else generated_audio
        validate_audio(selected)
        return (selected,)


class H3AVOptionalAudioRefine:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'enabled': ('BOOLEAN', {'default': False}),
            'audio_mode': ('STRING', {'forceInput': True}),
            'original': ('LATENT', {'lazy': True}),
            'refined': ('LATENT', {'lazy': True}),
        }}

    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('latent',)
    FUNCTION = 'select'
    CATEGORY = 'H3 AV Continuation/Refinement'

    def check_lazy_status(self, enabled, audio_mode, original=None, refined=None):
        if audio_mode not in MODES:
            raise ValueError('无效音频模式。')
        key = 'refined' if enabled and audio_mode != 'audio_drive' else 'original'
        return [key] if (refined if key == 'refined' else original) is None else []

    def select(self, enabled=False, audio_mode='generated', original=None, refined=None):
        if audio_mode not in MODES:
            raise ValueError('无效音频模式。')
        result = refined if enabled and audio_mode != 'audio_drive' else original
        if result is None:
            raise ValueError('所选音频细化分支没有 LATENT。')
        return (result,)


class H3AVOptionalVideoRefine:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'enabled': ('BOOLEAN', {'default': False}),
                             'original': ('LATENT', {'lazy': True}),
                             'refined': ('LATENT', {'lazy': True})}}

    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('latent',)
    FUNCTION = 'select'
    CATEGORY = 'H3 AV Continuation/Controls'

    def check_lazy_status(self, enabled, original=None, refined=None):
        key = 'refined' if enabled else 'original'
        return [key] if (refined if enabled else original) is None else []

    def select(self, enabled=False, original=None, refined=None):
        selected = refined if enabled else original
        if selected is None:
            raise ValueError('所选分支没有输入。')
        return (selected,)


class H3AVOptionalUpscale(H3AVOptionalVideoRefine):
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'enabled': ('BOOLEAN', {'default': False}),
                             'original': ('IMAGE', {'lazy': True}),
                             'refined': ('IMAGE', {'lazy': True})}}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('images',)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    H3AVAudioModes, H3AVAudioOutput, H3AVOptionalAudioRefine, H3AVOptionalVideoRefine, H3AVOptionalUpscale)}
