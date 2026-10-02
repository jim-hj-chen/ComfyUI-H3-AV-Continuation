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
