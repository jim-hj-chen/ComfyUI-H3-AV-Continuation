"""MiniMax H3: disk-backed AV continuation and frame-accurate final mux.

Core H3 handles generation. These nodes only plan, apply a native AV guide,
and write one segment at a time. No independent audio denoising or time stretch.
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import folder_paths

FPS = 24
OUTPUT_RATE = 48000
DELIMITER = '----------SEGMENT_SEPARATOR----------'
RUN_ID = re.compile(r'^[0-9a-f]{24}$')
CATEGORY = 'H3 AV Continuation/Planning'


def _ffmpeg():
    configured = os.environ.get('H3_FFMPEG')
    if configured:
        if not Path(configured).is_file():
            raise RuntimeError('H3_FFMPEG 指向的文件不存在。')
        return configured
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError('需要支持 libx264/AAC 的 FFmpeg：加入 PATH 或安装 imageio-ffmpeg。') from exc


def _context(ctx, terminal=False):
    if not isinstance(ctx, dict) or int(ctx.get('version', 0)) != 3:
        raise ValueError('需要 MieLoop v3 的 loop_ctx。')
    run_id = str(ctx.get('run_id', ''))
    if not RUN_ID.fullmatch(run_id):
        raise ValueError('无效的 MieLoop run_id。')
    index, count = int(ctx['index']), int(ctx['count'])
    if count < 1 or not 0 <= index <= (count if terminal else count - 1):
        raise ValueError('循环序号超出范围。')
    directory = Path(folder_paths.get_output_directory()) / 'h3_av_sync' / run_id
    return directory, index, count


def _space(directory, required=2 * 1024 ** 3):
    if shutil.disk_usage(directory).free < required:
        raise RuntimeError(f'磁盘空间不足：{directory}。已完成的分段会保留。')


def _atomic_json(path, data):
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     delete=False, suffix='.json.part') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        name = f.name
    os.replace(name, path)


def _run(cmd, log):
    with log.open('wb') as dst:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=dst)
    if result.returncode:
        tail = log.read_text(encoding='utf-8', errors='replace')[-3000:]
        raise RuntimeError(f'FFmpeg 失败，退出码 {result.returncode}：{tail}')


def _numpy(value):
    return value.detach().to('cpu').float().numpy()


def _audio_array(audio):
    if audio is None:
        raise ValueError('没有 AUDIO：必须连接同一段联合采样的音频解码。')
    arr = _numpy(audio['waveform'])
    rate = int(audio['sample_rate'])
    if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[1] not in (1, 2) or arr.shape[2] == 0 or rate < 1:
        raise ValueError('AUDIO 必须为非空 [1, 1或2声道, samples]，采样率须为正数。')
    if not np.isfinite(arr).all():
        raise ValueError('音频含 NaN/Inf，请检查采样和音频 VAE；拒绝写出坏音轨。')
    return np.ascontiguousarray(arr[0]), rate


def frame_plan(seconds, index, overlap):
    seconds = float(seconds)
    overlap = int(overlap)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('本段秒数必须为正有限数。')
    if overlap < 5 or overlap % 17 != 5:
        raise ValueError('续写上下文帧数必须为 17k+5，例如 5、22、39。')
    skip = overlap if index else 0
    requested = max(5, round(seconds * FPS) + skip)
    frames = requested + (5 - requested % 17) % 17
    if frames < 124 or frames > 362:
        raise ValueError(f'本段窗口 {frames} 帧超出 H3 常用范围 124–362；请调整秒数或上下文帧数。')
    if frames <= overlap:
        raise ValueError('本段窗口必须大于续写上下文。')
    return frames, skip


def shift_timecodes(text, offset):
    def replace(match):
        value = int(match[1]) * 60 + float(match[2]) + offset
        return f'{int(value // 60):02d}:{value % 60:06.3f}'
    return re.sub(r'(?<![\d:])(\d{2}):(\d{2}(?:\.\d+)?)(?![\d:])', replace, text)


def prepare_prompt(text, index, frames, skip, overlap):
    text = re.sub(r'final five frames', f'final {overlap} frames', text, flags=re.IGNORECASE)
    if index:
        text = shift_timecodes(text, skip / FPS)
        text = text.replace('Continuous natural speech begins at the first frame and runs through the last.',
                            'New dialogue starts immediately after the continuation context. Keep speech natural and synchronized.')
    header = (f'Generation window: {frames} frames at 24 fps ({frames / FPS:.6f} seconds). '
              'Generate the video and its audible Mandarin speech jointly. '
              'Whenever the presenter is on camera speaking, her lips match the exact spoken words. '
              'Every <d> dialogue line is audible speech, not subtitles. No music.\n')
    if index:
        header += (f'<Video 1> and its paired <Audio 1> are the preceding segment\'s final {overlap} frames '
                   'and synchronized soundtrack. Preserve the same presenter, voice, camera state, and action. '
                   f'The first {skip / FPS:.6f} seconds reproduce that preceding AV context and will be removed together. '
                   'Do not speak the new dialogue during that context and do not repeat the preceding dialogue. '
                   'Begin the new script after it. All shot timecodes below already include this offset.\n')
        header += ('<Audio 2> is a short voice reference from the first segment. '
                   'Use its speaker identity and timbre for (S1), while speaking only this segment\'s new dialogue. '
                   'Do not replay or quote the reference recording.\n')
    return header + text.strip()


class H3AVSegmentPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'loop_ctx': ('MIE_LOOP_CTX',),
            'prompts': ('STRING', {'forceInput': True}),
            'seconds': ('FLOAT', {'forceInput': True}),
            'overlap_frames': ('INT', {'default': 22, 'min': 5, 'max': 90, 'step': 17}),
            'base_seed': ('INT', {'default': 1087183784971949, 'min': 0, 'max': 0xffffffffffffffff}),
        }, 'optional': {'segment_prompt': ('STRING', {'forceInput': True})}}
    RETURN_TYPES = ('STRING', 'INT', 'IMAGE', 'AUDIO', 'INT', 'INT', 'AUDIO')
    RETURN_NAMES = ('prompt', 'window_frames', 'previous_frames', 'previous_audio', 'skip_frames', 'seed', 'speaker_reference')
    FUNCTION = 'plan'
    CATEGORY = CATEGORY

    def plan(self, loop_ctx, prompts, seconds, overlap_frames=22, base_seed=1087183784971949,
             segment_prompt=None):
        directory, index, count = _context(loop_ctx)
        scripts = [s.strip() for s in str(prompts).split(DELIMITER)]
        if len(scripts) != count or any(not s for s in scripts):
            raise ValueError(f'提示词段数 {len(scripts)} 与循环段数 {count} 不匹配，或有空段。')
        if segment_prompt is not None and str(segment_prompt).strip() != scripts[index]:
            raise ValueError('外部分割器选出的提示词与当前循环序号不匹配。请检查分隔符与取段连接。')
        frames, skip = frame_plan(seconds, index, overlap_frames)
        directory.mkdir(parents=True, exist_ok=True)
        _space(directory)
        previous_frames = previous_audio = speaker_reference = None
        if index:
            manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
            if len(manifest['segments']) != index or manifest['overlap_frames'] != overlap_frames:
                raise ValueError('上一段尚未提交，或上下文帧数在循环中发生了变化。')
            tail = directory / f'tail_{index - 1:05d}.npz'
            import torch
            with np.load(tail, allow_pickle=False) as data:
                if int(data['index']) != index - 1 or data['images'].shape[0] != overlap_frames:
                    raise ValueError('续写上下文与当前段不匹配。')
                previous_frames = torch.from_numpy(data['images'].copy())
                previous_audio = {'waveform': torch.from_numpy(data['audio'].copy()),
                                  'sample_rate': int(data['sample_rate'])}
            with np.load(directory / 'speaker_reference.npz', allow_pickle=False) as data:
                speaker_reference = {'waveform': torch.from_numpy(data['audio'].copy()),
                                     'sample_rate': int(data['sample_rate'])}
        seed = (int(base_seed) + index) % (1 << 64)
        prompt = prepare_prompt(scripts[index], index, frames, skip, overlap_frames)
        (directory / f'prompt_{index:05d}.txt').write_text(prompt, encoding='utf-8')
        _atomic_json(directory / f'plan_{index:05d}.json', {
            'index': index, 'count': count, 'seconds_requested': float(seconds),
            'window_frames': frames, 'skip_frames': skip, 'overlap_frames': overlap_frames,
            'fps': FPS, 'seed': seed, 'output_seconds': (frames - skip) / FPS})
        print(f'[H3AVSync] 段 {index + 1}/{count}: 窗口 {frames} 帧，去重 {skip} 帧，输出 {(frames - skip) / FPS:.6f}s')
        return prompt, frames, previous_frames, previous_audio, skip, seed, speaker_reference


class H3AVPickPrompt:
    """Accept either a ComfyUI output list or one wrapped Python list."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'prompts': ('STRING', {'forceInput': True}),
                             'count': ('INT', {'forceInput': True}),
                             'loop_ctx': ('MIE_LOOP_CTX',)}}
    INPUT_IS_LIST = True
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('segment_prompt',)
    FUNCTION = 'pick'
    CATEGORY = CATEGORY

    def pick(self, prompts, count, loop_ctx):
        if len(loop_ctx) != 1 or len(count) != 1:
            raise ValueError('提示词取段节点每次只接受一个循环上下文和一个段数。')
        _, index, expected = _context(loop_ctx[0])
        scripts = prompts[0] if len(prompts) == 1 and isinstance(prompts[0], (list, tuple)) else prompts
        if int(count[0]) != expected or len(scripts) != expected:
            raise ValueError(f'分割提示词数 {len(scripts)} / count {count[0]} 与循环段数 {expected} 不匹配。')
        if any(not isinstance(s, str) or not s.strip() for s in scripts):
            raise ValueError('分割结果含空段或非字符串。')
        return (scripts[index].strip(),)


def _av_streams(latent):
    samples = latent['samples']
    if not getattr(samples, 'is_nested', False):
        raise ValueError('需要 H3 音画联合 NestedTensor latent。')
    streams = samples.unbind()
    if len(streams) != 2 or streams[0].ndim != 5 or streams[1].ndim != 4:
        raise ValueError('H3 latent 必须包含一个5维视频流和一个4维音频流。')
    return streams


class H3AVFreezeAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'latent': ('LATENT',)}}
    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'freeze'
    CATEGORY = 'H3 AV Continuation/Refinement'

    def freeze(self, latent):
        import torch
        from comfy.nested_tensor import NestedTensor
        video, audio = _av_streams(latent)
        result = latent.copy()
        # Native per-stream masks: refine the video against the clean fixed audio.
        result['noise_mask'] = NestedTensor((
            torch.ones((1, 1, *video.shape[2:]), dtype=torch.float32),
            torch.zeros((1, 1, *audio.shape[2:]), dtype=torch.float32)))
        return (result,)


class H3AVRestoreAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'refined': ('LATENT',), 'original': ('LATENT',)}}
    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'restore'
    CATEGORY = 'H3 AV Continuation/Refinement'

    def restore(self, refined, original):
        from comfy.nested_tensor import NestedTensor
        video, audio = _av_streams(refined)
        source_video, source_audio = _av_streams(original)
        if video.shape[:3] != source_video.shape[:3] or audio.shape != source_audio.shape:
            raise ValueError('二采改变了音画时间维度；拒绝替换或拉伸音轨。')
        result = refined.copy()
        result.pop('noise_mask', None)
        result['samples'] = NestedTensor((video, source_audio))
        return (result,)


class H3AVContinuationGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'loop_ctx': ('MIE_LOOP_CTX',), 'positive': ('CONDITIONING',),
                             'latent': ('LATENT',), 'vae': ('VAE',), 'audio_vae': ('VAE',)},
                'optional': {'previous_frames': ('IMAGE',), 'previous_audio': ('AUDIO',)}}
    RETURN_TYPES = ('CONDITIONING',)
    RETURN_NAMES = ('positive',)
    FUNCTION = 'guide'
    CATEGORY = 'H3 AV Continuation/References'

    def guide(self, loop_ctx, positive, latent, vae, audio_vae, previous_frames=None, previous_audio=None):
        _, index, _ = _context(loop_ctx)
        if not index:
            return (positive,)
        if previous_frames is None or previous_audio is None:
            raise ValueError('后续段必须同时承接上一段画面及对应音频。')
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide
        # Native core node; do not synthesize a different masking implementation.
        result = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0,
                                         vae=vae, audio_vae=audio_vae,
                                         image=previous_frames, audio=previous_audio)
        if hasattr(result, 'result'):
            return (result.result[0],)
        return (result[0],)


def audio_stats(arr):
    peak = float(np.max(np.abs(arr)))
    rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
    dbfs = 20 * math.log10(max(rms, 1e-12))
    return {'peak': peak, 'rms_dbfs': dbfs, 'clipped_fraction': float(np.mean(np.abs(arr) >= 1))}


class H3AVEncodeSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'loop_ctx': ('MIE_LOOP_CTX',), 'images': ('IMAGE',), 'audio': ('AUDIO',),
            'expected_frames': ('INT', {'forceInput': True}),
            'skip_frames': ('INT', {'forceInput': True}),
            'overlap_frames': ('INT', {'default': 22, 'min': 5, 'max': 90, 'step': 17}),
            'crf': ('INT', {'default': 19, 'min': 0, 'max': 51}),
            'preset': (['medium', 'fast', 'veryfast', 'slow'],),
            'reject_silence': ('BOOLEAN', {'default': True}),
        }, 'optional': {
            'continuation_images': ('IMAGE',),
            'create_av_preview': ('BOOLEAN', {'default': False}),
        }}
    RETURN_TYPES = ('MIE_LOOP_CTX', 'STRING', 'STRING')
    RETURN_NAMES = ('loop_ctx', 'video_path', 'diagnostics')
    FUNCTION = 'encode'
    CATEGORY = 'H3 AV Continuation/Output'

    def encode(self, loop_ctx, images, audio, expected_frames, skip_frames,
               overlap_frames=22, crf=19, preset='medium', reject_silence=True,
               continuation_images=None, create_av_preview=False):
        directory, index, count = _context(loop_ctx)
        directory.mkdir(parents=True, exist_ok=True)
        _space(directory)
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError('图像必须为 RGB [frames, height, width, 3]。')
        frames, height, width, _ = map(int, images.shape)
        context_images = images if continuation_images is None else continuation_images
        if context_images.ndim != 4 or context_images.shape[0] != frames or context_images.shape[-1] != 3:
            raise ValueError('续写参考画面的帧数必须与输出画面一致。')
        if frames != expected_frames:
            raise ValueError(f'解码帧数 {frames} != 计划帧数 {expected_frames}。禁止按错误帧数裁剪音频。')
        if width % 2 or height % 2:
            raise ValueError('H.264 宽高必须为偶数。')
        if preset not in {'medium', 'fast', 'veryfast', 'slow'}:
            raise ValueError('无效 preset。')
        if overlap_frames < 5 or overlap_frames % 17 != 5 or frames <= overlap_frames:
            raise ValueError('无效的续写上下文帧数。')
        if int(skip_frames) != (overlap_frames if index else 0):
            raise ValueError('首段去重应为 0，后续段去重应等于上下文帧数。')
        arr, rate = _audio_array(audio)
        source_duration = arr.shape[-1] / rate
        mismatch = source_duration - frames / FPS
        if abs(mismatch) > 0.05:
            raise ValueError(f'本段音频 {source_duration:.6f}s 与画面 {frames / FPS:.6f}s 相差 {mismatch:.6f}s；拒绝用静音掩盖问题。')
        start = round(skip_frames * rate / FPS)
        end = min(arr.shape[-1], round(frames * rate / FPS))
        arr = arr[:, start:end]
        stats = audio_stats(arr)
        if reject_silence and stats['rms_dbfs'] < -65:
            # Save raw PCM and its format/levels before failing; no silent fallback.
            raw = directory / f'rejected_{index:05d}.f32le'
            raw.write_bytes(np.ascontiguousarray(arr.T, dtype='<f4').tobytes())
            _atomic_json(directory / f'rejected_{index:05d}.json', {
                'reason': 'near_silence', 'sample_rate': rate, 'channels': arr.shape[0], **stats})
            raise ValueError(f'第 {index + 1} 段近乎静音（{stats["rms_dbfs"]:.1f} dBFS），已保留诊断原始音频；请重新生成。')
        output_frames = frames - skip_frames
        target_samples = output_frames * (OUTPUT_RATE // FPS)
        manifest_path = directory / 'manifest.json'
        if index:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if (manifest['run_id'], manifest['count'], len(manifest['segments']),
                manifest['width'], manifest['height'], manifest['overlap_frames']) != (
                    loop_ctx['run_id'], count, index, width, height, overlap_frames):
                raise ValueError('分段顺序、分辨率或上下文设置不一致。')
        else:
            if manifest_path.exists():
                raise ValueError('当前 run_id 已有已提交分段，拒绝覆盖。请重新启动循环。')
            manifest = {'run_id': loop_ctx['run_id'], 'count': count, 'fps': FPS,
                        'width': width, 'height': height, 'overlap_frames': overlap_frames,
                        'audio_rate': OUTPUT_RATE, 'segments': []}
        video = directory / f'segment_{index:05d}.mp4'
        wav = directory / f'segment_{index:05d}.wav'
        tail = directory / f'tail_{index:05d}.npz'
        if video.exists() or wav.exists() or tail.exists():
            raise ValueError('发现未提交或已存在的分段文件；请检查日志，或重新启动循环。')
        vpart, apart, tpart = Path(str(video) + '.part'), Path(str(wav) + '.part'), Path(str(tail) + '.part')
        raw = directory / f'audio_{index:05d}.f32le.part'
        raw.write_bytes(np.ascontiguousarray(np.clip(arr, -1, 1).T, dtype='<f4').tobytes())
        try:
            _run([_ffmpeg(), '-hide_banner', '-loglevel', 'warning', '-y',
                  '-f', 'f32le', '-ar', str(rate), '-ac', str(arr.shape[0]), '-i', str(raw),
                  '-af', f'aresample={OUTPUT_RATE},apad=whole_len={target_samples},atrim=end_sample={target_samples},asetpts=PTS-STARTPTS',
                  '-ar', str(OUTPUT_RATE), '-ac', '2', '-c:a', 'pcm_s16le', '-f', 'wav', str(apart)],
                 directory / f'audio_{index:05d}.log')
            with wave.open(str(apart), 'rb') as reader:
                if (reader.getframerate(), reader.getnchannels(), reader.getnframes()) != (OUTPUT_RATE, 2, target_samples):
                    raise RuntimeError('FFmpeg 写出的 PCM 音轨长度不符合帧数。')
            cmd = [_ffmpeg(), '-hide_banner', '-loglevel', 'warning', '-y',
                   '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                   '-framerate', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                   '-preset', preset, '-crf', str(crf), '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', '-f', 'mp4', str(vpart)]
            log_path = directory / f'video_{index:05d}.log'
            with log_path.open('wb') as log:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                try:
                    for i in range(skip_frames, frames):
                        frame = _numpy(images[i])
                        if not np.isfinite(frame).all():
                            raise ValueError(f'第 {i} 帧含 NaN/Inf。')
                        piece = np.ascontiguousarray(np.clip(frame * 255 + 0.5, 0, 255).astype(np.uint8))
                        proc.stdin.write(piece.tobytes())
                    proc.stdin.close()
                    code = proc.wait()
                except BaseException:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait()
                    if isinstance(sys_exc := sys.exc_info()[1], BrokenPipeError):
                        raise RuntimeError(log_path.read_text(encoding='utf-8', errors='replace')[-3000:]) from sys_exc
                    raise
            if code or not vpart.is_file() or vpart.stat().st_size == 0:
                raise RuntimeError(log_path.read_text(encoding='utf-8', errors='replace')[-3000:])
            # Tail timing follows video time, never the end of a possibly padded audio track.
            full_arr, _ = _audio_array(audio)
            a0, a1 = round((frames - overlap_frames) * rate / FPS), round(frames * rate / FPS)
            tail_audio = full_arr[:, a0:min(a1, full_arr.shape[-1])]
            if tail_audio.shape[-1] < a1 - a0:
                tail_audio = np.pad(tail_audio, ((0, 0), (0, a1 - a0 - tail_audio.shape[-1])))
            with tpart.open('wb') as dst:
                tail_images = _numpy(context_images[-overlap_frames:])
                if not np.isfinite(tail_images).all():
                    raise ValueError('续写参考画面含 NaN/Inf。')
                np.savez(dst, images=tail_images, audio=tail_audio[None],
                         sample_rate=np.array(rate), index=np.array(index))
            if index == 0:
                voice_path = directory / 'speaker_reference.npz'
                voice_part = Path(str(voice_path) + '.part')
                try:
                    with voice_part.open('wb') as dst:
                        np.savez(dst, audio=full_arr[None, :, :3 * rate], sample_rate=np.array(rate))
                    os.replace(voice_part, voice_path)
                finally:
                    voice_part.unlink(missing_ok=True)
            os.replace(vpart, video)
            os.replace(apart, wav)
            os.replace(tpart, tail)
            preview = directory / f'preview_{index:05d}.mp4'
            if create_av_preview:
                _run([_ffmpeg(), '-hide_banner', '-loglevel', 'warning', '-y',
                      '-i', str(video), '-i', str(wav), '-map', '0:v:0', '-map', '1:a:0',
                      '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                      '-t', f'{output_frames / FPS:.12f}', '-movflags', '+faststart', str(preview)],
                     directory / f'preview_{index:05d}.log')
            entry = {'index': index, 'video': video.name, 'audio': wav.name,
                     'window_frames': frames, 'skip_frames': skip_frames,
                     'output_frames': output_frames, 'audio_samples': target_samples,
                     'source_audio_seconds': source_duration, 'source_duration_error_seconds': mismatch,
                     'continuation_width': int(context_images.shape[2]),
                     'continuation_height': int(context_images.shape[1]),
                     'preview': preview.name if create_av_preview else None,
                     **stats}
            manifest['segments'].append(entry)
            _atomic_json(manifest_path, manifest)
            # Only retain the context required by the next segment, not every image tail.
            if index:
                (directory / f'tail_{index - 1:05d}.npz').unlink(missing_ok=True)
            print(f'[H3AVSync] 段 {index + 1}/{count}: {output_frames} 帧，{target_samples} PCM samples，{stats["rms_dbfs"]:.1f} dBFS')
            return loop_ctx, str(video), json.dumps(entry, ensure_ascii=False)
        finally:
            for path in (raw, vpart, apart, tpart):
                path.unlink(missing_ok=True)


def _concat_quote(path):
    return "'" + Path(path).resolve().as_posix().replace("'", "'\\''") + "'"


def concat_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    segments = manifest['segments']
    if not segments or len(segments) != manifest['count']:
        raise ValueError('分段未全部完成，拒绝输出不完整成片。')
    video_lines, audio_lines = ['ffconcat version 1.0'], ['ffconcat version 1.0']
    total_frames, total_bytes = 0, 0
    for index, item in enumerate(segments):
        if item['index'] != index:
            raise ValueError('分段清单顺序错误。')
        # Canonical names prevent manifest content from directing reads elsewhere.
        v = directory / f'segment_{index:05d}.mp4'
        a = directory / f'segment_{index:05d}.wav'
        if item['video'] != v.name or item['audio'] != a.name or not v.is_file() or not a.is_file():
            raise ValueError(f'分段文件缺失：{index}')
        samples = item['output_frames'] * (OUTPUT_RATE // FPS)
        with wave.open(str(a), 'rb') as reader:
            if (reader.getframerate(), reader.getnchannels(), reader.getnframes()) != (OUTPUT_RATE, 2, samples):
                raise ValueError(f'第 {index + 1} 段音轨长度已改变。')
        video_lines += [f'file {_concat_quote(v)}', f'duration {item["output_frames"] / FPS:.12f}']
        audio_lines += [f'file {_concat_quote(a)}']
        total_frames += item['output_frames']
        total_bytes += v.stat().st_size + a.stat().st_size
    _space(directory, total_bytes + 1024 ** 3)
    vl, al = directory / 'video.ffconcat', directory / 'audio.ffconcat'
    vl.write_text('\n'.join(video_lines) + '\n', encoding='utf-8')
    al.write_text('\n'.join(audio_lines) + '\n', encoding='utf-8')
    output = directory / 'final.mp4'
    partial = directory / 'final.mp4.part'
    # Copy the H.264 video; encode the continuous PCM audio once, eliminating
    # segment-specific AAC priming/padding from the concatenation time axis.
    cmd = [_ffmpeg(), '-hide_banner', '-loglevel', 'warning', '-y',
           '-f', 'concat', '-safe', '0', '-i', str(vl),
           '-f', 'concat', '-safe', '0', '-i', str(al),
           '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac',
           '-b:a', '192k', '-ar', str(OUTPUT_RATE), '-ac', '2',
           '-t', f'{total_frames / FPS:.12f}', '-movflags', '+faststart', '-f', 'mp4', str(partial)]
    try:
        _run(cmd, directory / 'concat.log')
        if not partial.exists() or partial.stat().st_size == 0:
            raise RuntimeError('成片文件为空。')
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    _atomic_json(directory / 'final_info.json', {'video': str(output), 'frames': total_frames,
                 'duration_seconds': total_frames / FPS, 'pcm_samples': total_frames * 2000,
                 'audio_encoded_once': True, 'video_reencoded': False})
    return output


class H3AVConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'loop_ctx': ('MIE_LOOP_CTX',), 'done': ('BOOLEAN', {'forceInput': True})}}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('video_path',)
    FUNCTION = 'concat'
    CATEGORY = 'H3 AV Continuation/Output'
    OUTPUT_NODE = True

    def concat(self, loop_ctx, done):
        if not done:
            return ('',)
        directory, _, count = _context(loop_ctx, terminal=True)
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['run_id'] != loop_ctx['run_id'] or manifest['count'] != count:
            raise ValueError('分段清单与循环上下文不匹配。')
        output = concat_run(directory)
        return {'ui': {'text': [str(output)], 'gifs': [{'filename': output.name,
            'subfolder': f'h3_av_sync/{loop_ctx["run_id"]}', 'type': 'output', 'format': 'video/h264-mp4'}]},
            'result': (str(output),)}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in
    (H3AVSegmentPlan, H3AVPickPrompt, H3AVFreezeAudio, H3AVRestoreAudio,
     H3AVContinuationGuide, H3AVEncodeSegment, H3AVConcat)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'H3AVSegmentPlan': 'H3 分段计划与上一段音画上下文',
    'H3AVPickPrompt': 'H3 按循环序号取一段提示词（支持列表输出）',
    'H3AVFreezeAudio': 'H3 二采冻结音频（只重绘画面）',
    'H3AVRestoreAudio': 'H3 二采后恢复原音频（保持时间维度）',
    'H3AVContinuationGuide': 'H3 原生音画联合续写锚定',
    'H3AVEncodeSegment': 'H3 按帧裁剪与分段落盘（PCM音轨）',
    'H3AVConcat': 'H3 拼接画面与连续音轨（仅编码一次AAC）',
}

# Operation helpers; existing AV generation and disk logic stays intact.
from .workflow_tools import NODE_CLASS_MAPPINGS as _UI_NODES, NODE_DISPLAY_NAME_MAPPINGS as _UI_NAMES
NODE_CLASS_MAPPINGS.update(_UI_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(_UI_NAMES)


# Keep stable node IDs and socket keys for existing workflows.
from .node_metadata import apply_metadata
apply_metadata(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
WEB_DIRECTORY = "./web"
