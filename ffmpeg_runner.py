"""Bounded, interruptible FFmpeg calls, including blocked rawvideo pipes."""
import os
import subprocess
import threading
import time


def _interrupt():
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def run_ffmpeg(cmd, log, frames=None, timeout=None):
    timeout = float(os.environ.get('H3_FFMPEG_TIMEOUT_SECONDS', '600')) if timeout is None else float(timeout)
    if not 0 < timeout < float('inf'):
        raise ValueError('H3_FFMPEG_TIMEOUT_SECONDS 必须为正有限秒数。')
    errors = []
    worker = None
    started = time.monotonic()
    print(f'[H3AVSync] FFmpeg 开始：{log.name}，超时 {timeout:g}s。', flush=True)
    with log.open('wb') as dst:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if frames is not None else subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=dst)

        def write_frames():
            try:
                for frame in frames:
                    proc.stdin.write(frame)
                proc.stdin.close()
            except BaseException as exc:
                errors.append(exc)

        try:
            if frames is not None:
                worker = threading.Thread(target=write_frames, daemon=True, name='h3-ffmpeg-input')
                worker.start()
            while proc.poll() is None:
                _interrupt()
                if errors:
                    raise errors[0]
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f'FFmpeg 超过 {timeout:g}s，已停止。日志：{log}')
                try:
                    proc.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    pass
            if worker is not None:
                worker.join(timeout=5)
                if worker.is_alive():
                    raise RuntimeError('FFmpeg 已退出但输入帧准备仍未返回，请检查 GPU/驱动状态。')
            if errors:
                raise errors[0]
            if proc.returncode:
                raise RuntimeError(f'FFmpeg 退出码 {proc.returncode}：{log}')
        except BaseException as exc:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            if worker is not None:
                worker.join(timeout=5)
            if isinstance(exc, (BrokenPipeError, RuntimeError)):
                dst.flush()
                tail = log.read_text(encoding='utf-8', errors='replace')[-3000:]
                raise RuntimeError(f'{exc}\n{tail}') from exc
            raise
    print(f'[H3AVSync] FFmpeg 完成：{log.name}，{time.monotonic() - started:.1f}s。', flush=True)
