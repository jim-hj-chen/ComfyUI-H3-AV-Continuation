"""Run-scoped CPU caching of native H3 fixed-image VAE latents.

The native encoder still resizes images and builds all conditioning. A local
proxy intercepts only its leading single-image VAE calls; video/audio and CLIP
always execute normally. No global VAE methods or Comfy model state are patched.
"""
from collections import OrderedDict, defaultdict
import hashlib
import math
import threading
import time
import weakref


class ReferenceCache:
    def __init__(self, max_bytes=256 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.vaes = {}
        self.lock = threading.RLock()

    def _remove(self, key):
        _, size = self.entries.pop(key)
        self.bytes -= size

    def _forget_vae(self, identity, reference):
        with self.lock:
            if self.vaes.get(identity) is not reference:
                return
            self.vaes.pop(identity, None)
            for key in list(self.entries):
                if key[1][0] == identity:
                    self._remove(key)

    def stamp(self, vae):
        identity = id(vae)
        with self.lock:
            reference = self.vaes.get(identity)
            if reference is None or reference() is not vae:
                reference = weakref.ref(vae, lambda ref: self._forget_vae(identity, ref))
                self.vaes[identity] = reference
        patcher = getattr(vae, 'patcher', None)
        output_dtype = getattr(vae, 'vae_output_dtype', lambda: None)()
        return (identity, id(getattr(vae, 'first_stage_model', None)),
                id(patcher), str(getattr(patcher, 'patches_uuid', None)),
                str(getattr(vae, 'vae_dtype', None)), str(output_dtype))

    def get(self, key, device):
        with self.lock:
            item = self.entries.get(key)
            if item is None:
                return None
            self.entries.move_to_end(key)
            # Never expose the stored tensor to downstream in-place edits.
            value = item[0].clone()
        return value.to(device=device)

    def put(self, key, value):
        if value.numel() * value.element_size() > self.max_bytes:
            return
        copy = value.detach().to(device='cpu').clone()
        size = copy.numel() * copy.element_size()
        if size > self.max_bytes:
            return
        with self.lock:
            if key in self.entries:
                self._remove(key)
            while self.entries and self.bytes + size > self.max_bytes:
                self._remove(next(iter(self.entries)))
            self.entries[key] = (copy, size)
            self.bytes += size

    def release(self, scope):
        with self.lock:
            for key in list(self.entries):
                if key[0] == scope:
                    self._remove(key)

    def usage(self):
        with self.lock:
            return self.bytes


CACHE = ReferenceCache()


def loop_scope(loop_ctx):
    if not isinstance(loop_ctx, dict):
        return None, False
    run_id = loop_ctx.get('run_id')
    index, count = loop_ctx.get('index'), loop_ctx.get('count')
    if not isinstance(run_id, str) or not run_id or not isinstance(index, int) or not isinstance(count, int) or not 0 <= index < count:
        return None, False
    return run_id, index == count - 1


def image_shapes(refs, width, height, sizing, short_edge=2048, multiple=32):
    """Guard current native image call order; the native code does the resizing."""
    shapes = []
    for image in refs.values():
        if image is None:
            continue
        h, w = image.shape[1:3]
        scale = min(1.0, math.sqrt(width * height / (w * h))) if sizing == 'match' else min(1.0, short_edge / min(w, h))
        tw = max(multiple, round(w * scale / multiple) * multiple)
        th = max(multiple, round(h * scale / multiple) * multiple)
        shapes.append((1, th, tw, 3))
    return shapes


def pixel_key(pixels):
    cpu = pixels.detach().to(device='cpu').contiguous()
    # Native image resizing receives float32 RGB. Hash the actual pixels rather
    # than paths or modification timestamps, including when files are replaced.
    array = cpu.numpy()
    digest = hashlib.sha256(memoryview(array).cast('B')).digest()
    return tuple(cpu.shape), str(cpu.dtype), digest


class VideoVAEProxy:
    def __init__(self, vae, scope, shapes, timings, cache=CACHE):
        self.vae, self.scope, self.shapes = vae, scope, shapes
        self.timings, self.cache = timings, cache
        self.calls = self.hits = self.misses = 0
        self.enabled = scope is not None
        self.reason = '' if self.enabled else '未连接循环上下文'

    def __getattr__(self, name):
        return getattr(self.vae, name)

    def encode(self, pixels, *args, **kwargs):
        call = self.calls
        self.calls += 1
        fixed = call < len(self.shapes)
        label = 'fixed_images' if fixed else 'previous_video'
        started = time.monotonic()
        key = None
        if fixed and self.enabled:
            if args or kwargs or tuple(pixels.shape) != self.shapes[call]:
                self.enabled = False
                self.reason = '原生图片编码顺序或尺寸变化'
            else:
                try:
                    key = (self.scope, self.cache.stamp(self.vae), pixel_key(pixels))
                    cached = self.cache.get(key, self.vae.output_device)
                    if cached is not None:
                        self.hits += 1
                        self.timings[label] += time.monotonic() - started
                        return cached
                except Exception as error:
                    # Cache failures may fall back. The actual encode below is
                    # deliberately outside this handler so errors/cancel surface.
                    key = None
                    self.enabled = False
                    self.reason = f'缓存回退：{type(error).__name__}'
        if fixed:
            self.misses += 1
        result = self.vae.encode(pixels, *args, **kwargs)
        if key is not None:
            try:
                self.cache.put(key, result)
            except Exception as error:
                self.enabled = False
                self.reason = f'缓存写入回退：{type(error).__name__}'
        self.timings[label] += time.monotonic() - started
        return result


class TimedProxy:
    def __init__(self, target, timings, methods):
        self.target, self.timings, self.methods = target, timings, methods

    def __getattr__(self, name):
        original = getattr(self.target, name)
        if name not in self.methods:
            return original
        def measured(*args, **kwargs):
            started = time.monotonic()
            try:
                return original(*args, **kwargs)
            finally:
                self.timings[self.methods[name]] += time.monotonic() - started
        return measured


def new_timings():
    return defaultdict(float)
