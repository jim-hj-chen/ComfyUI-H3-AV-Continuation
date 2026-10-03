"""CPU regression tests for exact fixed-image reuse, without GPU inference."""
import contextlib
import gc
import io
import os
import sys
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_contract import pack
from pack_under_test import reference_cache as caching
from pack_under_test import workflow_tools


class Tensor(np.ndarray):
    def detach(self): return self
    def numpy(self): return np.asarray(self)
    def contiguous(self): return np.ascontiguousarray(self).view(Tensor)
    def clone(self): return self.copy().view(Tensor)
    def numel(self): return self.size
    def element_size(self): return self.itemsize
    def to(self, *args, **kwargs): return self


def tensor(value, dtype=np.float32):
    return np.asarray(value, dtype=dtype).view(Tensor)


def picture(value, size=32, batch=1):
    return tensor(np.full((batch, size, size, 3), value))


class Nested:
    is_nested = True
    def __init__(self, streams): self.tensors = list(streams)
    def unbind(self): return tuple(self.tensors)


TORCH = types.SimpleNamespace(
    float32=np.float32, from_numpy=tensor,
    ones=lambda shape, **kw: tensor(np.ones(shape)),
    zeros=lambda shape, **kw: tensor(np.zeros(shape)),
    nn=types.SimpleNamespace(functional=types.SimpleNamespace(
        pad=lambda value, amount: tensor(np.pad(value, [(0, 0)] * (value.ndim - 1) + [amount])))))


class VideoVAE:
    output_device = 'cpu'
    vae_dtype = np.float32

    def __init__(self):
        self.first_stage_model = object()
        self.patcher = types.SimpleNamespace(patches_uuid='initial')
        self.output_dtype = np.float32
        self.calls = []

    def vae_output_dtype(self): return self.output_dtype

    def encode(self, pixels, *args, **kwargs):
        self.calls.append((pixels.clone(), args, kwargs))
        return tensor(np.full((1, 4, 1, pixels.shape[1] // 16, pixels.shape[2] // 16),
                              float(pixels.mean())), self.output_dtype)


class AudioVAE:
    sample_rate = 48000
    def __init__(self): self.calls = []
    def encode(self, waveform):
        self.calls.append(waveform.clone())
        return tensor(np.full((1, 32, 2, 207), float(waveform.mean())))


class Clip:
    def __init__(self): self.prompts = []; self.encodes = 0
    def tokenize(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return {'prompt': prompt, **kwargs}
    def encode_from_tokens_scheduled(self, tokens):
        self.encodes += 1
        return [[tokens['prompt'], {'per_segment': self.encodes}]]


class CacheContracts(unittest.TestCase):
    def setUp(self):
        self.cache = caching.ReferenceCache(1024 * 1024)
        self.vae = VideoVAE()
        self.images = [picture((i + 1) / 10) for i in range(9)]

    def proxy(self, images=None, scope='run', vae=None):
        images = self.images if images is None else images
        return caching.VideoVAEProxy(vae or self.vae, scope,
                                    [tuple(image.shape) for image in images],
                                    caching.new_timings(), cache=self.cache)

    def encode(self, images=None, scope='run', vae=None):
        images = self.images if images is None else images
        proxy = self.proxy(images, scope, vae)
        return proxy, [proxy.encode(image) for image in images]

    def test_nine_images_reuse_across_distinct_proxies_and_reordering(self):
        first, latents = self.encode()
        second, reused = self.encode()
        self.assertEqual((first.misses, first.hits, second.misses, second.hits), (9, 0, 0, 9))
        self.assertEqual(len(self.vae.calls), 9)
        for expected, actual in zip(latents, reused): np.testing.assert_array_equal(expected, actual)
        third, reordered = self.encode(list(reversed(self.images)))
        self.assertEqual(third.hits, 9)
        for expected, actual in zip(reversed(latents), reordered): np.testing.assert_array_equal(expected, actual)

    def test_changed_pixels_and_native_resized_shape_invalidate_only_that_image(self):
        self.encode()
        changed = list(self.images)
        changed[4] = changed[4].clone()
        changed[4][0, 0, 0, 0] += 0.01
        proxy, _ = self.encode(changed)
        self.assertEqual((proxy.hits, proxy.misses), (8, 1))
        resized = picture(0.1, size=64)
        proxy, _ = self.encode([resized])
        self.assertEqual((proxy.hits, proxy.misses), (0, 1))
        self.assertEqual(len(self.vae.calls), 11)

    def test_vae_model_patches_and_dtypes_invalidate(self):
        image = [self.images[0]]
        self.encode(image)
        held = []
        changes = [
            ('first_stage_model', object()),
            ('patcher', types.SimpleNamespace(patches_uuid='initial')),
            ('vae_dtype', np.float16),
            ('output_dtype', np.float64),
        ]
        for name, value in changes:
            held.append(getattr(self.vae, name))
            setattr(self.vae, name, value)
            with self.subTest(attribute=name): self.assertEqual(self.encode(image)[0].misses, 1)
        self.vae.patcher.patches_uuid = 'repatched'
        self.assertEqual(self.encode(image)[0].misses, 1)
        replacement = VideoVAE()
        self.assertEqual(self.encode(image, vae=replacement)[0].misses, 1)
        self.assertEqual(len(replacement.calls), 1)

    def test_different_runs_do_not_reuse_and_absent_scope_disables(self):
        self.encode([self.images[0]], scope='one')
        self.assertEqual(self.encode([self.images[0]], scope='two')[0].misses, 1)
        before = self.cache.usage()
        for _ in range(2):
            proxy, _ = self.encode([self.images[0]], scope=None)
            self.assertFalse(proxy.enabled)
            self.assertEqual((proxy.hits, proxy.misses), (0, 1))
        self.assertEqual(self.cache.usage(), before)

    def test_video_and_audio_are_encoded_each_time(self):
        video = picture(0.4, batch=5)
        audio = AudioVAE()
        timings = caching.new_timings()
        audio_proxy = caching.TimedProxy(audio, timings, {'encode': 'reference_audio'})
        for _ in range(2):
            proxy, _ = self.encode([self.images[0]])
            proxy.encode(video)
            audio_proxy.encode(tensor(np.ones((1, 1, 480))))
            self.assertIn('previous_video', proxy.timings)
        self.assertEqual(len(self.vae.calls), 3)
        self.assertEqual([call[0].shape[0] for call in self.vae.calls], [1, 5, 5])
        self.assertEqual(len(audio.calls), 2)
        self.assertEqual(audio_proxy.sample_rate, audio.sample_rate)
        self.assertIn('reference_audio', timings)

    def test_cached_tensors_are_isolated_from_miss_and_hit_mutation(self):
        _, original = self.encode([self.images[0]])
        expected = original[0].clone()
        original[0][:] = 999
        _, hit = self.encode([self.images[0]])
        np.testing.assert_array_equal(hit[0], expected)
        hit[0][:] = -999
        _, later = self.encode([self.images[0]])
        np.testing.assert_array_equal(later[0], expected)
        self.assertEqual(len(self.vae.calls), 1)

    def test_lru_budget_and_run_release(self):
        # A fake 32px VAE latent is 64 bytes, so only two fit.
        self.cache = caching.ReferenceCache(128)
        self.encode([self.images[0], self.images[1]])
        self.assertEqual(self.cache.usage(), 128)
        self.assertEqual(self.encode([self.images[0]])[0].hits, 1)
        self.encode([self.images[2]])
        self.assertEqual(self.encode([self.images[0]])[0].hits, 1)
        self.assertEqual(self.encode([self.images[1]])[0].misses, 1)
        self.assertLessEqual(self.cache.usage(), 128)
        self.encode([self.images[3]], scope='other')
        self.cache.release('run')
        self.assertEqual(self.cache.usage(), 64)
        self.cache.release('other')
        self.assertEqual(self.cache.usage(), 0)
        self.cache = caching.ReferenceCache(63)
        self.encode([self.images[0]])
        self.assertEqual(self.cache.usage(), 0)
        self.assertEqual(self.encode([self.images[0]])[0].misses, 1)

    def test_collected_vae_does_not_stay_alive_in_cache(self):
        vae = VideoVAE()
        reference = weakref.ref(vae)
        self.encode([self.images[0]], vae=vae)
        self.assertGreater(self.cache.usage(), 0)
        del vae
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(self.cache.usage(), 0)
        self.assertFalse(self.cache.vaes)

    def test_cache_read_write_and_hash_errors_fall_back(self):
        for method in ('stamp', 'get', 'put'):
            with self.subTest(method=method), patch.object(self.cache, method, side_effect=RuntimeError('cache issue')):
                proxy, result = self.encode([self.images[0]])
                self.assertFalse(proxy.enabled)
                self.assertEqual(proxy.misses, 1)
                np.testing.assert_allclose(result[0], self.images[0].mean())
        with patch.object(caching, 'pixel_key', side_effect=ValueError('hash issue')):
            proxy, _ = self.encode([self.images[0]])
            self.assertFalse(proxy.enabled)
            self.assertEqual(proxy.misses, 1)

    def test_actual_encoder_errors_and_cancellation_propagate(self):
        class Cancelled(BaseException): pass
        for error in (RuntimeError('model error'), Cancelled('user cancelled')):
            with self.subTest(error=type(error).__name__), patch.object(self.vae, 'encode', side_effect=error):
                with self.assertRaises(type(error)): self.encode([self.images[0]])
        self.assertEqual(self.cache.usage(), 0)

    def test_unexpected_shape_or_call_arguments_disable_proxy_cache(self):
        for pixels, kwargs in ((picture(0.2, batch=2), {}), (self.images[0], {'custom': True})):
            with self.subTest(shape=pixels.shape, kwargs=kwargs):
                proxy = self.proxy(self.images[:2])
                proxy.encode(pixels, **kwargs)
                proxy.encode(self.images[1])
                self.assertFalse(proxy.enabled)
                self.assertEqual((proxy.hits, proxy.misses), (0, 2))
        self.assertEqual(self.cache.usage(), 0)

    def test_shape_guard_matches_native_max_and_match_rules(self):
        refs = {'ref_image_0': np.broadcast_to(tensor(np.zeros((1, 1, 1, 3))), (1, 3840, 3840, 3))}
        self.assertEqual(caching.image_shapes(refs, 864, 480, 'max'), [(1, 2048, 2048, 3)])
        self.assertEqual(caching.image_shapes(refs, 864, 480, 'match'), [(1, 640, 640, 3)])

    def test_loop_scope_accepts_valid_indexes_and_identifies_final_segment(self):
        self.assertEqual(caching.loop_scope({'run_id': 'run', 'index': 0, 'count': 3}), ('run', False))
        self.assertEqual(caching.loop_scope({'run_id': 'run', 'index': 2, 'count': 3}), ('run', True))
        for ctx in (None, [], {}, {'run_id': 'run', 'index': 3, 'count': 3},
                    {'run_id': 'run', 'index': -1, 'count': 3}, {'run_id': '', 'index': 0, 'count': 3}):
            with self.subTest(context=ctx): self.assertEqual(caching.loop_scope(ctx), (None, False))


class NativeReferenceRouting(unittest.TestCase):
    def setUp(self):
        self.cache = caching.ReferenceCache(1024 * 1024)
        self.vae, self.audio, self.clip = VideoVAE(), AudioVAE(), Clip()
        self.refs = {f'ref_image_{i}': picture((i + 1) / 10) for i in range(9)}
        self.native_calls = []
        native = types.ModuleType('comfy_extras.nodes_minimax_h3')
        native.MiniMaxH3ReferenceToVideo = types.SimpleNamespace(execute=self.execute_native)
        native._encode_ref_audio = lambda vae, audio: (vae.encode(audio['waveform']), 207)
        extras = types.ModuleType('comfy_extras')
        extras.nodes_minimax_h3 = native
        nested = types.SimpleNamespace(NestedTensor=Nested)
        self.patches = [
            patch.dict(sys.modules, {'torch': TORCH, 'torch.nn.functional': TORCH.nn.functional,
                                    'comfy_extras': extras, 'comfy_extras.nodes_minimax_h3': native,
                                    'comfy.nested_tensor': nested}),
            patch.object(workflow_tools, 'CACHE', self.cache),
            patch.object(workflow_tools, 'VideoVAEProxy', side_effect=lambda *args, **kw:
                         caching.VideoVAEProxy(*args, cache=self.cache, **kw)),
        ]
        for active in self.patches: active.start()
        self.output = io.StringIO()
        self.stdout = contextlib.redirect_stdout(self.output)
        self.stdout.__enter__()

    def tearDown(self):
        self.stdout.__exit__(None, None, None)
        for active in reversed(self.patches): active.stop()

    def execute_native(self, **kwargs):
        # Mirror the relevant native order: fixed images, previous video,
        # reference audio, and fresh text/visual conditioning every segment.
        encoded = [kwargs['vae'].encode(image) for image in kwargs['ref_images'].values()]
        videos = [kwargs['vae'].encode(video) for video in kwargs['ref_videos'].values()]
        audios = [kwargs['audio_vae'].encode(audio['waveform'])
                  for audio in (*kwargs['ref_video_audios'].values(), *kwargs['ref_audios'].values())]
        tokens = kwargs['clip'].tokenize(kwargs['prompt'], minimax_ref_items=list(kwargs['ref_images']))
        conditioning = kwargs['clip'].encode_from_tokens_scheduled(tokens)
        self.native_calls.append({'kwargs': kwargs, 'images': encoded, 'videos': videos, 'audios': audios})
        latent = {'samples': Nested((tensor(np.zeros((1, 24, 37, 2, 2))),
                                     tensor(np.zeros((1, 32, 2, 207)))))}
        return conditioning, latent

    def call(self, index, count=3, mode='generated', load=True):
        node = pack.NODE_CLASS_MAPPINGS['H3AVReferenceListToVideo']()
        audio = {'waveform': tensor(np.full((1, 1, 480), 0.1 + index)), 'sample_rate': 48000}
        args = {'vae': self.vae, 'audio_vae': self.audio,
                'loop_ctx': {'version': 3, 'run_id': 'b' * 24, 'index': index, 'count': count}}
        if index:
            args.update(previous_frames=picture(0.2 + index, batch=5), previous_audio=audio)
        if mode == 'voice_reference': args['speaker_reference'] = audio
        if mode == 'audio_drive': args['driving_audio'] = audio
        loader = patch.object(node, 'load_references', return_value=self.refs) if load else contextlib.nullcontext()
        with loader:
            return node.encode(self.clip, None, f'prompt {index}', 32, 32, 124, **args)

    def test_each_audio_mode_keeps_live_conditions_while_reusing_fixed_images(self):
        for mode in ('generated', 'voice_reference', 'audio_drive'):
            with self.subTest(mode=mode):
                self.cache.release('b' * 24)
                self.vae.calls.clear(); self.audio.calls.clear(); self.clip.prompts.clear()
                self.native_calls.clear()
                results = [self.call(index, mode=mode) for index in range(3)]
                self.assertEqual(len(self.vae.calls), 11)  # Nine fixed images + two live video tails.
                self.assertEqual(self.clip.prompts, ['prompt 0', 'prompt 1', 'prompt 2'])
                self.assertEqual([result[0][0][0] for result in results], self.clip.prompts)
                self.assertEqual(len(self.audio.calls), 2 if mode == 'generated' else 5)
                self.assertEqual(self.cache.usage(), 0)  # Last segment releases this run.
                for current in self.native_calls[1:]:
                    for expected, actual in zip(self.native_calls[0]['images'], current['images']):
                        np.testing.assert_array_equal(expected, actual)
                if mode == 'audio_drive':
                    for index, (_, latent) in enumerate(results):
                        self.assertFalse(latent['noise_mask'].unbind()[1].any())
                        self.assertTrue(latent['noise_mask'].unbind()[0].all())
                        np.testing.assert_allclose(latent['samples'].unbind()[1], 0.1 + index, atol=1e-6)
                else:
                    for _, latent in results: self.assertNotIn('noise_mask', latent)
        self.assertIn('固定图缓存：命中 9/9，实际编码 0 张', self.output.getvalue())
        self.assertIn('文本/视觉条件', self.output.getvalue())

    def test_same_file_replaced_with_preserved_timestamp_is_reencoded(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / 'fixed.png'
            Image.new('RGB', (32, 32), color=(32, 64, 96)).save(path)
            timestamp = path.stat().st_mtime_ns
            node = pack.NODE_CLASS_MAPPINGS['H3AVReferenceListToVideo']()
            images = {'items': [{'path': str(path)}]}
            args = dict(vae=self.vae, audio_vae=self.audio, loop_ctx={'run_id': 'file', 'index': 0, 'count': 3})
            node.encode(self.clip, images, 'first', 32, 32, 124, **args)
            node.encode(self.clip, images, 'second', 32, 32, 124, **args)
            self.assertEqual(len(self.vae.calls), 1)
            Image.new('RGB', (32, 32), color=(96, 64, 32)).save(path)
            os.utime(path, ns=(timestamp, timestamp))
            node.encode(self.clip, images, 'changed', 32, 32, 124, **args)
            self.assertEqual(len(self.vae.calls), 2)
            self.assertEqual(self.clip.prompts, ['first', 'second', 'changed'])

    def test_final_segment_native_error_releases_run_and_propagates(self):
        self.call(0, count=2)
        self.assertGreater(self.cache.usage(), 0)
        with patch.object(self.vae, 'encode', side_effect=RuntimeError('live tail failed')):
            with self.assertRaisesRegex(RuntimeError, 'live tail failed'): self.call(1, count=2)
        self.assertEqual(self.cache.usage(), 0)


if __name__ == '__main__': unittest.main()
