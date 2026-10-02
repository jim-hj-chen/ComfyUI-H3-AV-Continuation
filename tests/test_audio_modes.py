"""CPU contract tests; fake tensors exercise routing, not GPU model quality."""
import json
import subprocess
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_contract import pack


class Tensor(np.ndarray):
    def detach(self): return self
    def cpu(self): return self
    def float(self): return self
    def numpy(self): return np.asarray(self)
    def to(self, *args, **kwargs): return self


def tensor(value):
    return np.asarray(value, dtype=np.float32).view(Tensor)


class Nested:
    is_nested = True
    def __init__(self, streams): self.tensors = list(streams)
    def unbind(self): return tuple(self.tensors)


TORCH = types.SimpleNamespace(
    isfinite=np.isfinite, float32=np.float32, from_numpy=tensor,
    ones=lambda shape, **kw: tensor(np.ones(shape)), zeros=lambda shape, **kw: tensor(np.zeros(shape)),
    nn=types.SimpleNamespace(functional=types.SimpleNamespace(
        pad=lambda value, amount: tensor(np.pad(value, [(0, 0)] * (value.ndim - 1) + [amount])))))


class AudioContracts(unittest.TestCase):
    def setUp(self):
        self.fake = patch.dict(sys.modules, {'torch': TORCH})
        self.fake.start()
        self.audio = {'waveform': tensor(np.arange(4800)[None, None] / 10000), 'sample_rate': 480}

    def tearDown(self): self.fake.stop()

    def test_planner_loads_source_only_for_selected_mode(self):
        node = pack.H3AVSegmentPlan()
        self.assertEqual(node.check_lazy_status(), [])
        for mode in ('voice_reference','audio_drive'):
            self.assertEqual(node.check_lazy_status(audio_mode=mode), ['source_audio'])
            self.assertEqual(node.check_lazy_status(audio_mode=mode,source_audio=self.audio), [])
        with self.assertRaises(ValueError): node.check_lazy_status(audio_mode='invalid')

    def test_continuous_source_slices_and_discarded_padding(self):
        from pack_under_test.audio_modes import driving_window
        first = driving_window(self.audio, 0, 120, 124, 0)['waveform']
        second = driving_window(self.audio, 120, 120, 158, 22)['waveform']
        np.testing.assert_equal(first[..., :2400], self.audio['waveform'][..., :2400])
        np.testing.assert_equal(second[..., 440:2840], self.audio['waveform'][..., 2400:4800])
        np.testing.assert_equal(second[..., :440], self.audio['waveform'][..., 1960:2400])
        self.assertFalse(first[..., 2400:].any())
        self.assertFalse(second[..., 2840:].any())
        with self.assertRaises(ValueError): driving_window(self.audio, 120, 144, 175, 22)

    def test_writer_skips_decoding_when_source_is_present(self):
        node = pack.H3AVEncodeSegment()
        self.assertEqual(node.check_lazy_status(), ['audio'])
        self.assertEqual(node.check_lazy_status(driving_audio=self.audio), [])
        self.assertEqual(node.check_lazy_status(audio=self.audio), [])
        self.assertTrue(node.INPUT_TYPES()['required']['audio'][1]['lazy'])

    def test_prompt_does_not_force_mandarin_and_voice_tags_match(self):
        self.assertNotIn('Mandarin', pack.prepare_prompt('English speech.', 0, 124, 0, 22))
        self.assertIn('<Audio 1>', pack.prepare_prompt('hello', 0, 124, 0, 22, 'voice_reference'))
        self.assertIn('<Audio 2>', pack.prepare_prompt('hello', 1, 158, 22, 22, 'voice_reference'))
        self.assertNotIn('short speaker voice reference', pack.prepare_prompt('hello', 1, 158, 22, 22, 'audio_drive'))

    def test_drive_encodes_fixed_audio_in_native_latent(self):
        cond = [['embedding', {}]]
        latent = {'samples': Nested((tensor(np.zeros((1, 24, 37, 2, 2))), tensor(np.zeros((1, 32, 2, 207)))))}
        upstream = types.SimpleNamespace(
            MiniMaxH3ReferenceToVideo=types.SimpleNamespace(execute=lambda **kw: (cond, latent)),
            _encode_ref_audio=lambda vae, audio: (tensor(np.ones((1, 32, 2, 206))), 206))
        node = pack.NODE_CLASS_MAPPINGS['H3AVReferenceListToVideo']()
        with patch.dict(sys.modules, {'comfy_extras.nodes_minimax_h3': upstream,
                                     'comfy.nested_tensor': types.SimpleNamespace(NestedTensor=Nested),
                                     'torch.nn.functional': TORCH.nn.functional}), patch.object(node, 'load_references', return_value={}):
            _, result = node.encode(None, None, 'test', 32, 32, 124, audio_vae=object(), driving_audio=self.audio)
        self.assertTrue(result['samples'].unbind()[1][..., :-1].all())
        self.assertFalse(result['noise_mask'].unbind()[1].any())
        self.assertTrue(result['noise_mask'].unbind()[0].all())
        self.assertNotIn('noise_mask', latent)

    def test_real_ffmpeg_driven_segments_keep_timing_and_tail(self):
        try:
            import imageio_ffmpeg
        except ImportError:
            self.skipTest('imageio-ffmpeg unavailable')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ctx = {'version': 3, 'run_id': 'b' * 24, 'index': 0, 'count': 2}
            rate = 48000
            audio = {'waveform': tensor((0.1 * np.sin(np.arange(rate * 10) / 20))[None, None]), 'sample_rate': rate}
            with patch.object(sys.modules['folder_paths'], 'get_output_directory', create=True, return_value=temporary), \
                 patch.object(pack, '_space'), patch.object(pack, '_ffmpeg', return_value=imageio_ffmpeg.get_ffmpeg_exe()):
                for index in range(2):
                    ctx['index'] = index
                    planned = pack.H3AVSegmentPlan().plan(ctx, 'a' + pack.DELIMITER + 'b', 5, audio_mode='audio_drive', source_audio=audio)
                    self.assertEqual(planned[8], 120)
                    images = tensor(np.ones((planned[1], 32, 32, 3)) * (0.2 + index * 0.3))
                    pack.H3AVEncodeSegment().encode(ctx, images, None, planned[1], planned[4], kept_frames=planned[8], driving_audio=planned[7])
                run = directory / 'h3_av_sync' / ctx['run_id']
                manifest = json.loads((run / 'manifest.json').read_text())
                self.assertEqual([x['output_frames'] for x in manifest['segments']], [120, 120])
                for index in range(2):
                    with wave.open(str(run / f'segment_{index:05d}.wav')) as pcm:
                        self.assertEqual(pcm.getnframes(), rate * 5)
                        samples = np.frombuffer(pcm.readframes(rate * 5), dtype='<i2').reshape(-1, 2)[:, 0] / 32768
                        # Mono -> stereo conversion can use sqrt(1/2); no drift at the join.
                        target = audio['waveform'][0, 0, index * rate * 5:(index + 1) * rate * 5]
                        self.assertGreater(np.corrcoef(samples, target)[0, 1], 0.999)
                with np.load(run / 'tail_00001.npz') as tail:
                    np.testing.assert_allclose(tail['audio'][0, 0], audio['waveform'][0, 0, -44000:])
                final = pack.concat_run(run)
                self.assertTrue(final.is_file())
                self.assertEqual(json.loads((run / 'final_info.json').read_text())['duration_seconds'], 10)


class FFmpegTermination(unittest.TestCase):
    def test_comfy_interrupt_stops_process(self):
        from pack_under_test.ffmpeg_runner import run_ffmpeg
        def interrupt(): raise RuntimeError('test cancellation')
        with tempfile.TemporaryDirectory() as directory, patch('pack_under_test.ffmpeg_runner._interrupt', interrupt):
            with self.assertRaisesRegex(RuntimeError, 'test cancellation'):
                run_ffmpeg([sys.executable, '-c', 'import time; time.sleep(20)'], Path(directory) / 'cancel.log')

    def test_waiting_process_times_out(self):
        from pack_under_test.ffmpeg_runner import run_ffmpeg
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TimeoutError):
                run_ffmpeg([sys.executable, '-c', 'import time; time.sleep(20)'], Path(directory) / 'timeout.log', timeout=0.2)

    def test_blocked_pipe_times_out(self):
        from pack_under_test.ffmpeg_runner import run_ffmpeg
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TimeoutError):
                run_ffmpeg([sys.executable, '-c', 'import time; time.sleep(20)'], Path(directory) / 'pipe.log',
                           frames=iter([b'x' * (8 * 1024 * 1024)]), timeout=0.2)


if __name__ == '__main__': unittest.main()
