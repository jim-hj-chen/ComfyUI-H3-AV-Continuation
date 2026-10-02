import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.modules.setdefault('folder_paths', types.ModuleType('folder_paths'))
spec = importlib.util.spec_from_file_location('pack_under_test', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
pack = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pack
spec.loader.exec_module(pack)
AV = 'H3AVSegmentPlan' in pack.NODE_CLASS_MAPPINGS


class LocalizationContract(unittest.TestCase):
    def test_complete_bilingual_contract_and_defaults(self):
        for lang in ('en', 'zh'):
            definitions = json.loads((ROOT / 'locales' / lang / 'nodeDefs.json').read_text(encoding='utf-8'))
            categories = json.loads((ROOT / 'locales' / lang / 'main.json').read_text(encoding='utf-8'))['nodeCategories']
            self.assertEqual(set(definitions), set(pack.NODE_CLASS_MAPPINGS))
            for node_id, cls in pack.NODE_CLASS_MAPPINGS.items():
                with self.subTest(language=lang, node=node_id):
                    entry = definitions[node_id]
                    self.assertTrue(entry['display_name'])
                    self.assertTrue(entry['description'])
                    for category in cls.CATEGORY.split('/'):
                        self.assertTrue(categories[category])
                    inputs = {key: value for group in cls.INPUT_TYPES().values() for key, value in group.items()}
                    self.assertEqual(set(entry['inputs']), set(inputs))
                    for key, value in inputs.items():
                        self.assertTrue(entry['inputs'][key]['name'])
                        self.assertTrue(entry['inputs'][key]['tooltip'])
                        self.assertTrue(value[1]['tooltip'])
                        if isinstance(value[0], list):
                            self.assertEqual(set(entry['inputs'][key]['options']), set(value[0]))
                    self.assertEqual(set(entry['outputs']), {str(i) for i in range(len(cls.RETURN_TYPES))})
                    self.assertEqual(len(cls.OUTPUT_TOOLTIPS), len(cls.RETURN_TYPES))
                    for output in entry['outputs'].values():
                        self.assertTrue(output['name'])
                        self.assertTrue(output['tooltip'])
                    if lang == 'en':
                        self.assertEqual(pack.NODE_DISPLAY_NAME_MAPPINGS[node_id], entry['display_name'])
                        self.assertEqual(cls.DESCRIPTION, entry['description'])

    def test_original_workflow_ids_and_socket_contract(self):
        expected = {
            'H3AVSegmentPlan': ['loop_ctx', 'prompts', 'seconds', 'overlap_frames', 'base_seed', 'segment_prompt', 'audio_mode', 'source_audio'],
            'H3AVPickPrompt': ['prompts', 'count', 'loop_ctx'],
            'H3AVFreezeAudio': ['latent'],
            'H3AVRestoreAudio': ['refined', 'original'],
            'H3AVContinuationGuide': ['loop_ctx', 'positive', 'latent', 'vae', 'audio_vae', 'previous_frames', 'previous_audio'],
            'H3AVEncodeSegment': ['loop_ctx', 'images', 'audio', 'expected_frames', 'skip_frames', 'overlap_frames', 'crf', 'preset', 'reject_silence', 'continuation_images', 'create_av_preview', 'kept_frames'],
            'H3AVConcat': ['loop_ctx', 'done'],
            'H3AVSamplingSteps': ['总步数', '一采步数'],
            'H3AVReferenceListToVideo': ['clip', 'images', 'prompt', 'width', 'height', 'length', 'ref_image_size', 'vae', 'audio_vae', 'previous_frames', 'previous_audio', 'speaker_reference', 'driving_audio'],
            'H3AVAudioModes': ['voice_reference', 'audio_drive', 'audio'],
            'H3AVAudioOutput': ['audio_mode', 'generated_audio', 'driving_audio'],
            'H3AVOptionalAudioRefine': ['enabled', 'audio_mode', 'original', 'refined'],
            'H3AVOptionalVideoRefine': ['enabled', 'original', 'refined'],
            'H3AVOptionalUpscale': ['enabled', 'original', 'refined'],
        } if AV else {
            'H3DiskEncodeSegment': ['loop_ctx', 'images', 'audio', 'fps', 'crf', 'preset'],
            'H3DiskConcat': ['loop_ctx', 'done'],
        }
        self.assertEqual(set(pack.NODE_CLASS_MAPPINGS), set(expected))
        for node_id, keys in expected.items():
            cls = pack.NODE_CLASS_MAPPINGS[node_id]
            self.assertEqual([key for group in cls.INPUT_TYPES().values() for key in group], keys)
            self.assertTrue(callable(getattr(cls(), cls.FUNCTION)))
        writer = pack.NODE_CLASS_MAPPINGS['H3AVEncodeSegment' if AV else 'H3DiskEncodeSegment']
        inputs = writer.INPUT_TYPES()['required']
        self.assertEqual(inputs['crf'][1]['default'], 19)
        self.assertEqual(inputs['preset'][0], ['medium', 'fast', 'veryfast', 'slow'])
        if AV:
            self.assertEqual(inputs['overlap_frames'][1]['default'], 22)
            self.assertEqual(pack.NODE_CLASS_MAPPINGS['H3AVSamplingSteps'].RETURN_NAMES, ('一采步数', '二采步数', '总步数'))

    def test_metadata_returns_independent_option_dicts(self):
        for cls in pack.NODE_CLASS_MAPPINGS.values():
            before = cls.INPUT_TYPES()
            for group in before.values():
                for spec in group.values():
                    spec[1]['tooltip'] = 'modified by another extension'
            after = cls.INPUT_TYPES()
            for group in after.values():
                for spec in group.values():
                    self.assertNotEqual(spec[1]['tooltip'], 'modified by another extension')

    def test_unfinished_concat_has_no_disk_side_effects(self):
        cls = pack.H3AVConcat if AV else pack.H3DiskConcat
        self.assertEqual(cls().concat(None, False), ('',))

    @unittest.skipUnless(AV, 'H3 planner only')
    def test_frame_alignment_and_timecode_offset(self):
        self.assertEqual(pack.frame_plan(5, 0, 22), (124, 0))
        self.assertEqual(pack.frame_plan(5, 1, 22), (158, 22))
        self.assertEqual(pack.shift_timecodes('00:00.000 shot; 01:59.500', 0.75), '00:00.750 shot; 02:00.250')
        for seconds, overlap in ((0, 22), (float('nan'), 22), (5, 6), (20, 22)):
            with self.assertRaises(ValueError):
                pack.frame_plan(seconds, 1, overlap)

    @unittest.skipUnless(AV, 'H3 sampling controls only')
    def test_two_pass_step_split_preserves_chinese_api_keys(self):
        steps = pack.NODE_CLASS_MAPPINGS['H3AVSamplingSteps']()
        self.assertEqual(steps.split(**{'总步数': 16, '一采步数': 10}), (10, 6, 16))
        for total, first in ((1, 1), (16, 16), (16, 0), (10001, 1)):
            with self.assertRaises(ValueError):
                steps.split(total, first)

    @unittest.skipUnless(AV, 'H3 script picker only')
    def test_script_picker_handles_list_and_wrapped_list(self):
        ctx = {'version': 3, 'run_id': 'a' * 24, 'index': 1, 'count': 2}
        sys.modules['folder_paths'].get_output_directory = lambda: str(ROOT)
        picker = pack.H3AVPickPrompt()
        self.assertEqual(picker.pick(['first', 'second'], [2], [ctx]), ('second',))
        self.assertEqual(picker.pick([['first', 'second']], [2], [ctx]), ('second',))
        with self.assertRaises(ValueError):
            picker.pick(['first'], [2], [ctx])


if __name__ == '__main__':
    unittest.main()
