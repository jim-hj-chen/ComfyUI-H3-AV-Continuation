"""Independent first/second pass counts; Goohaitool controls stage execution."""


def _step_count(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name}必须为 1–10000 的整数。')
    try:
        integer = int(value)
        valid = integer == float(value)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid or not 1 <= integer <= 10000:
        raise ValueError(f'{name}必须为 1–10000 的整数。')
    return integer


class H3AVSamplingSteps:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            '一采步数': ('INT', {'default': 10, 'min': 1, 'max': 10000}),
            '二采步数': ('INT', {'default': 4, 'min': 1, 'max': 10000}),
        }}

    # Keep the original output positions and names for existing graph links.
    RETURN_TYPES = ('INT', 'INT', 'INT')
    RETURN_NAMES = ('一采步数', '二采步数', '总步数')
    FUNCTION = 'split'
    CATEGORY = 'H3 AV Continuation/Controls'

    def split(self, 一采步数=10, 二采步数=None, **legacy_inputs):
        first = _step_count(一采步数, '一采步数')
        if '总步数' in legacy_inputs:
            # Direct Python callers can retain the old named input. Legacy raw
            # API prompts still need the new schema's explicit second count.
            # It is intentionally absent from INPUT_TYPES and the node UI.
            total = _step_count(legacy_inputs.pop('总步数'), '旧版总步数')
            if 二采步数 is not None:
                raise ValueError('请使用独立的一采步数与二采步数，或仅使用旧版总步数；不要同时提供两种配置。')
            if total < first:
                raise ValueError('旧版总步数不能小于一采步数；请改为独立填写一采步数与二采步数。')
            # A single-pass legacy 14/14 configuration is now valid. Its spare
            # second-pass value is ignored when the existing group is bypassed.
            second = max(1, total - first)
        else:
            second = 4 if 二采步数 is None else _step_count(二采步数, '二采步数')
        if legacy_inputs:
            raise TypeError(f'不支持的采样步数输入：{", ".join(legacy_inputs)}')
        return first, second, first + second
