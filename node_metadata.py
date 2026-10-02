"""English backend metadata; ComfyUI localizes each browser independently."""
import json
from pathlib import Path


def apply_metadata(node_classes, display_names):
    definitions = json.loads((Path(__file__).parent / "locales" / "en" / "nodeDefs.json").read_text(encoding="utf-8"))
    for node_id, cls in node_classes.items():
        entry = definitions[node_id]
        display_names[node_id] = entry["display_name"]
        cls.DESCRIPTION = entry["description"]
        original = cls.INPUT_TYPES

        def input_types(node_cls, original=original, entry=entry):
            result = {}
            for group, inputs in original().items():
                result[group] = {}
                for name, spec in inputs.items():
                    options = dict(spec[1]) if len(spec) > 1 else {}
                    options["tooltip"] = entry["inputs"][name]["tooltip"]
                    result[group][name] = (spec[0], options, *spec[2:])
            return result

        cls.INPUT_TYPES = classmethod(input_types)
        cls.OUTPUT_TOOLTIPS = tuple(entry["outputs"][str(i)]["tooltip"] for i in range(len(cls.RETURN_TYPES)))
