/** Update presentation without changing backend socket keys or widget values. */
export function createLocalizer(translations, legacyTitles) {
  const previousTitles = new WeakMap();
  const nodeIds = new Set(Object.keys(legacyTitles));

  return function localize(node, locale) {
    if (!nodeIds.has(node.type)) return;
    // ComfyUI uses zh for Simplified Chinese; accept regional Chinese tags too.
    const language = String(locale || "en").toLowerCase().startsWith("zh") ? "zh" : "en";
    const english = translations.en?.nodeDefs?.[node.type];
    const entry = translations[language]?.nodeDefs?.[node.type] ?? english;
    if (!entry) return;
    const standardTitles = new Set([
      node.type,
      legacyTitles[node.type],
      english?.display_name,
      translations.zh?.nodeDefs?.[node.type]?.display_name,
      previousTitles.get(node),
      node.constructor?.title,
    ]);
    if (standardTitles.has(node.title)) {
      node.title = entry.display_name;
      previousTitles.set(node, node.title);
    }
    for (const widget of node.widgets ?? []) {
      const input = entry.inputs?.[widget.name];
      if (!input) continue;
      widget.label = input.name;
      widget.tooltip = input.tooltip;
      // getOptionLabel is a presentation hook supported by ComfyUI combos.
      // Keep canonical options/values so workflows and FFmpeg still receive
      // tokens such as medium/match, rather than translated display strings.
      if (input.options && widget.options) {
        widget.options.getOptionLabel = (value) => input.options[value] ?? String(value ?? "");
      }
    }
    for (const slot of node.inputs ?? []) {
      const input = entry.inputs?.[slot.name];
      if (!input) continue;
      slot.label = input.name;
      slot.localized_name = input.name;
      slot.tooltip = input.tooltip;
    }
    for (const [index, slot] of (node.outputs ?? []).entries()) {
      const output = entry.outputs?.[String(index)];
      if (!output) continue;
      slot.label = output.name;
      slot.localized_name = output.name;
      slot.tooltip = output.tooltip;
    }
    node.setDirtyCanvas?.(true, true);
  };
}
