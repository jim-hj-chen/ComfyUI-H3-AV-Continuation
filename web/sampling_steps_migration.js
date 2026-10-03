/** Upgrade saved inline [total, first] widgets without changing graph links. */
export const SAMPLING_STEPS_VERSION = 2;
const VERSION_KEY = "h3_sampling_steps_version";

function validStep(value, name) {
  if (!Number.isInteger(value) || value < 1 || value > 10000) {
    throw new Error(`[H3AVSamplingSteps] ${name}必须是 1–10000 的整数。`);
  }
  return value;
}

export function stampSamplingNodeData(data) {
  data.properties ??= {};
  data.properties[VERSION_KEY] = SAMPLING_STEPS_VERSION;
  return data;
}

export function migrateSamplingNodeData(data) {
  if (data.type !== "H3AVSamplingSteps") return false;
  const raw = data.widgets_values;
  // The workflow schema permits arrays, indexed array-like objects, and named
  // records. Clipboard loading can move named records to widgets_values_named.
  const rawNamed = raw && typeof raw === "object" && !Array.isArray(raw)
    && typeof raw.length !== "number" ? raw : {};
  const named = { ...rawNamed, ...data.widgets_values_named };
  if (Number(data.properties?.[VERSION_KEY]) >= SAMPLING_STEPS_VERSION) {
    // Preserve an ordinary new-format array as authoritative. Named records
    // and array-like custom serializers need canonical values for LiteGraph.
    if (Array.isArray(raw) || (raw == null && data.widgets_values_named == null)) return false;
    const first = validStep(named.一采步数 ?? raw?.[0] ?? 10, "一采步数");
    const second = validStep(named.二采步数 ?? raw?.[1] ?? 4, "二采步数");
    data.widgets_values = [first, second];
    if (Object.keys(rawNamed).length || data.widgets_values_named) {
      data.widgets_values_named = { ...named, 一采步数: first, 二采步数: second };
    }
    return true;
  }
  const legacyKeys = new Set(["总步数", "一采步数"]);
  const connected = (data.inputs ?? []).find((slot) => slot.link != null
    && (legacyKeys.has(slot.name) || legacyKeys.has(slot.widget?.name)));
  if (connected) {
    // A total-minus-first link cannot become an independent second-pass link
    // without an arithmetic node. Reject before changing any saved data.
    throw new Error("[H3AVSamplingSteps] 旧版采样步数使用了外部输入连线，无法安全自动改为独立步数。请先在旧版节点取消总步数/一采步数的输入连线并保存，再加载新版；或重新创建采样步数节点，分别连接一采步数和二采步数。 Legacy linked step controls need manual migration.");
  }
  const total = validStep(named.总步数 ?? raw?.[0] ?? 16, "旧版总步数");
  const first = validStep(named.一采步数 ?? raw?.[1] ?? 10, "旧版一采步数");
  if (total < first) {
    throw new Error("[H3AVSamplingSteps] 旧版总步数小于一采步数。请先修正旧配置，或重新创建独立采样步数节点。");
  }
  const second = Math.max(1, total - first);
  data.widgets_values = [first, second];
  if (Object.keys(rawNamed).length || data.widgets_values_named) {
    delete named.总步数;
    named.一采步数 = first;
    named.二采步数 = second;
    data.widgets_values_named = named;
  }
  // Disconnected converted widgets can safely drop the retired total socket.
  if (data.inputs) {
    data.inputs = data.inputs.filter((slot) => slot.name !== "总步数" && slot.widget?.name !== "总步数");
  }
  stampSamplingNodeData(data);
  return true;
}
