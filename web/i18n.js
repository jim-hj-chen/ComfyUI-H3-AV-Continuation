import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { createLocalizer } from "./localization.js";

let localize;
let locale;
let nodeIds;
const liveNodes = new Set();

function currentLocale() {
  return locale ?? app.extensionManager?.setting?.get?.("Comfy.Locale")
    ?? app.ui?.settings?.getSettingValue?.("Comfy.Locale") ?? "en";
}

function updateNodes() {
  if (!localize) return;
  for (const node of liveNodes) localize(node, currentLocale());
}

function track(node) {
  if (liveNodes.has(node) || (nodeIds && !nodeIds.has(node.type))) return;
  liveNodes.add(node);
  const onRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    liveNodes.delete(this);
    return onRemoved?.apply(this, args);
  };
  localize?.(node, currentLocale());
}

app.registerExtension({
  name: "ComfyUI-H3-AV-Continuation",
  async setup() {
    try {
      const [packResponse, translationResponse] = await Promise.all([
        fetch(new URL("./pack.json", import.meta.url)),
        api.fetchApi("/i18n"),
      ]);
      if (!packResponse.ok || !translationResponse.ok) {
        throw new Error("ComfyUI /api/i18n or pack metadata is unavailable");
      }
      const pack = await packResponse.json();
      const translations = await translationResponse.json();
      nodeIds = new Set(Object.keys(pack.legacyTitles));
      for (const node of liveNodes) {
        if (!nodeIds.has(node.type)) liveNodes.delete(node);
      }
      localize = createLocalizer(translations, pack.legacyTitles);
      updateNodes();
    } catch (error) {
      // Native locales still provide labels on frontends supporting i18n.
      console.warn("[ComfyUI bilingual nodes] Canvas localization unavailable:", error);
    }
    app.ui?.settings?.addEventListener?.("Comfy.Locale.change", (event) => {
      locale = event.detail?.value;
      updateNodes();
      // Let the native definition refresh settle, then restore canvas labels.
      requestAnimationFrame(updateNodes);
    });
  },
  nodeCreated(node) {
    // Track only this pack's H3 nodes; other extensions are unaffected.
    if (node.type?.startsWith("H3")) track(node);
  },
  loadedGraphNode(node) {
    localize?.(node, currentLocale());
  },
  afterConfigureGraph() {
    updateNodes();
  },
});
