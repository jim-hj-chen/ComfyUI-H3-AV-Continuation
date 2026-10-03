import { app } from "../../../scripts/app.js";
import { migrateSamplingNodeData, stampSamplingNodeData } from "./sampling_steps_migration.js";

app.registerExtension({
  name: "ComfyUI-H3-AV-Continuation.SamplingSteps",
  beforeConfigureGraph(graphData) {
    for (const node of graphData?.nodes ?? []) migrateSamplingNodeData(node);
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "H3AVSamplingSteps") return;
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (data, ...args) {
      // Also support frontends that do not call beforeConfigureGraph. LiteGraph
      // assigns widget values before onConfigure, so update the live widgets.
      const migrated = migrateSamplingNodeData(data);
      const result = configure?.call(this, data, ...args);
      if (migrated) {
        for (const [index, name] of ["一采步数", "二采步数"].entries()) {
          const widget = this.widgets?.find((item) => item.name === name);
          if (widget) widget.value = data.widgets_values[index];
        }
        for (let index = (this.inputs?.length ?? 0) - 1; index >= 0; index--) {
          const slot = this.inputs[index];
          if (slot.name === "总步数" || slot.widget?.name === "总步数") {
            this.removeInput?.(index);
          }
        }
      }
      stampSamplingNodeData(this);
      return result;
    };
    const serialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (data, ...args) {
      const result = serialize?.call(this, data, ...args);
      stampSamplingNodeData(data);
      return result;
    };
  },
  nodeCreated(node) {
    if (node.type === "H3AVSamplingSteps") stampSamplingNodeData(node);
  },
});
