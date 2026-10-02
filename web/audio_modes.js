import { app } from '../../../scripts/app.js';
import { installAudioModeControls } from './audio_mode_controls.js';

app.registerExtension({
  name: 'ComfyUI-H3-AV-Continuation.AudioModes',
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== 'H3AVAudioModes') return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = created?.apply(this, args);
      installAudioModeControls(this);
      return result;
    };
  },
  loadedGraphNode(node) {
    if (node.type === 'H3AVAudioModes') installAudioModeControls(node);
  },
});
