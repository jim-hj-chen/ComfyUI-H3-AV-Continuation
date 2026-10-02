/** Two separate toggles; enabling either clears the other immediately. */
export function installAudioModeControls(node) {
  const voice = node.widgets?.find(w => w.name === 'voice_reference');
  const drive = node.widgets?.find(w => w.name === 'audio_drive');
  if (!voice || !drive || node.__h3AudioModesInstalled) return;
  node.__h3AudioModesInstalled = true;
  for (const [current, other] of [[voice, drive], [drive, voice]]) {
    const callback = current.callback;
    current.callback = function (value, ...args) {
      current.value = Boolean(value);
      if (current.value) other.value = false;
      const result = callback?.call(this, value, ...args);
      node.setDirtyCanvas?.(true, true);
      return result;
    };
  }
  const configure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = configure?.apply(this, args);
    if (voice.value && drive.value) voice.value = false;
    return result;
  };
}
