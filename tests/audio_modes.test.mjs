import test from 'node:test';
import assert from 'node:assert/strict';
import { installAudioModeControls } from '../web/audio_mode_controls.js';

test('each toggle switches off the other and keeps existing callbacks', () => {
  let count = 0;
  const voice = { name: 'voice_reference', value: false, callback() { count++; } };
  const drive = { name: 'audio_drive', value: false };
  const node = { widgets: [voice, drive], onConfigure() { count++; } };
  installAudioModeControls(node);
  installAudioModeControls(node);
  voice.callback(true);
  assert.equal(voice.value, true);
  assert.equal(drive.value, false);
  drive.callback(true);
  assert.equal(voice.value, false);
  assert.equal(drive.value, true);
  drive.callback(false);
  assert.equal(voice.value || drive.value, false);
  voice.value = drive.value = true;
  node.onConfigure({});
  assert.equal(voice.value, false);
  assert.equal(drive.value, true);
  assert.equal(count, 2);
});
