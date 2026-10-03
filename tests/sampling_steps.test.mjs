import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { migrateSamplingNodeData, stampSamplingNodeData } from '../web/sampling_steps_migration.js';

function legacy(total, first) {
  return {
    id: 462, type: 'H3AVSamplingSteps', title: 'My sampling controls',
    properties: { 'Node name for S&R': 'H3AVSamplingSteps', custom: 'retained' },
    widgets_values: [total, first], inputs: [],
    outputs: [{ name: '一采步数', links: [1201] }, { name: '二采步数', links: [1202] }, { name: '总步数', links: null }],
  };
}

test('legacy 16 total / 10 first becomes independent 10 / 6 with output links and custom title intact', () => {
  const data = legacy(16, 10);
  const outputs = data.outputs;
  assert.equal(migrateSamplingNodeData(data), true);
  assert.deepEqual(data.widgets_values, [10, 6]);
  assert.equal(data.outputs, outputs);
  assert.equal(data.title, 'My sampling controls');
  assert.equal(data.properties.custom, 'retained');
  assert.equal(data.properties.h3_sampling_steps_version, 2);
  assert.equal(migrateSamplingNodeData(data), false);
  assert.deepEqual(data.widgets_values, [10, 6]);
});

test('legacy 14 / 14 retains a complete 14-step first pass with a valid spare second count', () => {
  const data = legacy(14, 14);
  migrateSamplingNodeData(data);
  assert.deepEqual(data.widgets_values, [14, 1]);
});

test('new 14 / 4 survives JSON round trips without reinterpreting it as a legacy total', () => {
  const data = stampSamplingNodeData(legacy(14, 4));
  for (let index = 0; index < 3; index++) {
    const copy = JSON.parse(JSON.stringify(data));
    assert.equal(migrateSamplingNodeData(copy), false);
    assert.deepEqual(copy.widgets_values, [14, 4]);
  }
});

test('named legacy values and disconnected converted sockets migrate safely', () => {
  const data = legacy(16, 10);
  data.widgets_values_named = { 总步数: 14, 一采步数: 14 };
  data.inputs = [
    { name: '总步数', type: 'INT', link: null, widget: { name: '总步数' } },
    { name: '一采步数', type: 'INT', link: null, widget: { name: '一采步数' } },
  ];
  migrateSamplingNodeData(data);
  assert.deepEqual(data.widgets_values, [14, 1]);
  assert.deepEqual(data.widgets_values_named, { 一采步数: 14, 二采步数: 1 });
  assert.deepEqual(data.inputs.map((input) => input.name), ['一采步数']);
  assert.equal(data.inputs[0].widget.name, '一采步数');
});

test('legacy named-record and indexed custom serializers preserve the intended step counts', () => {
  for (const values of [{ 总步数: 14, 一采步数: 14 }, { 0: 14, 1: 14, length: 2 }]) {
    const data = legacy(16, 10);
    data.widgets_values = values;
    migrateSamplingNodeData(data);
    assert.deepEqual(data.widgets_values, [14, 1]);
    assert.equal(data.properties.h3_sampling_steps_version, 2);
  }
  const named = legacy(16, 10);
  named.widgets_values = { 总步数: 16, 一采步数: 10, custom: 'retained' };
  migrateSamplingNodeData(named);
  assert.deepEqual(named.widgets_values_named, { 一采步数: 10, 二采步数: 6, custom: 'retained' });
});

test('new named/indexed serializers normalize without subtracting the first count from a total', () => {
  for (const values of [{ 一采步数: 14, 二采步数: 4 }, { 0: 14, 1: 4, length: 2 }]) {
    const data = stampSamplingNodeData(legacy(14, 4));
    data.widgets_values = values;
    assert.equal(migrateSamplingNodeData(data), true);
    assert.deepEqual(data.widgets_values, [14, 4]);
    assert.equal(migrateSamplingNodeData(data), false);
  }
  const clipboard = stampSamplingNodeData(legacy(14, 4));
  delete clipboard.widgets_values;
  clipboard.widgets_values_named = { 一采步数: 14, 二采步数: 4 };
  migrateSamplingNodeData(clipboard);
  assert.deepEqual(clipboard.widgets_values, [14, 4]);
});

test('legacy linked total or first controls are explicitly rejected without touching saved data', () => {
  for (const name of ['总步数', '一采步数']) {
    const data = legacy(16, 10);
    data.inputs = [{ name, type: 'INT', link: 91, widget: { name } }];
    const before = JSON.stringify(data);
    assert.throws(() => migrateSamplingNodeData(data), /Legacy linked step controls need manual migration/);
    assert.equal(JSON.stringify(data), before);
  }
  // Widget names, rather than presentation labels, also identify converted inputs.
  const translated = legacy(16, 10);
  translated.inputs = [{ name: 'Total Sampling Steps', link: 92, widget: { name: '总步数' } }];
  assert.throws(() => migrateSamplingNodeData(translated), /manual migration/);
});

test('invalid legacy counts are rejected before migration and new connected controls remain untouched', () => {
  for (const [total, first] of [[10, 14], [0, 10], [16, 0], [16, 2.5]]) {
    const data = legacy(total, first);
    const before = JSON.stringify(data);
    assert.throws(() => migrateSamplingNodeData(data));
    assert.equal(JSON.stringify(data), before);
  }
  const data = stampSamplingNodeData(legacy(14, 4));
  data.inputs = [{ name: '二采步数', link: 93, widget: { name: '二采步数' } }];
  const before = JSON.stringify(data);
  migrateSamplingNodeData(data);
  assert.equal(JSON.stringify(data), before);
});

async function extension() {
  const source = await readFile(new URL('../web/sampling_steps.js', import.meta.url), 'utf8');
  let registered;
  const app = { registerExtension(value) { registered = value; } };
  const body = source.replace(/^import .*;\r?\n/gm, '');
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  await new AsyncFunction('app', 'migrateSamplingNodeData', 'stampSamplingNodeData', body)(app, migrateSamplingNodeData, stampSamplingNodeData);
  return registered;
}

test('graph configuration migrates before widget assignment while preserving graph links', async () => {
  const hook = await extension();
  const other = { id: 20, type: 'SamplerCustomAdvanced', widgets_values: [16, 10] };
  const graph = { nodes: [legacy(16, 10), other], links: [[1201, 462, 0, 512, 0, 'INT'], [1202, 462, 1, 514, 0, 'INT']] };
  const links = JSON.stringify(graph.links);
  hook.beforeConfigureGraph(graph);
  assert.deepEqual(graph.nodes[0].widgets_values, [10, 6]);
  assert.deepEqual(other.widgets_values, [16, 10]);
  assert.equal(JSON.stringify(graph.links), links);
});

test('configure fallback restores live values and serialization stamps version without replacing existing hooks', async () => {
  const hook = await extension();
  const events = [];
  class Node {
    constructor() {
      this.type = 'H3AVSamplingSteps';
      this.widgets = [{ name: '一采步数', value: 16 }, { name: '二采步数', value: 10 }];
      this.inputs = [{ name: '总步数', link: null, widget: { name: '总步数' } }];
    }
    onConfigure(data) { events.push(data.id); return 'configured'; }
    onSerialize(data) { data.extra = 'retained'; return 'serialized'; }
    removeInput(index) { this.inputs.splice(index, 1); }
  }
  hook.beforeRegisterNodeDef(Node, { name: 'H3AVSamplingSteps' });
  const node = new Node();
  assert.equal(node.onConfigure(legacy(16, 10)), 'configured');
  assert.deepEqual(node.widgets.map((widget) => widget.value), [10, 6]);
  assert.deepEqual(node.inputs, []);
  assert.deepEqual(events, [462]);
  const serialized = { type: node.type, widgets_values: [14, 4] };
  assert.equal(node.onSerialize(serialized), 'serialized');
  assert.equal(serialized.extra, 'retained');
  assert.equal(serialized.properties.h3_sampling_steps_version, 2);
  assert.equal(migrateSamplingNodeData(serialized), false);
  assert.deepEqual(serialized.widgets_values, [14, 4]);
  const newNode = new Node();
  hook.nodeCreated(newNode);
  assert.equal(newNode.properties.h3_sampling_steps_version, 2);
});
