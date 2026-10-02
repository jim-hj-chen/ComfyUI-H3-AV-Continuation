import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { createLocalizer } from '../web/localization.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const pack = await read('../web/pack.json');
const translations = {
  en: { nodeDefs: await read('../locales/en/nodeDefs.json') },
  zh: { nodeDefs: await read('../locales/zh/nodeDefs.json') },
};

function makeNode(id) {
  const def = translations.en.nodeDefs[id];
  return {
    type: id, title: pack.legacyTitles[id],
    widgets: Object.entries(def.inputs).map(([name, input]) => ({ name, value: 19, options: { values: Object.keys(input.options ?? {}) } })),
    inputs: Object.keys(def.inputs).map((name, index) => ({ name, link: index + 123 })),
    outputs: Object.keys(def.outputs).map((name, index) => ({ name, links: [index + 234] })),
  };
}

test('every node switches en -> zh -> en without changing serialized keys, values or links', () => {
  const localize = createLocalizer(translations, pack.legacyTitles);
  for (const id of Object.keys(pack.legacyTitles)) {
    const node = makeNode(id);
    const stable = () => JSON.stringify({
      type: node.type,
      widgets: node.widgets.map(({ name, value, options }) => ({ name, value, options })),
      inputs: node.inputs.map(({ name, link }) => ({ name, link })),
      outputs: node.outputs.map(({ name, links }) => ({ name, links })),
    });
    const before = stable();
    for (const locale of ['en', 'zh', 'en']) {
      localize(node, locale);
      const def = translations[locale].nodeDefs[id];
      assert.equal(node.title, def.display_name);
      for (const widget of node.widgets) {
        assert.equal(widget.label, def.inputs[widget.name].name);
        assert.equal(widget.tooltip, def.inputs[widget.name].tooltip);
        for (const value of widget.options.values) {
          assert.equal(widget.options.getOptionLabel(value), def.inputs[widget.name].options[value]);
        }
      }
      for (const [index, output] of node.outputs.entries()) {
        assert.equal(output.label, def.outputs[index].name);
      }
      assert.equal(stable(), before);
    }
  }
});

test('custom titles survive switching before and after auto-localization', () => {
  const localize = createLocalizer(translations, pack.legacyTitles);
  const id = Object.keys(pack.legacyTitles)[0];
  const node = makeNode(id);
  node.title = 'My camera workflow';
  localize(node, 'zh');
  assert.equal(node.title, 'My camera workflow');
  node.title = pack.legacyTitles[id];
  localize(node, 'en');
  node.title = 'Custom second title';
  localize(node, 'zh');
  assert.equal(node.title, 'Custom second title');
});

test('unprovided locales use English and other packs remain untouched', () => {
  const localize = createLocalizer(translations, pack.legacyTitles);
  const id = Object.keys(pack.legacyTitles)[0];
  const node = makeNode(id);
  localize(node, 'fr');
  assert.equal(node.title, translations.en.nodeDefs[id].display_name);
  localize(node, 'zh-CN');
  assert.equal(node.title, translations.zh.nodeDefs[id].display_name);
  const other = { type: 'KSampler', title: 'My sampler', inputs: [{ name: 'seed' }] };
  const before = JSON.stringify(other);
  localize(other, 'zh');
  assert.equal(JSON.stringify(other), before);
});

test('a missing Chinese definition falls back to English', () => {
  const id = Object.keys(pack.legacyTitles)[0];
  const node = makeNode(id);
  createLocalizer({ en: translations.en }, pack.legacyTitles)(node, 'zh');
  assert.equal(node.title, translations.en.nodeDefs[id].display_name);
});

test('extension startup, locale events and workflow reload localize live nodes', async () => {
  const { createLocalizer } = await import('../web/localization.js');
  const source = await readFile(new URL('../web/i18n.js', import.meta.url), 'utf8');
  let extension;
  const settings = new EventTarget();
  settings.getSettingValue = () => 'en';
  const app = { ui: { settings }, registerExtension(value) { extension = value; } };
  const api = { fetchApi: async () => ({ ok: true, json: async () => translations }) };
  const fetch = async () => ({ ok: true, json: async () => pack });
  const body = source.replace(/^import .*;\r?\n/gm, '').replaceAll('import.meta.url', '"https://example.test/extensions/pack/i18n.js"');
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  await new AsyncFunction('app', 'api', 'createLocalizer', 'fetch', 'requestAnimationFrame', body)(app, api, createLocalizer, fetch, (fn) => fn());
  const node = makeNode(Object.keys(pack.legacyTitles)[0]);
  extension.nodeCreated(node);
  await extension.setup();
  assert.equal(node.title, translations.en.nodeDefs[node.type].display_name);
  settings.dispatchEvent(new CustomEvent('Comfy.Locale.change', { detail: { value: 'zh' } }));
  assert.equal(node.title, translations.zh.nodeDefs[node.type].display_name);
  node.title = pack.legacyTitles[node.type];
  extension.loadedGraphNode(node);
  assert.equal(node.title, translations.zh.nodeDefs[node.type].display_name);
  node.onRemoved();
  settings.dispatchEvent(new CustomEvent('Comfy.Locale.change', { detail: { value: 'en' } }));
  assert.equal(node.title, translations.zh.nodeDefs[node.type].display_name);
});
