// free-model-aggregator-01 entry: one server, two client wires, ten free providers.
//
// Serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and SSE) by
// routing each request to the highest-rated FREE model that meets the request's requirements,
// failing over to the next-best free model when a provider refuses (402 / 429 / 5xx).
//
// How it works
//   - Reads the brief at $MAKEFUN_PROMPT_FILE: the provider directory, the sealed quality
//     table (ratings, capabilities, free flags) and the workload manifest. The catalog is built
//     from the brief alone, so `node main.ts --list` and the register body are the same document,
//     byte for byte, with no network dependency at build time.
//   - Registers {protocol_version: 1, port, catalog} at POST $MAKEFUN_ENV_URL/register, serves
//     the client's 20 requests one at a time, and polls GET $MAKEFUN_ENV_URL/work until it says
//     {"done": true}, then exits 0.
//   - Routing: every free model meeting the request's requirements (tools / vision / JSON /
//     min_context) is a candidate, ranked by the quality table's rating_milli, highest first.
//     A JSON-mode request from an OpenAI client is only routed to a wire that can carry
//     response_format (openai or gemini); the Anthropic wire cannot echo JSON mode.
//   - The provider's answer text is passed through whole (it carries the echo the client
//     checks), with usage counts and the stop reason translated into the client's wire.
//   - Deterministic: no wall clock, no unseeded randomness, fixed ids and timestamps. Every
//     choice is a function of the brief, the requests and the providers' answers so far.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider call may take, in milliseconds. */
const PROVIDER_CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Fixed epoch second for every OpenAI-shaped id/created: nothing here reads the clock. */
const FIXED_CREATED = 1767225600;
/** Fixed instant for every Anthropic-shaped created_at. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** SSE text pieces are cut to this many characters, as the providers cut theirs. */
const SSE_DELTA_CHARS = 13;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief
// ---------------------------------------------------------------------------

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

interface ProviderDirectoryEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: ProviderDirectoryEntry[];
  ratings: RatingRow[];
  workload: WorkloadRequest[];
  error: string | null;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function providerOf(v: unknown): ProviderDirectoryEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    stringOf(v.path) !== ''
      ? stringOf(v.path)
      : stringOf(v.base_path) !== ''
        ? stringOf(v.base_path)
        : stringOf(v.base_url) !== ''
          ? stringOf(v.base_url)
          : stringOf(v.base);
  return { id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)
        ? { min_context: requires.min_context }
        : {}),
    },
  };
}

/** The brief at `path`: its three documents, found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload =
    workloadDoc !== null && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const w = workloadRequestOf(v);
          return w === null ? [] : [w];
        })
      : [];
  return { providers, ratings, workload, error: null };
}

// ---------------------------------------------------------------------------
// The catalog (built from the brief alone: --list and the register body agree)
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

/**
 * One catalog entry per provider, in the brief's order, each model in the quality table's
 * order (the sealed free-model order). base_url is the directory's own path (/p/<id>), never
 * joined to the env URL: the env's origin differs between a live pass and its replay, and the
 * catalog is part of the state export a replay must reproduce byte for byte. The quality
 * table lists free models only, so the catalog lists exactly the free models; a paid model
 * is never listed as free.
 */
function buildCatalog(brief: Brief): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const p of brief.providers) {
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const row of brief.ratings) {
      if (row.provider !== p.id || !row.free || seen.has(row.model)) continue;
      seen.add(row.model);
      models.push({ id: row.model, free: true, context_length: row.context_length });
    }
    providers.push({ id: p.id, base_url: p.base, key_env: p.key_env, wire: p.wire, models });
  }
  return { providers };
}

const brief = readBrief(PROMPT_FILE);
if (brief.error !== null) log(brief.error);
const catalog: CatalogReport = buildCatalog(brief);
const providerById = new Map<string, ProviderDirectoryEntry>(brief.providers.map((p) => [p.id, p]));
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;

/** The free models this entry serves, in catalog order, with their capabilities. */
function servedModels(): { id: string; provider: string; display_name: string; context_length: number | null; tools: boolean; vision: boolean; json: boolean }[] {
  const out: { id: string; provider: string; display_name: string; context_length: number | null; tools: boolean; vision: boolean; json: boolean }[] = [];
  for (const p of catalog.providers) {
    for (const m of p.models) {
      const row = brief.ratings.find((r) => r.provider === p.id && r.model === m.id);
      out.push({
        id: m.id,
        provider: p.id,
        display_name: m.id,
        context_length: m.context_length,
        tools: row?.tools ?? false,
        vision: row?.vision ?? false,
        json: row?.json ?? false,
      });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Routing: rank the free candidates by rating, fail over on a refusal
// ---------------------------------------------------------------------------

interface Candidate {
  providerId: string;
  modelId: string;
  rating: number;
  providerIndex: number;
}

/**
 * Every free model meeting the request's requirements, best rating first. Ties break by the
 * provider's place in the brief, then by model id: deterministic, no clock, no randomness.
 * When the client asked for JSON mode on the OpenAI wire, only a wire that can carry
 * response_format (openai or gemini) is a candidate: the Anthropic wire echoes json_mode
 * absent, which would fail the client's echo check.
 */
function candidatesFor(requires: Requirement, needsJsonWire: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const row of brief.ratings) {
    if (!row.free) continue;
    if (requires.tools === true && !row.tools) continue;
    if (requires.vision === true && !row.vision) continue;
    if (requires.json === true && !row.json) continue;
    if (requires.min_context !== undefined && (row.context_length === null || row.context_length < requires.min_context)) continue;
    const provider = providerById.get(row.provider);
    if (provider === undefined) continue;
    if (needsJsonWire && provider.wire === 'anthropic') continue;
    out.push({
      providerId: row.provider,
      modelId: row.model,
      rating: row.rating_milli,
      providerIndex: brief.providers.findIndex((p) => p.id === row.provider),
    });
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      a.providerIndex - b.providerIndex ||
      (a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.workload[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

// ---------------------------------------------------------------------------
// The request, normalized: what the provider must echo back
// ---------------------------------------------------------------------------

type ImageRef = { kind: 'data'; mime: string; data: string } | { kind: 'url'; url: string };

interface ToolDef {
  name: string;
  description: string;
  parameters: Json;
}

interface NormRequest {
  systemText: string;
  userText: string;
  images: ImageRef[];
  tools: ToolDef[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
  temperature: number | undefined;
  topP: number | undefined;
  toolChoice: unknown;
}

/** The text of a string or of a block array's text blocks, joined as the providers join them. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** One image reference from a data: URL or a plain URL; null when there is no image. */
function imageRefOfUrl(url: unknown): ImageRef | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { kind: 'data', mime: data[1] !== '' ? data[1] : 'image/png', data: data[2] };
  return { kind: 'url', url };
}

/** An OpenAI Chat Completions body, normalized. */
function normFromOpenAI(body: Json): NormRequest {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = '';
  const images: ImageRef[] = [];
  for (const message of messages) {
    if (message.role === 'system') {
      const text = textOf(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user') {
      userText = textOf(message.content);
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'image_url' && isRecord(block.image_url)) {
            const ref = imageRefOfUrl(block.image_url.url);
            if (ref !== null) images.push(ref);
          }
        }
      }
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const maxTokens =
    typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens)
      ? body.max_tokens
      : typeof body.max_completion_tokens === 'number' && Number.isFinite(body.max_completion_tokens)
        ? body.max_completion_tokens
        : null;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    userText,
    images,
    tools,
    maxTokens,
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    toolChoice: body.tool_choice,
  };
}

/** An Anthropic Messages body, normalized. The Anthropic wire carries no JSON mode. */
function normFromAnthropic(body: Json): NormRequest {
  const systemText = textOf(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = '';
  const images: ImageRef[] = [];
  for (const message of messages) {
    if (message.role !== 'user') continue;
    userText = textOf(message.content);
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          images.push({
            kind: 'data',
            mime: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png',
            data: block.source.data,
          });
        } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
          const ref = imageRefOfUrl(block.source.url);
          if (ref !== null) images.push(ref);
        }
      }
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return {
    systemText,
    userText,
    images,
    tools,
    maxTokens: typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens) ? body.max_tokens : null,
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    toolChoice: body.tool_choice,
  };
}

// ---------------------------------------------------------------------------
// Upstream requests: the same normalized request in each provider's wire
// ---------------------------------------------------------------------------

/** A base path joined to the env's URL; an absolute base is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function imageUrlOf(ref: ImageRef): string {
  return ref.kind === 'data' ? `data:${ref.mime};base64,${ref.data}` : ref.url;
}

/** The client's tool_choice, in the OpenAI wire's spelling. */
function openAIToolChoice(choice: unknown, hasTools: boolean): unknown {
  if (!hasTools) return undefined;
  if (typeof choice === 'string') return choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') return 'auto';
    if (choice.type === 'any') return 'required';
    if (choice.type === 'none') return 'none';
    if (choice.type === 'tool' && typeof choice.name === 'string') return { type: 'function', function: { name: choice.name } };
  }
  return 'auto';
}

/** The normalized request as an OpenAI Chat Completions body naming `model`. */
function openAIUpstream(model: string, norm: NormRequest): Json {
  const messages: Json[] = [];
  if (norm.systemText !== '') messages.push({ role: 'system', content: norm.systemText });
  if (norm.images.length === 0) {
    messages.push({ role: 'user', content: norm.userText });
  } else {
    const parts: Json[] = [{ type: 'text', text: norm.userText }];
    for (const ref of norm.images) parts.push({ type: 'image_url', image_url: { url: imageUrlOf(ref) } });
    messages.push({ role: 'user', content: parts });
  }
  const body: Json = { model, messages };
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.temperature !== undefined) body.temperature = norm.temperature;
  if (norm.topP !== undefined) body.top_p = norm.topP;
  if (norm.stops.length > 0) body.stop = norm.stops;
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
    const choice = openAIToolChoice(norm.toolChoice, true);
    if (choice !== undefined) body.tool_choice = choice;
  }
  if (norm.jsonMode) body.response_format = { type: 'json_object' };
  if (norm.stream) {
    body.stream = true;
    body.stream_options = { include_usage: true };
  }
  return body;
}

/** The normalized request as an Anthropic Messages body naming `model`. */
function anthropicUpstream(model: string, norm: NormRequest): Json {
  const parts: Json[] = [{ type: 'text', text: norm.userText }];
  for (const ref of norm.images) {
    if (ref.kind === 'data') {
      parts.push({ type: 'image', source: { type: 'base64', media_type: ref.mime, data: ref.data } });
    } else {
      parts.push({ type: 'image', source: { type: 'url', url: ref.url } });
    }
  }
  const body: Json = { model, messages: [{ role: 'user', content: norm.images.length === 0 ? norm.userText : parts }] };
  if (norm.systemText !== '') body.system = norm.systemText;
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.temperature !== undefined) body.temperature = norm.temperature;
  if (norm.topP !== undefined) body.top_p = norm.topP;
  if (norm.stops.length > 0) body.stop_sequences = norm.stops;
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
    if (isRecord(norm.toolChoice)) {
      if (norm.toolChoice.type === 'auto') body.tool_choice = { type: 'auto' };
      else if (norm.toolChoice.type === 'any') body.tool_choice = { type: 'any' };
      else if (norm.toolChoice.type === 'tool' && typeof norm.toolChoice.name === 'string') {
        body.tool_choice = { type: 'tool', name: norm.toolChoice.name };
      }
    }
  }
  if (norm.stream) body.stream = true;
  return body;
}

/** The normalized request as a Gemini generateContent body. */
function geminiUpstream(norm: NormRequest): Json {
  const parts: Json[] = [{ text: norm.userText }];
  for (const ref of norm.images) {
    if (ref.kind === 'data') parts.push({ inlineData: { mimeType: ref.mime, data: ref.data } });
    else parts.push({ fileData: { fileUri: ref.url } });
  }
  const body: Json = { contents: [{ role: 'user', parts }] };
  if (norm.systemText !== '') body.systemInstruction = { parts: [{ text: norm.systemText }] };
  if (norm.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: norm.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      },
    ];
  }
  const config: Json = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (norm.temperature !== undefined) config.temperature = norm.temperature;
  if (norm.topP !== undefined) config.topP = norm.topP;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

/**
 * The upstream body for one candidate. When the provider speaks the client's own wire, the
 * client's body passes through with only the model replaced (and include_usage set on an
 * OpenAI stream, so the provider's stream carries the usage the client checks for): the
 * provider then echoes exactly what the client sent.
 */
function upstreamBody(provider: ProviderDirectoryEntry, model: string, norm: NormRequest, clientBody: Json, clientApi: 'openai' | 'anthropic'): Json {
  if (clientApi === 'openai' && provider.wire === 'openai') {
    const body: Json = { ...clientBody, model };
    if (norm.stream) body.stream_options = { include_usage: true };
    return body;
  }
  if (clientApi === 'anthropic' && provider.wire === 'anthropic') {
    return { ...clientBody, model };
  }
  if (provider.wire === 'openai') return openAIUpstream(model, norm);
  if (provider.wire === 'anthropic') return anthropicUpstream(model, norm);
  return geminiUpstream(norm);
}

/** One POST to a provider's inference route. */
async function callProvider(provider: ProviderDirectoryEntry, model: string, body: Json, stream: boolean): Promise<Response> {
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
  } else {
    url = `${base}/v1beta/models/${model}:${stream ? 'streamGenerateContent' : 'generateContent'}${stream ? '?alt=sse' : ''}`;
  }
  if (stream) headers.accept = 'text/event-stream';
  return fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(PROVIDER_CALL_TIMEOUT_MS),
  });
}

// ---------------------------------------------------------------------------
// Provider answers, parsed: text, stop reason, usage
// ---------------------------------------------------------------------------

interface UpstreamAnswer {
  text: string;
  stop: string;
  input: number;
  output: number;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of an OpenAI message: a string, or its text blocks joined. */
function openAIText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** One provider JSON answer, in its wire. */
function parseUpstreamJson(wire: Wire, value: unknown): UpstreamAnswer | null {
  if (!isRecord(value)) return null;
  if (wire === 'openai') {
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: openAIText(message.content),
      stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
      input: numberOrZero(usage.prompt_tokens),
      output: numberOrZero(usage.completion_tokens),
    };
  }
  if (wire === 'anthropic') {
    const content = Array.isArray(value.content) ? value.content : [];
    const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text,
      stop: typeof value.stop_reason === 'string' ? value.stop_reason : 'end_turn',
      input: numberOrZero(usage.input_tokens),
      output: numberOrZero(usage.output_tokens),
    };
  }
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
  return {
    text,
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
    input: numberOrZero(usage.promptTokenCount),
    output: numberOrZero(usage.candidatesTokenCount),
  };
}

/** SSE frames: each frame's event name (or null) and its data lines joined. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** One provider SSE answer, in its wire. */
function parseUpstreamStream(wire: Wire, text: string): UpstreamAnswer | null {
  let out = '';
  let stop = wire === 'openai' ? 'stop' : wire === 'anthropic' ? 'end_turn' : 'STOP';
  let input = 0;
  let output = 0;
  let sawAnything = false;
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    sawAnything = true;
    if (wire === 'openai') {
      if (isRecord(chunk.usage)) {
        input = numberOrZero(chunk.usage.prompt_tokens);
        output = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') out += delta.content;
      if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
    } else if (wire === 'anthropic') {
      if (frame.event === 'message_start' && isRecord(chunk.message) && isRecord(chunk.message.usage)) {
        input = numberOrZero(chunk.message.usage.input_tokens);
      }
      if (frame.event === 'content_block_delta' && isRecord(chunk.delta) && chunk.delta.type === 'text_delta' && typeof chunk.delta.text === 'string') {
        out += chunk.delta.text;
      }
      if (frame.event === 'message_delta') {
        if (isRecord(chunk.delta) && typeof chunk.delta.stop_reason === 'string') stop = chunk.delta.stop_reason;
        if (isRecord(chunk.usage)) output = numberOrZero(chunk.usage.output_tokens);
      }
    } else {
      const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
      const candidate = isRecord(candidates[0]) ? candidates[0] : null;
      if (candidate !== null) {
        const content = isRecord(candidate.content) ? candidate.content : {};
        const parts = Array.isArray(content.parts) ? content.parts : [];
        for (const part of parts) if (isRecord(part) && typeof part.text === 'string') out += part.text;
        if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
      }
      if (isRecord(chunk.usageMetadata)) {
        input = numberOrZero(chunk.usageMetadata.promptTokenCount);
        output = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
      }
    }
  }
  if (!sawAnything && out === '') return null;
  return { text: out, stop, input, output };
}

function parseUpstreamAnswer(wire: Wire, text: string, stream: boolean): UpstreamAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    return parseUpstreamJson(wire, value);
  }
  return parseUpstreamStream(wire, text);
}

// ---------------------------------------------------------------------------
// Client answers: the provider's text, usage and stop reason in the client's wire
// ---------------------------------------------------------------------------

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function stopReasonAnthropic(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** Any wire's stop reason as OpenAI's finish_reason. */
function stopReasonOpenAI(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_DELTA_CHARS) pieces.push(text.slice(at, at + SSE_DELTA_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** An OpenAI Chat Completions answer (JSON or SSE) from a provider's answer. */
function openAIClientAnswer(answer: UpstreamAnswer, model: string, stream: boolean): string {
  const finish = stopReasonOpenAI(answer.stop);
  if (!stream) {
    return JSON.stringify({
      id: 'chatcmpl-agg',
      object: 'chat.completion',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: finish, logprobs: null }],
      usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
    });
  }
  const chunk = (delta: Json, finishReason: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });
  let out = sseEvent(null, chunk({ role: 'assistant', content: '' }, null));
  for (const piece of splitText(answer.text)) out += sseEvent(null, chunk({ content: piece }, null));
  out += sseEvent(null, chunk({}, finish));
  out += sseEvent(null, {
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  });
  out += 'data: [DONE]\n\n';
  return out;
}

/** An Anthropic Messages answer (JSON or SSE) from a provider's answer. */
function anthropicClientAnswer(answer: UpstreamAnswer, model: string, stream: boolean): string {
  const stop = stopReasonAnthropic(answer.stop);
  if (!stream) {
    return JSON.stringify({
      id: 'msg_agg',
      type: 'message',
      role: 'assistant',
      model,
      content: [{ type: 'text', text: answer.text }],
      stop_reason: stop,
      stop_sequence: null,
      usage: { input_tokens: answer.input, output_tokens: answer.output },
    });
  }
  let out = sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_agg',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.input, output_tokens: 0 },
    },
  });
  out += sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const piece of splitText(answer.text)) {
    out += sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } });
  }
  out += sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  out += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stop, stop_sequence: null },
    usage: { output_tokens: answer.output },
  });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}

// ---------------------------------------------------------------------------
// Serving one request: best free model first, fail over on a refusal
// ---------------------------------------------------------------------------

interface ServedResponse {
  status: number;
  contentType: string;
  body: string;
}

/**
 * Serve one client request. Tries the candidates in rating order; a provider that answers
 * 402, 429 or a 5xx (or is unreachable, or answers unparseably) costs one call and the
 * request moves to the next-best free model. Exactly one provider call may succeed: the
 * first success is the answer, and no further provider is tried. A refusal is never
 * retried on the same provider for this request: the failure schedule is per call number,
 * so a retry would spend another call on the same provider instead of failing over.
 */
async function serveRequest(api: 'openai' | 'anthropic', body: Json): Promise<ServedResponse | null> {
  const norm = api === 'openai' ? normFromOpenAI(body) : normFromAnthropic(body);
  const requires = requirementsFor(body);
  const needsJsonWire = api === 'openai' && norm.jsonMode;
  const candidates = candidatesFor(requires, needsJsonWire);
  for (const candidate of candidates) {
    const provider = providerById.get(candidate.providerId);
    if (provider === undefined) continue;
    const upstream = upstreamBody(provider, candidate.modelId, norm, body, api);
    let response: Response;
    try {
      response = await callProvider(provider, candidate.modelId, upstream, norm.stream);
    } catch (err) {
      log(`provider ${provider.id}/${candidate.modelId} unreachable: ${(err as Error).message}`);
      continue;
    }
    const text = await response.text().catch(() => '');
    if (response.status < 200 || response.status >= 300) {
      log(`provider ${provider.id}/${candidate.modelId} answered HTTP ${response.status}`);
      continue;
    }
    // Same wire, JSON: the provider's answer is already the client's shape; forward it whole.
    if (!norm.stream && provider.wire === api) {
      return { status: 200, contentType: 'application/json', body: text };
    }
    // Same wire, SSE: the provider's stream is already the client's shape; forward it whole.
    if (norm.stream && provider.wire === api) {
      return { status: 200, contentType: 'text/event-stream; charset=utf-8', body: text };
    }
    const answer = parseUpstreamAnswer(provider.wire, text, norm.stream);
    if (answer === null) {
      log(`provider ${provider.id}/${candidate.modelId} answered unparseably`);
      continue;
    }
    if (api === 'openai') {
      return { status: 200, contentType: norm.stream ? 'text/event-stream; charset=utf-8' : 'application/json', body: openAIClientAnswer(answer, candidate.modelId, norm.stream) };
    }
    return { status: 200, contentType: norm.stream ? 'text/event-stream; charset=utf-8' : 'application/json', body: anthropicClientAnswer(answer, candidate.modelId, norm.stream) };
  }
  return null;
}

// ---------------------------------------------------------------------------
// The HTTP server
// ---------------------------------------------------------------------------

/** GET /v1/models on an OpenAI-shaped request: the free models this entry serves. */
function openAIModelList(): Json {
  return {
    object: 'list',
    data: servedModels().map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_CREATED,
      owned_by: m.provider,
      free: true,
      context_length: m.context_length,
      tools: m.tools,
      vision: m.vision,
      json: m.json,
    })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
function anthropicModelList(): Json {
  return {
    data: servedModels().map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.display_name,
      created_at: FIXED_ANTHROPIC_CREATED_AT,
      free: true,
      context_length: m.context_length,
      tools: m.tools,
      vision: m.vision,
      json: m.json,
    })),
    has_more: false,
  };
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: null } });
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendServed(res: ServerResponse, served: ServedResponse): void {
  const headers: Record<string, string> = { 'content-type': served.contentType, 'content-length': String(Buffer.byteLength(served.body)) };
  if (served.contentType.startsWith('text/event-stream')) {
    headers['cache-control'] = 'no-cache';
    headers.connection = 'keep-alive';
  }
  res.writeHead(served.status, headers);
  res.end(served.body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList());
    else sendJson(res, 200, openAIModelList());
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    const served = await serveRequest('openai', body);
    if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model could serve this request');
    sendServed(res, served);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    const served = await serveRequest('anthropic', body);
    if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model could serve this request');
    sendServed(res, served);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  // --list prints the same catalog the register body carries, byte for byte, and stops
  // before any server starts. With no brief to read it prints an empty catalog.
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (brief.error !== null) throw new Error(`the brief could not be read: ${brief.error}`);
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
