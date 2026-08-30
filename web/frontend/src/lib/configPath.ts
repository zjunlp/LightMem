import type { MemoryConfig, UiField } from "./api";

export function getPath(config: MemoryConfig, path: string): any {
  return path
    .split(".")
    .reduce<any>((node, key) => (node == null ? undefined : node[key]), config);
}

export function setPath(
  config: MemoryConfig,
  path: string,
  value: unknown,
): MemoryConfig {
  const keys = path.split(".");
  const next = { ...config };
  let node: any = next;
  for (let i = 0; i < keys.length - 1; i += 1) {
    const key = keys[i];
    node[key] =
      typeof node[key] === "object" && node[key] !== null
        ? { ...node[key] }
        : {};
    node = node[key];
  }
  node[keys[keys.length - 1]] = value;
  return next;
}

export function deletePath(config: MemoryConfig, path: string): MemoryConfig {
  const keys = path.split(".");
  const next = structuredClone(config);
  const stack: any[] = [next];
  let node: any = next;
  for (let i = 0; i < keys.length - 1; i += 1) {
    node = node?.[keys[i]];
    if (node == null) return next;
    stack.push(node);
  }
  delete node[keys[keys.length - 1]];
  for (let i = stack.length - 1; i > 0; i -= 1) {
    if (Object.keys(stack[i]).length === 0) delete stack[i - 1][keys[i - 1]];
    else break;
  }
  return next;
}

export function shouldShow(field: UiField, config: MemoryConfig): boolean {
  if (!field.showIf) return true;
  const [path, expected] = field.showIf.split("==");
  const actual = getPath(config, path.trim());
  if (expected === undefined) return Boolean(actual);
  return String(actual) === expected.trim();
}

export function coerce(field: UiField, raw: string | boolean): unknown {
  if (field.kind === "bool") return Boolean(raw);
  if (raw === "") return undefined;
  if (field.kind === "int") {
    const n = Number.parseInt(String(raw), 10);
    return Number.isNaN(n) ? undefined : n;
  }
  if (field.kind === "number") {
    const n = Number.parseFloat(String(raw));
    return Number.isNaN(n) ? undefined : n;
  }
  return raw;
}

export function prune(value: any): any {
  if (Array.isArray(value)) return value.map(prune);
  if (value && typeof value === "object") {
    const out: Record<string, any> = {};
    for (const [k, v] of Object.entries(value)) {
      const cleaned = prune(v);
      if (cleaned === undefined || cleaned === null || cleaned === "") continue;
      if (
        typeof cleaned === "object" &&
        !Array.isArray(cleaned) &&
        Object.keys(cleaned).length === 0
      )
        continue;
      out[k] = cleaned;
    }
    return out;
  }
  return value;
}

export function stripSecrets(value: any): any {
  if (Array.isArray(value)) return value.map(stripSecrets);
  if (value && typeof value === "object") {
    const out: Record<string, any> = {};
    for (const [k, v] of Object.entries(value)) {
      if (k === "api_key") continue;
      out[k] = stripSecrets(v);
    }
    return out;
  }
  return value;
}
