import { useSyncExternalStore } from "react";

export type Prefs = {
  topK: number;
  keepTopN: number;
  scoreThreshold: number;
  retrieveLimit: number;
};

export const DEFAULT_PREFS: Prefs = {
  topK: 20,
  keepTopN: 10,
  scoreThreshold: 0.8,
  retrieveLimit: 10,
};

const KEY = "lightmem.console.prefs";

function read(): Prefs {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULT_PREFS };
    return { ...DEFAULT_PREFS, ...(JSON.parse(raw) as Partial<Prefs>) };
  } catch {
    return { ...DEFAULT_PREFS };
  }
}

let prefs: Prefs = read();
const listeners = new Set<() => void>();

export function setPrefs(patch: Partial<Prefs>) {
  prefs = { ...prefs, ...patch };
  try {
    localStorage.setItem(KEY, JSON.stringify(prefs));
  } catch {
    /* Use the in-memory value when storage is unavailable. */
  }
  listeners.forEach((fn) => fn());
}

export function getPrefs(): Prefs {
  return prefs;
}

function subscribe(fn: () => void) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function usePrefs(): Prefs {
  return useSyncExternalStore(subscribe, getPrefs, getPrefs);
}
