import { useSyncExternalStore } from "react";
import type { MemoryConfig } from "./api";

const KEY = "lightmem.console.draft";

let draft: MemoryConfig | null = readInitial();
const listeners = new Set<() => void>();

function readInitial(): MemoryConfig | null {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as MemoryConfig) : null;
  } catch {
    return null;
  }
}

function emit() {
  listeners.forEach((fn) => fn());
}

export function setDraft(next: MemoryConfig | null) {
  draft = next;
  try {
    if (next) localStorage.setItem(KEY, JSON.stringify(next));
    else localStorage.removeItem(KEY);
  } catch {
    /* Use the in-memory value when storage is unavailable. */
  }
  emit();
}

export function getDraft(): MemoryConfig | null {
  return draft;
}

function subscribe(fn: () => void) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function useDraft(): MemoryConfig | null {
  return useSyncExternalStore(subscribe, getDraft, getDraft);
}
