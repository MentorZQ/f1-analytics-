const KEY_STORAGE = "f1rag.anthropicKey";

// Browser storage can throw in private windows; the key then lasts for the page only.
let memoryKey: string | null = null;

export function getKey(): string | null {
  try {
    return sessionStorage.getItem(KEY_STORAGE) || memoryKey;
  } catch {
    return memoryKey;
  }
}

export function setKey(value: string | null): void {
  memoryKey = value;
  try {
    if (value) sessionStorage.setItem(KEY_STORAGE, value);
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch {
    // Kept in memoryKey only.
  }
}
