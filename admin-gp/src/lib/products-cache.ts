/**
 * sessionStorage-backed cache for the products list and individual products.
 *
 * Goals:
 *   - Navigating list → detail → back doesn't refetch.
 *   - Cache survives in-tab navigation but not new tabs / new sessions.
 *   - Writes (create/update/delete/image change) invalidate the relevant keys.
 *
 * 5-minute TTL keeps content reasonably fresh; the user can also force a refresh
 * via the "Refresh" button on the list page.
 */

const TTL_MS = 5 * 60 * 1000;
const PREFIX = "products-cache:";

type Wrapped<T> = { value: T; storedAt: number };

function safe<T>(fn: () => T, fallback: T): T {
  try {
    if (typeof window === "undefined") return fallback;
    return fn();
  } catch {
    return fallback;
  }
}

export function getCached<T>(key: string): T | null {
  return safe(() => {
    const raw = sessionStorage.getItem(PREFIX + key);
    if (!raw) return null;
    const c: Wrapped<T> = JSON.parse(raw);
    if (Date.now() - c.storedAt > TTL_MS) {
      sessionStorage.removeItem(PREFIX + key);
      return null;
    }
    return c.value;
  }, null);
}

export function setCached<T>(key: string, value: T): void {
  safe(() => {
    sessionStorage.setItem(
      PREFIX + key,
      JSON.stringify({ value, storedAt: Date.now() })
    );
    return null;
  }, null);
}

export function invalidate(key?: string): void {
  safe(() => {
    if (key) {
      sessionStorage.removeItem(PREFIX + key);
      return null;
    }
    // Clear the whole namespace (used after writes — list AND detail go stale).
    for (let i = sessionStorage.length - 1; i >= 0; i--) {
      const k = sessionStorage.key(i);
      if (k && k.startsWith(PREFIX)) sessionStorage.removeItem(k);
    }
    return null;
  }, null);
}

export const KEYS = {
  list: "list",
  detail: (sku: string) => `detail:${sku}`,
};
