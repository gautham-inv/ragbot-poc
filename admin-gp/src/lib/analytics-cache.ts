/**
 * sessionStorage cache for the analytics dashboard payload, keyed by range.
 *
 * 2-minute TTL — short enough that the dashboard feels fresh, long enough that
 * navigating away and back doesn't trigger a re-aggregate. The Refresh button
 * forces a bypass.
 */

const TTL_MS = 2 * 60 * 1000;
const PREFIX = "analytics-cache:";

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
    for (let i = sessionStorage.length - 1; i >= 0; i--) {
      const k = sessionStorage.key(i);
      if (k && k.startsWith(PREFIX)) sessionStorage.removeItem(k);
    }
    return null;
  }, null);
}

export function cacheAge(key: string): number | null {
  return safe(() => {
    const raw = sessionStorage.getItem(PREFIX + key);
    if (!raw) return null;
    const c: Wrapped<unknown> = JSON.parse(raw);
    return Date.now() - c.storedAt;
  }, null);
}
