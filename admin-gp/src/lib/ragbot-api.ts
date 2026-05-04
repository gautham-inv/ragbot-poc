/**
 * Server-side helper for proxying admin requests to ragbot-api (FastAPI).
 *
 * All callers MUST be server-side (Route Handlers) — the cookie must travel
 * from the user's browser through Next.js to ragbot-api so FastAPI can
 * re-verify the Better Auth session as defense-in-depth.
 */

const INTERNAL_URL =
  process.env.RAGBOT_API_INTERNAL_URL || "http://ragbot-api:8000";

export async function ragbotApi(
  path: string,
  init: RequestInit & { cookie?: string } = {}
): Promise<Response> {
  const { cookie, headers, ...rest } = init;
  return fetch(`${INTERNAL_URL}${path}`, {
    ...rest,
    headers: {
      ...(headers || {}),
      ...(cookie ? { cookie } : {}),
    },
    cache: "no-store",
  });
}
