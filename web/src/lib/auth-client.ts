import { createAuthClient } from "better-auth/react";

export const authClient = createAuthClient({
  // Use same-origin auth routes; Caddy proxies `/api/auth/*` to the auth-server.
  // This works for both the main domain and `admin.*` subdomain and ensures cookies
  // are set for the active host.
  basePath: "/api/auth",
});

export const { signIn, signUp, signOut, useSession } = authClient;
