import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { betterFetch } from "@better-fetch/fetch";
import type { Session } from "better-auth/types";

type AuthSession = {
  session: {
    id: string;
    expiresAt: Date;
    userId: string;
  };
  user: {
    id: string;
    email: string;
    name: string;
    role?: string;
  };
};

const AUTH_SERVER_URL =
  process.env.AUTH_SERVER_INTERNAL_URL || "http://ragbot_auth_server:4000";

export async function middleware(request: NextRequest) {
  const url = request.nextUrl;
  const hostname = request.headers.get("host") || "";

  // Exclude static files and API routes from subdomain routing interference
  if (
    url.pathname.startsWith("/_next") ||
    url.pathname.startsWith("/api") ||
    url.pathname.includes(".") // e.g. favicon.ico
  ) {
    return NextResponse.next();
  }

  // --- ADMIN SUBDOMAIN LOGIC ---
  if (hostname === "admin-gp.innovin.win") {
    // Allow access to the admin login page without an existing session.
    if (url.pathname === "/login" || url.pathname.startsWith("/login/")) {
      return NextResponse.rewrite(new URL(`/admin${url.pathname}`, request.url));
    }

    const { data: session } = await betterFetch<AuthSession>(
      `${AUTH_SERVER_URL}/api/auth/get-session`,
      {
        headers: {
          cookie: request.headers.get("cookie") || "",
        },
      }
    );

    if (!session || session.user.role !== "admin") {
      // Not logged in (or not an admin): send to admin login page.
      const loginUrl = new URL("/login", request.url);
      loginUrl.searchParams.set("next", url.pathname + url.search);
      return NextResponse.redirect(loginUrl);
    }

    // Rewrite to the explicit /admin folder
    return NextResponse.rewrite(new URL(`/admin${url.pathname}`, request.url));
  }

  // --- CUSTOMER DOMAIN LOGIC ---
  return NextResponse.rewrite(new URL(`/user${url.pathname}`, request.url));
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};