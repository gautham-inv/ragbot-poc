import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { betterFetch } from "@better-fetch/fetch";

type AuthSession = {
  user?: {
    role?: string;
  };
};

const AUTH_SERVER_URL =
  process.env.AUTH_SERVER_INTERNAL_URL || "http://auth-server:4000";

export async function middleware(request: NextRequest) {
  const url = request.nextUrl;

  if (
    url.pathname.startsWith("/_next") ||
    url.pathname.startsWith("/api") ||
    url.pathname.includes(".")
  ) {
    return NextResponse.next();
  }

  // Allow access to the login page without an existing session.
  if (url.pathname === "/login" || url.pathname.startsWith("/login/")) {
    return NextResponse.next();
  }

  // Allow access to sign up without an existing session.
  if (url.pathname === "/signup" || url.pathname.startsWith("/signup/")) {
    return NextResponse.next();
  }

  const { data: session } = await betterFetch<AuthSession>(
    `${AUTH_SERVER_URL}/api/auth/get-session`,
    {
      headers: {
        cookie: request.headers.get("cookie") || "",
      },
    }
  );

  const isAdmin = Boolean(session?.user?.role === "admin");
  if (!isAdmin) {
    if (url.pathname.startsWith("/admin-api/")) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("next", url.pathname + url.search);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};
