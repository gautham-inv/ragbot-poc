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
  if (hostname.startsWith("admin.")) {
    const { data: session } = await betterFetch<AuthSession>(
      `${process.env.NEXT_PUBLIC_AUTH_URL || "http://localhost:4000"}/api/auth/session`,
      {
        baseURL: request.nextUrl.origin,
        headers: {
          cookie: request.headers.get("cookie") || "",
        },
      }
    );

    if (!session || session.user.role !== "admin") {
      // If not an admin, kick them back to the main domain to prevent an infinite redirect loop
      const mainDomainUrl = request.nextUrl.origin.replace("admin.", "");
      return NextResponse.redirect(`${mainDomainUrl}/`);
    }

    // Rewrite to the explicit /admin folder
    return NextResponse.rewrite(new URL(`/admin${url.pathname}`, request.url));
  }

  // --- CUSTOMER DOMAIN LOGIC ---
  return NextResponse.rewrite(new URL(`/user${url.pathname}`, request.url));
}

export const config = {
  // Match all paths EXCEPT those starting with /api, /_next/static, /_next/image, favicon.ico, etc.
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};
