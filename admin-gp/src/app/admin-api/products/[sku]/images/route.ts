import { NextResponse } from "next/server";
import { ragbotApi } from "@/lib/ragbot-api";

type Ctx = { params: Promise<{ sku: string }> };

export async function POST(request: Request, ctx: Ctx) {
  const { sku } = await ctx.params;
  const formData = await request.formData();

  // Re-emit as multipart for FastAPI. fetch() will set the boundary header.
  const upstream = new FormData();
  for (const [key, value] of formData.entries()) {
    upstream.append(key, value as Blob | string);
  }

  const res = await ragbotApi(`/admin/products/${encodeURIComponent(sku)}/images`, {
    method: "POST",
    body: upstream,
    cookie: request.headers.get("cookie") || "",
  });
  const text = await res.text();
  return new NextResponse(text, {
    status: res.status,
    headers: { "content-type": res.headers.get("content-type") || "application/json" },
  });
}
