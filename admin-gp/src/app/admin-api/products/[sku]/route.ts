import { NextResponse } from "next/server";
import { ragbotApi } from "@/lib/ragbot-api";

type Ctx = { params: Promise<{ sku: string }> };

export async function GET(request: Request, ctx: Ctx) {
  const { sku } = await ctx.params;
  const res = await ragbotApi(`/admin/products/${encodeURIComponent(sku)}`, {
    method: "GET",
    cookie: request.headers.get("cookie") || "",
  });
  const body = await res.text();
  return new NextResponse(body, {
    status: res.status,
    headers: { "content-type": res.headers.get("content-type") || "application/json" },
  });
}

export async function PATCH(request: Request, ctx: Ctx) {
  const { sku } = await ctx.params;
  const body = await request.text();
  const res = await ragbotApi(`/admin/products/${encodeURIComponent(sku)}`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body,
    cookie: request.headers.get("cookie") || "",
  });
  const text = await res.text();
  return new NextResponse(text, {
    status: res.status,
    headers: { "content-type": res.headers.get("content-type") || "application/json" },
  });
}

export async function DELETE(request: Request, ctx: Ctx) {
  const { sku } = await ctx.params;
  const res = await ragbotApi(`/admin/products/${encodeURIComponent(sku)}`, {
    method: "DELETE",
    cookie: request.headers.get("cookie") || "",
  });
  if (res.status === 204) return new NextResponse(null, { status: 204 });
  const text = await res.text();
  return new NextResponse(text, {
    status: res.status,
    headers: { "content-type": res.headers.get("content-type") || "application/json" },
  });
}
