import { NextResponse } from "next/server";
import { ragbotApi } from "@/lib/ragbot-api";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const qs = url.search;
  const res = await ragbotApi(`/admin/products${qs}`, {
    method: "GET",
    cookie: request.headers.get("cookie") || "",
  });
  const body = await res.text();
  return new NextResponse(body, {
    status: res.status,
    headers: { "content-type": res.headers.get("content-type") || "application/json" },
  });
}

export async function POST(request: Request) {
  const body = await request.text();
  const res = await ragbotApi("/admin/products", {
    method: "POST",
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
