import { NextResponse } from "next/server";
import { ragbotApi } from "@/lib/ragbot-api";

type Ctx = { params: Promise<{ sku: string; publicId: string }> };

export async function DELETE(request: Request, ctx: Ctx) {
  const { sku, publicId } = await ctx.params;
  const res = await ragbotApi(
    `/admin/products/${encodeURIComponent(sku)}/images/${encodeURIComponent(publicId)}`,
    {
      method: "DELETE",
      cookie: request.headers.get("cookie") || "",
    }
  );
  if (res.status === 204) return new NextResponse(null, { status: 204 });
  const text = await res.text();
  return new NextResponse(text, {
    status: res.status,
    headers: { "content-type": res.headers.get("content-type") || "application/json" },
  });
}
