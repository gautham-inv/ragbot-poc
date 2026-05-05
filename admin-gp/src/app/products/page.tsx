"use client";

import React, { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { getCached, setCached, KEYS, invalidate } from "@/lib/products-cache";

type Product = {
  sku: string;
  brand?: string;
  name_es?: string;
  name_en?: string;
  category?: string;
  subcategory?: string;
  species?: string[];
  price_pvpr?: number;
  thumbnail?: string;
  primary_image?: string;
};

type ListResponse = { products: Product[]; next_offset: string | null };
type ListCache = { products: Product[]; offset: string | null; done: boolean };

const PAGE_SIZE = 50;

export default function ProductsPage() {
  const [products, setProducts] = useState<Product[]>([]);
  const [offset, setOffset] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [query, setQuery] = useState("");
  const [hydrated, setHydrated] = useState(false);

  async function loadPage(currentOffset: string | null, append: boolean) {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ limit: String(PAGE_SIZE) });
      if (currentOffset) params.set("offset", currentOffset);
      const res = await fetch(`/admin-api/products?${params.toString()}`, {
        credentials: "include",
      });
      if (!res.ok) throw new Error(`list failed (${res.status}): ${await res.text()}`);
      const data: ListResponse = await res.json();
      const next = append ? [...products, ...(data.products || [])] : (data.products || []);
      setProducts(next);
      setOffset(data.next_offset);
      const isDone = !data.next_offset;
      setDone(isDone);
      setCached<ListCache>(KEYS.list, {
        products: next,
        offset: data.next_offset,
        done: isDone,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    const cached = getCached<ListCache>(KEYS.list);
    if (cached) {
      setProducts(cached.products);
      setOffset(cached.offset);
      setDone(cached.done);
      setHydrated(true);
      return;
    }
    setHydrated(true);
    loadPage(null, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function refresh() {
    invalidate();
    setProducts([]);
    setOffset(null);
    setDone(false);
    loadPage(null, false);
  }

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return products;
    return products.filter((p) =>
      [p.sku, p.brand, p.name_es, p.name_en, p.category, p.subcategory]
        .some((f) => (f || "").toLowerCase().includes(q))
    );
  }, [products, query]);

  return (
    <main className="mx-auto w-full max-w-[1600px] px-7 pt-6 pb-12">
      <div className="mb-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="inline-flex h-8 items-center rounded-md border border-rule bg-card px-2.5 text-[12.5px] text-ink-2 hover:border-rule-strong hover:text-ink-text"
            title="Back to analytics dashboard"
          >
            ← Dashboard
          </Link>
          <div>
            <h1 className="text-[15px] font-semibold text-ink-text">Products</h1>
            <p className="text-[12.5px] text-ink-3">
              {products.length} loaded{done ? "" : "+"}
              {filtered.length !== products.length && ` · ${filtered.length} match`}
              {loading && " · loading…"}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={refresh}
            disabled={loading}
            title="Re-fetch from server"
            className="h-8 rounded-md border border-rule bg-card px-2.5 text-[12.5px] text-ink-2 hover:border-rule-strong hover:text-ink-text disabled:opacity-50"
          >
            Refresh
          </button>
          <Link
            href="/products/new"
            className="inline-flex h-8 items-center rounded-md bg-ink-700 px-3 text-[12.5px] font-medium text-white hover:bg-ink-600"
          >
            + Add product
          </Link>
        </div>
      </div>

      <div className="mb-3">
        <input
          type="search"
          placeholder="Search SKU, brand, name, category…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="h-9 w-full rounded-md border border-rule bg-card px-3 text-[12.5px] text-ink-text placeholder:text-ink-3 focus:border-rule-strong focus:outline-none"
        />
      </div>

      {error && (
        <div className="mb-3 rounded-md border border-bad/30 bg-bad-bg p-3 text-[12.5px] text-bad">
          {error}
        </div>
      )}

      <div className="overflow-hidden rounded-[10px] border border-rule bg-card shadow-soft-sm">
        <div className="overflow-x-auto">
          <table className="min-w-full text-[12.5px]">
            <thead>
              <tr className="bg-ink-50 text-left text-[10.5px] uppercase tracking-wider text-ink-3">
                <th className="w-14 px-3 py-2.5 font-medium"></th>
                <th className="px-3 py-2.5 font-medium">SKU</th>
                <th className="px-3 py-2.5 font-medium">Brand</th>
                <th className="px-3 py-2.5 font-medium">Name</th>
                <th className="px-3 py-2.5 font-medium">Category</th>
                <th className="px-3 py-2.5 font-medium">Species</th>
                <th className="px-3 py-2.5 text-right font-medium">Price</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((p) => (
                <tr
                  key={p.sku}
                  className="border-t border-rule transition-colors hover:bg-ink-50"
                >
                  <td className="px-3 py-2">
                    {p.thumbnail || p.primary_image ? (
                      <img
                        src={p.thumbnail || p.primary_image}
                        alt={p.sku}
                        className="h-9 w-9 rounded-md border border-rule object-cover"
                      />
                    ) : (
                      <div className="h-9 w-9 rounded-md border border-rule bg-ink-50 text-center text-[10px] leading-9 text-ink-4">
                        —
                      </div>
                    )}
                  </td>
                  <td className="px-3 py-2 font-mono text-[11px] text-ink-2">
                    <Link
                      href={`/products/${encodeURIComponent(p.sku)}`}
                      className="text-ink-700 hover:underline"
                    >
                      {p.sku}
                    </Link>
                  </td>
                  <td className="px-3 py-2 text-ink-text">{p.brand || "—"}</td>
                  <td className="px-3 py-2 text-ink-text">
                    {p.name_es || p.name_en || "—"}
                  </td>
                  <td className="px-3 py-2 text-ink-2">
                    {[p.category, p.subcategory].filter(Boolean).join(" / ") || "—"}
                  </td>
                  <td className="px-3 py-2 text-ink-2">
                    {(p.species || []).join(", ") || "—"}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-ink-text">
                    {typeof p.price_pvpr === "number"
                      ? `${p.price_pvpr.toFixed(2)} €`
                      : "—"}
                  </td>
                </tr>
              ))}
              {hydrated && filtered.length === 0 && !loading && (
                <tr>
                  <td colSpan={7} className="px-3 py-10 text-center text-[12.5px] text-ink-4">
                    No products{query ? " matching" : ""}.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-center gap-2">
        {!done && (
          <button
            onClick={() => loadPage(offset, true)}
            disabled={loading}
            className="h-9 rounded-md border border-rule bg-card px-4 text-[12.5px] text-ink-2 hover:border-rule-strong hover:text-ink-text disabled:opacity-50"
          >
            {loading ? "Loading…" : "Load more"}
          </button>
        )}
        {done && (
          <p className="text-[11px] text-ink-4">
            All {products.length} products loaded.
          </p>
        )}
      </div>
    </main>
  );
}
