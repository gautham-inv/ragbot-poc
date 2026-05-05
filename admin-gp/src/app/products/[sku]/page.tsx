"use client";

import React, { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { getCached, setCached, KEYS, invalidate } from "@/lib/products-cache";

type ImageAsset = {
  public_id: string;
  position: number;
  image_url: string;
  thumbnail: string;
};

type Product = {
  sku: string;
  brand?: string;
  name_es?: string;
  name_en?: string;
  name_fr?: string;
  name_pt?: string;
  name_it?: string;
  category?: string;
  subcategory?: string;
  species?: string[];
  price_pvpr?: number;
  price_per_unit?: number;
  min_purchase_qty?: number;
  weight_g?: number;
  color?: string;
  scent?: string;
  ean?: string;
  primary_image?: string;
  thumbnail?: string;
  images?: string[];
  thumbnails?: string[];
  image_assets?: ImageAsset[];
  raw_attributes?: Record<string, unknown>;
  [key: string]: unknown;
};

type EditState = {
  brand: string;
  name_es: string; name_en: string; name_fr: string; name_pt: string; name_it: string;
  category: string; subcategory: string; species: string;
  price_pvpr: string; price_per_unit: string; min_purchase_qty: string;
  weight_g: string; color: string; scent: string; ean: string;
};

function fromProduct(p: Product): EditState {
  return {
    brand: p.brand || "",
    name_es: p.name_es || "", name_en: p.name_en || "", name_fr: p.name_fr || "",
    name_pt: p.name_pt || "", name_it: p.name_it || "",
    category: p.category || "", subcategory: p.subcategory || "",
    species: (p.species || []).join(", "),
    price_pvpr: p.price_pvpr != null ? String(p.price_pvpr) : "",
    price_per_unit: p.price_per_unit != null ? String(p.price_per_unit) : "",
    min_purchase_qty: p.min_purchase_qty != null ? String(p.min_purchase_qty) : "",
    weight_g: p.weight_g != null ? String(p.weight_g) : "",
    color: p.color || "", scent: p.scent || "", ean: p.ean || "",
  };
}

function toPatchBody(s: EditState) {
  const speciesList = s.species.split(",").map((x) => x.trim()).filter(Boolean);
  return {
    brand: s.brand || undefined,
    name_es: s.name_es || undefined, name_en: s.name_en || undefined,
    name_fr: s.name_fr || undefined, name_pt: s.name_pt || undefined,
    name_it: s.name_it || undefined,
    category: s.category || undefined, subcategory: s.subcategory || undefined,
    species: speciesList.length ? speciesList : undefined,
    price_pvpr: s.price_pvpr ? Number(s.price_pvpr) : undefined,
    price_per_unit: s.price_per_unit ? Number(s.price_per_unit) : undefined,
    min_purchase_qty: s.min_purchase_qty ? Number(s.min_purchase_qty) : undefined,
    weight_g: s.weight_g ? Number(s.weight_g) : undefined,
    color: s.color || undefined, scent: s.scent || undefined, ean: s.ean || undefined,
  };
}

export default function ProductDetailPage() {
  const router = useRouter();
  const params = useParams<{ sku: string }>();
  const sku = decodeURIComponent(params.sku);

  const [product, setProduct] = useState<Product | null>(null);
  const [edit, setEdit] = useState<EditState | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busyMsg, setBusyMsg] = useState<string>("");

  async function refresh(useCache = false) {
    setLoading(true);
    setError(null);
    if (useCache) {
      const cached = getCached<Product>(KEYS.detail(sku));
      if (cached) {
        setProduct(cached);
        setEdit(fromProduct(cached));
        setLoading(false);
        return;
      }
    }
    try {
      const res = await fetch(`/admin-api/products/${encodeURIComponent(sku)}`, {
        credentials: "include",
      });
      if (res.status === 404) { setProduct(null); return; }
      if (!res.ok) throw new Error(`fetch failed (${res.status}): ${await res.text()}`);
      const data: Product = await res.json();
      setProduct(data);
      setEdit(fromProduct(data));
      setCached(KEYS.detail(sku), data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sku]);

  async function onSave(e: React.FormEvent) {
    e.preventDefault();
    if (!edit) return;
    setError(null); setSaving(true); setBusyMsg("Saving…");
    try {
      const res = await fetch(`/admin-api/products/${encodeURIComponent(sku)}`, {
        method: "PATCH", headers: { "content-type": "application/json" },
        credentials: "include", body: JSON.stringify(toPatchBody(edit)),
      });
      if (!res.ok) throw new Error(`save failed (${res.status}): ${await res.text()}`);
      const updated: Product = await res.json();
      setProduct(updated); setEdit(fromProduct(updated));
      invalidate(); // list and detail caches both stale now
      setCached(KEYS.detail(sku), updated);
      setBusyMsg("Saved.");
      setTimeout(() => setBusyMsg(""), 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function onAddImage(file: File) {
    setError(null); setBusyMsg(`Uploading ${file.name}…`);
    try {
      const fd = new FormData(); fd.append("file", file);
      const res = await fetch(`/admin-api/products/${encodeURIComponent(sku)}/images`, {
        method: "POST", body: fd, credentials: "include",
      });
      if (!res.ok) throw new Error(`upload failed (${res.status}): ${await res.text()}`);
      invalidate();
      await refresh();
      setBusyMsg("Image added.");
      setTimeout(() => setBusyMsg(""), 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function onRemoveImage(publicId: string) {
    if (!confirm(`Remove image ${publicId}?`)) return;
    setError(null); setBusyMsg("Removing…");
    try {
      const res = await fetch(
        `/admin-api/products/${encodeURIComponent(sku)}/images/${encodeURIComponent(publicId)}`,
        { method: "DELETE", credentials: "include" }
      );
      if (!res.ok && res.status !== 204) {
        throw new Error(`remove failed (${res.status}): ${await res.text()}`);
      }
      invalidate();
      await refresh();
      setBusyMsg("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function onDeleteProduct() {
    if (!confirm(`Delete product ${sku}? This cannot be undone.`)) return;
    setError(null); setBusyMsg("Deleting…");
    try {
      const res = await fetch(`/admin-api/products/${encodeURIComponent(sku)}`, {
        method: "DELETE", credentials: "include",
      });
      if (!res.ok && res.status !== 204) {
        throw new Error(`delete failed (${res.status}): ${await res.text()}`);
      }
      invalidate();
      router.push("/products");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusyMsg("");
    }
  }

  if (loading && !product) {
    return (
      <main className="mx-auto w-full max-w-[1600px] px-7 pt-6 pb-12">
        <div className="mx-auto max-w-3xl">
          <p className="text-[12.5px] text-ink-3">Loading…</p>
        </div>
      </main>
    );
  }

  if (!product || !edit) {
    return (
      <main className="mx-auto w-full max-w-[1600px] px-7 pt-6 pb-12">
        <div className="mx-auto max-w-3xl">
          <h1 className="mb-1 text-[15px] font-semibold text-ink-text">Product not found</h1>
          <p className="mb-4 text-[12.5px] text-ink-3">
            SKU <span className="font-mono">{sku}</span> isn't in the index.
          </p>
          <Link href="/products" className="text-[12.5px] text-ink-700 hover:underline">
            ← Back to products
          </Link>
        </div>
      </main>
    );
  }

  const adminImages = product.image_assets || [];
  const bulkImages = adminImages.length === 0
    ? (product.thumbnails || product.images || []).map((u, i) => ({ key: `bulk-${i}`, url: u }))
    : [];

  return (
    <main className="mx-auto w-full max-w-[1600px] px-7 pt-6 pb-12">
     <div className="mx-auto max-w-3xl">
      <div className="mb-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link
            href="/products"
            className="inline-flex h-8 items-center rounded-md border border-rule bg-card px-2.5 text-[12.5px] text-ink-2 hover:border-rule-strong hover:text-ink-text"
            title="Back to product list"
          >
            ← Products
          </Link>
          <div>
            <h1 className="text-[15px] font-semibold text-ink-text">
              {product.name_es || product.name_en || product.sku}
            </h1>
            <p className="font-mono text-[11px] text-ink-3">{product.sku}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onDeleteProduct}
            className="inline-flex h-8 items-center rounded-md border border-bad/30 bg-bad-bg px-2.5 text-[12.5px] text-bad hover:bg-red-100"
          >
            Delete
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-3 rounded-md border border-bad/30 bg-bad-bg p-3 text-[12.5px] text-bad">
          {error}
        </div>
      )}
      {busyMsg && (
        <div className="mb-3 text-[12.5px] text-ink-3">{busyMsg}</div>
      )}

      {/* Images card */}
      <section className="mb-6 overflow-hidden rounded-[10px] border border-rule bg-card shadow-soft-sm">
        <div className="flex items-center justify-between border-b border-rule px-4 py-3">
          <h2 className="text-[12px] font-semibold uppercase tracking-wider text-ink-3">
            Images
          </h2>
          <label>
            <span className="inline-flex h-8 cursor-pointer items-center rounded-md border border-rule bg-card px-2.5 text-[12.5px] text-ink-2 hover:border-rule-strong hover:text-ink-text">
              + Add image
            </span>
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onAddImage(f);
                e.target.value = "";
              }}
            />
          </label>
        </div>
        <div className="p-4">
          {adminImages.length === 0 && bulkImages.length > 0 && (
            <p className="mb-3 text-[11px] text-ink-3">
              {bulkImages.length} bulk-uploaded image{bulkImages.length === 1 ? "" : "s"} —
              not individually deletable here (managed via Cloudinary).
            </p>
          )}
          {adminImages.length === 0 && bulkImages.length === 0 && (
            <p className="text-[12.5px] text-ink-4">No images yet.</p>
          )}
          <div className="flex flex-wrap gap-3">
            {adminImages.map((a) => (
              <div key={a.public_id} className="relative">
                <img
                  src={a.thumbnail || a.image_url}
                  alt={a.public_id}
                  className="h-28 w-28 rounded-md border border-rule object-cover"
                />
                <button
                  onClick={() => onRemoveImage(a.public_id)}
                  className="absolute -right-2 -top-2 rounded-full border border-bad/40 bg-card px-1.5 py-0.5 text-[11px] font-medium text-bad shadow-soft-sm hover:bg-bad-bg"
                  title="Remove this image"
                >
                  ×
                </button>
              </div>
            ))}
            {bulkImages.map((img) => (
              <img
                key={img.key}
                src={img.url}
                alt={img.key}
                className="h-28 w-28 rounded-md border border-rule object-cover opacity-90"
              />
            ))}
          </div>
        </div>
      </section>

      {/* Edit form */}
      <form
        onSubmit={onSave}
        className="overflow-hidden rounded-[10px] border border-rule bg-card shadow-soft-sm"
      >
        <div className="border-b border-rule px-4 py-3">
          <h2 className="text-[12px] font-semibold uppercase tracking-wider text-ink-3">
            Details
          </h2>
        </div>

        <div className="space-y-4 p-4">
          <Field label="Brand" value={edit.brand} onChange={(v) => setEdit({ ...edit, brand: v })} />

          <div className="grid grid-cols-2 gap-3">
            <Field label="Name (ES)" value={edit.name_es} onChange={(v) => setEdit({ ...edit, name_es: v })} />
            <Field label="Name (EN)" value={edit.name_en} onChange={(v) => setEdit({ ...edit, name_en: v })} />
            <Field label="Name (FR)" value={edit.name_fr} onChange={(v) => setEdit({ ...edit, name_fr: v })} />
            <Field label="Name (PT)" value={edit.name_pt} onChange={(v) => setEdit({ ...edit, name_pt: v })} />
            <Field label="Name (IT)" value={edit.name_it} onChange={(v) => setEdit({ ...edit, name_it: v })} />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Field label="Category" value={edit.category} onChange={(v) => setEdit({ ...edit, category: v })} />
            <Field label="Subcategory" value={edit.subcategory} onChange={(v) => setEdit({ ...edit, subcategory: v })} />
          </div>

          <Field
            label="Species (comma-separated)"
            value={edit.species}
            onChange={(v) => setEdit({ ...edit, species: v })}
          />

          <div className="grid grid-cols-3 gap-3">
            <Field label="Price (€)" value={edit.price_pvpr} onChange={(v) => setEdit({ ...edit, price_pvpr: v })} type="number" />
            <Field label="Price/unit (€)" value={edit.price_per_unit} onChange={(v) => setEdit({ ...edit, price_per_unit: v })} type="number" />
            <Field label="Min order qty" value={edit.min_purchase_qty} onChange={(v) => setEdit({ ...edit, min_purchase_qty: v })} type="number" />
          </div>

          <div className="grid grid-cols-3 gap-3">
            <Field label="Weight (g)" value={edit.weight_g} onChange={(v) => setEdit({ ...edit, weight_g: v })} type="number" />
            <Field label="Color" value={edit.color} onChange={(v) => setEdit({ ...edit, color: v })} />
            <Field label="Scent" value={edit.scent} onChange={(v) => setEdit({ ...edit, scent: v })} />
          </div>

          <Field label="EAN/Barcode" value={edit.ean} onChange={(v) => setEdit({ ...edit, ean: v })} />
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-rule bg-canvas px-4 py-3">
          <button
            type="submit"
            disabled={saving}
            className="inline-flex h-8 items-center rounded-md bg-ink-700 px-3 text-[12.5px] font-medium text-white hover:bg-ink-600 disabled:opacity-40"
          >
            {saving ? "Saving…" : "Save changes"}
          </button>
        </div>
      </form>

      {/* Raw payload */}
      <details className="mt-6">
        <summary className="cursor-pointer text-[11px] text-ink-4 hover:text-ink-2">
          Raw Qdrant payload
        </summary>
        <pre className="mt-2 overflow-auto rounded-md border border-rule bg-canvas p-3 text-[11px] text-ink-2">
          {JSON.stringify(product, null, 2)}
        </pre>
      </details>
     </div>
    </main>
  );
}

function Field(props: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
  required?: boolean;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-[12px] font-medium text-ink-2">{props.label}</span>
      <input
        type={props.type || "text"}
        value={props.value}
        onChange={(e) => props.onChange(e.target.value)}
        required={props.required}
        className="h-9 w-full rounded-md border border-rule bg-card px-3 text-[12.5px] text-ink-text placeholder:text-ink-3 focus:border-rule-strong focus:outline-none"
      />
    </label>
  );
}
