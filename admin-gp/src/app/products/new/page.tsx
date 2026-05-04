"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { invalidate } from "@/lib/products-cache";

type FormState = {
  sku: string; brand: string;
  name_es: string; name_en: string;
  category: string; subcategory: string; species: string;
  price_pvpr: string; min_purchase_qty: string; weight_g: string;
  color: string; scent: string; ean: string;
};

const EMPTY: FormState = {
  sku: "", brand: "",
  name_es: "", name_en: "",
  category: "", subcategory: "", species: "",
  price_pvpr: "", min_purchase_qty: "", weight_g: "",
  color: "", scent: "", ean: "",
};

function toApiBody(s: FormState) {
  const speciesList = s.species.split(",").map((x) => x.trim()).filter(Boolean);
  return {
    sku: s.sku.trim(), brand: s.brand.trim(),
    name_es: s.name_es.trim() || undefined,
    name_en: s.name_en.trim() || undefined,
    category: s.category.trim() || undefined,
    subcategory: s.subcategory.trim() || undefined,
    species: speciesList.length ? speciesList : undefined,
    price_pvpr: s.price_pvpr ? Number(s.price_pvpr) : undefined,
    min_purchase_qty: s.min_purchase_qty ? Number(s.min_purchase_qty) : undefined,
    weight_g: s.weight_g ? Number(s.weight_g) : undefined,
    color: s.color.trim() || undefined,
    scent: s.scent.trim() || undefined,
    ean: s.ean.trim() || undefined,
  };
}

export default function NewProductPage() {
  const router = useRouter();
  const [form, setForm] = useState<FormState>(EMPTY);
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string>("");

  function set<K extends keyof FormState>(k: K, v: string) {
    setForm((s) => ({ ...s, [k]: v }));
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null); setBusy(true); setProgress("Creating product…");
    try {
      const createRes = await fetch("/admin-api/products", {
        method: "POST",
        headers: { "content-type": "application/json" },
        credentials: "include",
        body: JSON.stringify(toApiBody(form)),
      });
      if (!createRes.ok) {
        throw new Error(`create failed (${createRes.status}): ${await createRes.text()}`);
      }

      for (let i = 0; i < files.length; i++) {
        setProgress(`Uploading image ${i + 1}/${files.length}…`);
        const fd = new FormData();
        fd.append("file", files[i]);
        const up = await fetch(
          `/admin-api/products/${encodeURIComponent(form.sku)}/images`,
          { method: "POST", body: fd, credentials: "include" }
        );
        if (!up.ok) {
          throw new Error(`image ${i + 1} failed (${up.status}): ${await up.text()}`);
        }
      }

      invalidate(); // list cache stale
      setProgress("Done.");
      router.push(`/products/${encodeURIComponent(form.sku)}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto max-w-3xl px-7 pt-6 pb-12">
      <div className="mb-5 flex items-center justify-between">
        <div>
          <h1 className="text-[15px] font-semibold text-ink-text">Add product</h1>
          <p className="text-[12.5px] text-ink-3">
            Indexed into Qdrant immediately. Images upload to Cloudinary after the
            product record is created.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link
            href="/products"
            className="inline-flex h-8 items-center rounded-md border border-rule bg-card px-2.5 text-[12.5px] text-ink-2 hover:border-rule-strong hover:text-ink-text"
          >
            ← Products
          </Link>
        </div>
      </div>

      {error && (
        <div className="mb-3 rounded-md border border-bad/30 bg-bad-bg p-3 text-[12.5px] text-bad">
          {error}
        </div>
      )}

      <form
        onSubmit={onSubmit}
        className="overflow-hidden rounded-[10px] border border-rule bg-card shadow-soft-sm"
      >
        <div className="border-b border-rule px-4 py-3">
          <h2 className="text-[12px] font-semibold uppercase tracking-wider text-ink-3">
            Identity
          </h2>
        </div>
        <div className="space-y-4 p-4">
          <div className="grid grid-cols-2 gap-3">
            <Field label="SKU *" value={form.sku} onChange={(v) => set("sku", v)} required />
            <Field label="Brand *" value={form.brand} onChange={(v) => set("brand", v)} required />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Name (ES)" value={form.name_es} onChange={(v) => set("name_es", v)} />
            <Field label="Name (EN)" value={form.name_en} onChange={(v) => set("name_en", v)} />
          </div>
        </div>

        <div className="border-y border-rule bg-canvas px-4 py-3">
          <h2 className="text-[12px] font-semibold uppercase tracking-wider text-ink-3">
            Classification
          </h2>
        </div>
        <div className="space-y-4 p-4">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Category" value={form.category} onChange={(v) => set("category", v)} />
            <Field label="Subcategory" value={form.subcategory} onChange={(v) => set("subcategory", v)} />
          </div>
          <Field
            label="Species (comma-separated, e.g. dog, cat)"
            value={form.species}
            onChange={(v) => set("species", v)}
          />
        </div>

        <div className="border-y border-rule bg-canvas px-4 py-3">
          <h2 className="text-[12px] font-semibold uppercase tracking-wider text-ink-3">
            Commerce
          </h2>
        </div>
        <div className="space-y-4 p-4">
          <div className="grid grid-cols-3 gap-3">
            <Field label="Price (€)" value={form.price_pvpr} onChange={(v) => set("price_pvpr", v)} type="number" />
            <Field label="Min order qty" value={form.min_purchase_qty} onChange={(v) => set("min_purchase_qty", v)} type="number" />
            <Field label="Weight (g)" value={form.weight_g} onChange={(v) => set("weight_g", v)} type="number" />
          </div>
          <div className="grid grid-cols-3 gap-3">
            <Field label="Color" value={form.color} onChange={(v) => set("color", v)} />
            <Field label="Scent" value={form.scent} onChange={(v) => set("scent", v)} />
            <Field label="EAN/Barcode" value={form.ean} onChange={(v) => set("ean", v)} />
          </div>
        </div>

        <div className="border-y border-rule bg-canvas px-4 py-3">
          <h2 className="text-[12px] font-semibold uppercase tracking-wider text-ink-3">
            Images
          </h2>
        </div>
        <div className="space-y-3 p-4">
          <label>
            <span className="inline-flex h-9 cursor-pointer items-center rounded-md border border-rule bg-card px-3 text-[12.5px] text-ink-2 hover:border-rule-strong hover:text-ink-text">
              Choose files…
            </span>
            <input
              type="file"
              multiple
              accept="image/*"
              onChange={(e) => setFiles(Array.from(e.target.files || []))}
              className="hidden"
            />
          </label>
          {files.length > 0 && (
            <ul className="space-y-1 text-[12px] text-ink-2">
              {files.map((f, i) => (
                <li key={i} className="font-mono">
                  #{i}: {f.name} <span className="text-ink-4">({Math.round(f.size / 1024)} KB)</span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="flex items-center justify-end gap-3 border-t border-rule bg-canvas px-4 py-3">
          {busy && progress && (
            <span className="text-[12.5px] text-ink-3">{progress}</span>
          )}
          <button
            type="submit"
            disabled={busy || !form.sku || !form.brand}
            className="inline-flex h-8 items-center rounded-md bg-ink-700 px-3 text-[12.5px] font-medium text-white hover:bg-ink-600 disabled:opacity-40"
          >
            {busy ? "Working…" : "Create product"}
          </button>
        </div>
      </form>
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
      <span className="mb-1 block text-[12px] font-medium text-ink-2">
        {props.label}
      </span>
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
