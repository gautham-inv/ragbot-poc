"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { signOut } from "@/lib/auth-client";

/* ---------------- Types ---------------- */
type Range = "24h" | "7d" | "30d";

type Kpis = {
  total_queries: number;
  purchase_intent_rate: number;
  basket_queries: number;
  avg_basket_budget_eur: number;
  comparison_queries: number;
  sku_hit_rate: number;
  top_brand: string | null;
  top_category: string | null;
  p50_latency_sec: number;
  p95_latency_sec: number;
  zero_result_rate: number;
  fallback_rate: number;
  avg_confidence: number;
  low_confidence_rate: number;
};

type VolumeRow = {
  hour: string;
  product_search: number;
  product_recommendation: number;
  price_compare: number;
  basket_build: number;
  barcode_lookup: number;
  order_status: number;
  general_qa: number;
  other: number;
};

type Named = { name: string; value: number; color?: string };

type RecentQuery = {
  id: string;
  timestamp: string;
  query: string;
  intent: string;
  intent_confidence: number;
  latency: number;
  latency_known: boolean;
  answer: string;
  path: "stream" | "tools";
  tools_used: string[];
  status: "success" | "critical" | "zero_result";
  rawInput: any;
  rawOutput: any;
};

type TopSKU = { sku: string; product: string; hits: number; frequency: number };

type AnalyticsPayload = {
  kpis: Kpis;
  charts: {
    volume: VolumeRow[];
    latency: { bucket: string; count: number; color: string }[];
    confidence: { hour: string; confidence: number | null }[];
    intent: Named[];
    languages: { name: string; value: number }[];
    topBrands: Named[];
    topCategories: Named[];
    topSubcategories: Named[];
    toolUsage: Named[];
    pathDistribution: Named[];
  };
  topSKUs: TopSKU[];
  recentQueries: RecentQuery[];
};

/* ---------------- Icons ---------------- */
const Icon = {
  search: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className="h-3.5 w-3.5">
      <circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" />
    </svg>
  ),
  bell: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" className="h-4 w-4">
      <path d="M6 8a6 6 0 0 1 12 0c0 7 3 7 3 9H3c0-2 3-2 3-9zM9 21a3 3 0 0 0 6 0" />
    </svg>
  ),
  dl: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" className="h-4 w-4">
      <path d="M12 3v12m0 0 4-4m-4 4-4-4M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
    </svg>
  ),
  logout: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" className="h-4 w-4">
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9" />
    </svg>
  ),
  refresh: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" className="h-4 w-4">
      <path d="M21 12a9 9 0 1 1-3-6.7M21 4v5h-5" />
    </svg>
  ),
  x: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-3.5 w-3.5">
      <path d="M6 6l12 12M18 6 6 18" />
    </svg>
  ),
  filter: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" className="h-3 w-3">
      <path d="M4 5h16l-6 8v6l-4-2v-4z" />
    </svg>
  ),
  up: <svg viewBox="0 0 24 24" fill="currentColor" className="h-2.5 w-2.5"><path d="M7 14l5-5 5 5z" /></svg>,
  down: <svg viewBox="0 0 24 24" fill="currentColor" className="h-2.5 w-2.5"><path d="M7 10l5 5 5-5z" /></svg>,
  copy: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" className="h-3 w-3">
      <rect x="9" y="9" width="12" height="12" rx="2" />
      <path d="M15 9V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h4" />
    </svg>
  ),
  check: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" className="h-3.5 w-3.5">
      <path d="m5 12 5 5 9-11" />
    </svg>
  ),
  chev: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-3.5 w-3.5">
      <path d="m9 6 6 6-6 6" />
    </svg>
  ),
  csv: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" className="h-3.5 w-3.5">
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" /><path d="M9 14h6M9 17h4" />
    </svg>
  ),
  pdf: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" className="h-3.5 w-3.5">
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" /><path d="M9 13v5M9 13h2a1.5 1.5 0 0 1 0 3H9M13 13h2.5M13 13v5M13 16h2" />
    </svg>
  ),
  png: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" className="h-3.5 w-3.5">
      <rect x="3" y="5" width="18" height="14" rx="2" /><circle cx="9" cy="11" r="1.5" />
      <path d="m5 17 4-4 4 3 3-3 4 4" />
    </svg>
  ),
  link: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" className="h-3.5 w-3.5">
      <path d="M10 14a4 4 0 0 0 5.66 0l3-3a4 4 0 1 0-5.66-5.66l-1.5 1.5" />
      <path d="M14 10a4 4 0 0 0-5.66 0l-3 3a4 4 0 1 0 5.66 5.66l1.5-1.5" />
    </svg>
  ),
};

/* ---------------- Helpers ---------------- */
const cn = (...parts: (string | false | null | undefined)[]) => parts.filter(Boolean).join(" ");

function formatNum(n: number | null | undefined) {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toLocaleString("en-US");
}

function timeAgoISO(iso: string) {
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return "—";
  const ms = Date.now() - then;
  const m = Math.max(0, Math.round(ms / 60000));
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

/* intent → { badge class, label } */
const INTENT_LABEL: Record<string, string> = {
  product_search: "Product search",
  product_recommendation: "Recommendation",
  price_compare: "Compare",
  basket_build: "Basket",
  barcode_lookup: "Barcode",
  order_status: "Order",
  general_qa: "Q&A",
  other: "Other",
  unknown: "Unknown",
};

function intentBadge(intent: string) {
  // Map the 8 backend intents onto 5 badge styles.
  switch (intent) {
    case "product_search":
    case "barcode_lookup":
      return { cls: "text-[#0b5fd1] bg-[#e9f1fd] border-[#d3e3fa]", label: INTENT_LABEL[intent] || intent };
    case "price_compare":
      return { cls: "text-good bg-good-bg border-[#cfeadd]", label: INTENT_LABEL[intent] || intent };
    case "basket_build":
      return { cls: "text-warn bg-warn-bg border-[#f3e1c1]", label: INTENT_LABEL[intent] || intent };
    case "order_status":
      return { cls: "text-[#6a2fb4] bg-[#f1eafb] border-[#e1d3f5]", label: INTENT_LABEL[intent] || intent };
    default:
      return { cls: "text-[#576074] bg-[#eef0f4] border-[#e1e4eb]", label: INTENT_LABEL[intent] || intent };
  }
}

function buildSpark(volume: VolumeRow[], key: keyof VolumeRow | "total" | "purchase_rate"): number[] {
  if (!Array.isArray(volume) || volume.length === 0) return [];
  const PURCHASE_KEYS: (keyof VolumeRow)[] = [
    "product_search", "product_recommendation", "price_compare", "basket_build", "barcode_lookup",
  ];
  return volume.map((row) => {
    if (key === "total") {
      return PURCHASE_KEYS.reduce((s, k) => s + (row[k] as number), 0) +
        (row.order_status + row.general_qa + row.other);
    }
    if (key === "purchase_rate") {
      const total = PURCHASE_KEYS.reduce((s, k) => s + (row[k] as number), 0) +
        (row.order_status + row.general_qa + row.other);
      const purchase = PURCHASE_KEYS.reduce((s, k) => s + (row[k] as number), 0);
      return total > 0 ? (purchase / total) * 100 : 0;
    }
    return row[key] as number;
  });
}

/* ---------------- Spark ---------------- */
function Spark({ data, color, fill }: { data: number[]; color: string; fill: string }) {
  if (!data || data.length < 2) return <div className="h-7" />;
  const w = 160, h = 28;
  const max = Math.max(...data);
  const min = Math.min(...data);
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / (max - min || 1)) * (h - 4) - 2;
    return [x, y];
  });
  const d = "M" + pts.map((p) => p.join(",")).join(" L");
  const area = d + ` L${w},${h} L0,${h} Z`;
  return (
    <svg className="h-7 mt-3" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" width="100%" height="28">
      <path d={area} fill={fill} opacity="0.5" />
      <path d={d} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

/* ---------------- KPI ---------------- */
function KPI(props: {
  label: string;
  value: string;
  unit?: string;
  sub?: string;
  primary?: boolean;
  spark?: number[];
  sparkAlt?: boolean;
}) {
  const { label, value, unit, sub, primary, spark, sparkAlt } = props;
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-[10px] border border-rule shadow-soft-sm px-5 pt-[18px] pb-5",
        primary ? "bg-card" : "bg-[#fbfcfd]"
      )}
    >
      {primary && <div className="absolute left-0 top-3.5 bottom-3.5 w-[3px] rounded-r bg-ink-700" />}
      <div className={cn("flex items-center gap-2 text-xs font-medium mb-2.5", primary ? "text-ink-2" : "text-ink-3")}>
        <span>{label}</span>
        {!primary && (
          <span className="text-[10px] font-semibold uppercase text-ink-4 tracking-[0.06em]">· secondary</span>
        )}
      </div>
      <div className="flex items-baseline gap-2.5 tabular-nums">
        <span
          className={cn(
            "font-semibold tracking-[-0.02em] text-ink-text",
            primary ? "text-[34px]" : "text-[26px]"
          )}
        >
          {value}
        </span>
        {unit && <span className="text-sm text-ink-3 font-medium">{unit}</span>}
      </div>
      {sub && <div className="mt-2.5 text-xs text-ink-3">{sub}</div>}
      {spark && spark.length >= 2 && (
        <Spark
          data={spark}
          color={sparkAlt ? "#37BAD1" : "#003777"}
          fill={sparkAlt ? "#d7f0f5" : "#e3ecf7"}
        />
      )}
    </div>
  );
}

/* ---------------- Horizontal bars ---------------- */
function HBars({ data, alt, suffix }: { data: { name: string; value: number; color?: string }[]; alt?: boolean; suffix?: string }) {
  if (!data.length) return <div className="py-8 text-center text-ink-3 text-sm">No data</div>;
  const max = Math.max(...data.map((d) => d.value)) || 1;
  return (
    <div>
      {data.map((d) => (
        <div key={d.name} className="grid grid-cols-[140px_1fr_60px] gap-3 items-center py-1.5 text-[12.5px]">
          <div className="text-ink-text font-medium truncate" title={d.name}>{d.name}</div>
          <div className="relative h-2.5 rounded-full bg-[#eef1f5] overflow-hidden">
            <div
              className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-300 ease-ease"
              style={{
                width: `${(d.value / max) * 100}%`,
                background: d.color || (alt ? "#37BAD1" : "#003777"),
              }}
            />
          </div>
          <div className="text-right text-ink-2 font-medium tabular-nums">
            {d.value.toLocaleString()}{suffix || ""}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ---------------- Donut ---------------- */
function Donut({ data, size = 140 }: { data: Named[]; size?: number }) {
  const total = data.reduce((s, d) => s + d.value, 0);
  if (total === 0) return <div className="py-8 text-center text-ink-3 text-sm">No data</div>;
  const r = size / 2 - 4, cx = size / 2, cy = size / 2;
  let a0 = -Math.PI / 2;
  const segs = data.map((d) => {
    const frac = d.value / total;
    const a1 = a0 + frac * Math.PI * 2;
    const large = frac > 0.5 ? 1 : 0;
    const x0 = cx + r * Math.cos(a0), y0 = cy + r * Math.sin(a0);
    const x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
    const path = `M${cx},${cy} L${x0},${y0} A${r},${r} 0 ${large} 1 ${x1},${y1} Z`;
    a0 = a1;
    return { path, color: d.color || "#003777" };
  });
  return (
    <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size}>
      {segs.map((s, i) => <path key={i} d={s.path} fill={s.color} />)}
      <circle cx={cx} cy={cy} r={r * 0.58} fill="#fff" />
      <text x={cx} y={cy - 4} textAnchor="middle" fontSize="11" fill="#838a97" fontWeight="500">Total</text>
      <text x={cx} y={cy + 12} textAnchor="middle" fontSize="14" fill="#0f1420" fontWeight="600">
        {total.toLocaleString()}
      </text>
    </svg>
  );
}

function Legend({ data }: { data: Named[] }) {
  const total = data.reduce((s, d) => s + d.value, 0) || 1;
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs text-ink-2">
      {data.map((d) => (
        <div key={d.name} className="truncate">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm mr-2 align-middle"
            style={{ background: d.color || "#003777" }}
          />
          <span className="align-middle">{d.name}</span>
          <span className="float-right text-ink-3 tabular-nums">
            {Math.round((d.value / total) * 100)}%
          </span>
        </div>
      ))}
    </div>
  );
}

/* ---------------- Hour bars ---------------- */
function HourBars({ data }: { data: VolumeRow[] }) {
  const PURCHASE_KEYS: (keyof VolumeRow)[] = [
    "product_search", "product_recommendation", "price_compare", "basket_build", "barcode_lookup",
  ];
  const totals = data.map((row) =>
    PURCHASE_KEYS.reduce((s, k) => s + (row[k] as number), 0) + row.order_status + row.general_qa + row.other
  );
  const max = Math.max(...totals, 1);
  const nowH = new Date().getHours();
  return (
    <div>
      <div className="grid grid-cols-24 gap-[3px] h-[180px] items-end pt-2 relative"
        style={{ gridTemplateColumns: "repeat(24, minmax(0, 1fr))" }}>
        <div className="grid-lines"><div /><div /><div /><div /></div>
        {totals.map((v, i) => (
          <div key={i} className="relative group">
            <div
              className="rounded-t min-h-[2px] transition-all duration-150 ease-ease"
              style={{
                height: `${(v / max) * 100}%`,
                background: i === nowH ? "#37BAD1" : "#003777",
              }}
            />
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 bg-ink-text text-white text-[11px] px-1.5 py-1 rounded whitespace-nowrap opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity tabular-nums">
              {String(i).padStart(2, "0")}:00 · {v.toLocaleString()}
            </div>
          </div>
        ))}
      </div>
      <div className="grid text-[10px] text-ink-3 mt-1.5 text-center"
        style={{ gridTemplateColumns: "repeat(24, minmax(0, 1fr))" }}>
        {Array.from({ length: 24 }, (_, i) => (
          <span key={i} className={i % 2 === 0 ? "" : "invisible"}>
            {String(i).padStart(2, "0")}
          </span>
        ))}
      </div>
    </div>
  );
}

/* ---------------- Popover ---------------- */
function Popover({
  open, onClose, children, className,
}: { open: boolean; onClose: () => void; children: React.ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div
      ref={ref}
      className={cn(
        "absolute right-0 top-[calc(100%+8px)] bg-card border border-rule rounded-xl shadow-soft-lg z-30 overflow-hidden w-[340px]",
        className
      )}
    >
      {children}
    </div>
  );
}

/* ---------------- FilterMenu ---------------- */
function FilterMenu<T extends string>({
  label, options, selected, onChange, onClose,
}: {
  label: string;
  options: { value: T; label: string }[];
  selected: T[];
  onChange: (next: T[]) => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [onClose]);
  return (
    <div
      ref={ref}
      className="absolute right-0 top-[calc(100%+6px)] bg-card border border-rule rounded-[10px] shadow-soft p-2.5 w-[220px] z-10"
    >
      <div className="text-[11px] font-semibold text-ink-3 uppercase tracking-[0.06em] px-1.5 pb-2 pt-1">{label}</div>
      <div className="max-h-[260px] overflow-y-auto">
        {options.map((o) => (
          <label
            key={o.value}
            className="flex items-center gap-2 px-1.5 py-1.5 rounded-md text-[12.5px] text-ink-2 cursor-pointer hover:bg-[#f4f5f8] hover:text-ink-text select-none"
          >
            <input
              type="checkbox"
              className="accent-ink-700"
              checked={selected.includes(o.value)}
              onChange={() => {
                const next = selected.includes(o.value)
                  ? selected.filter((v) => v !== o.value)
                  : [...selected, o.value];
                onChange(next);
              }}
            />
            {o.label}
          </label>
        ))}
      </div>
      <div className="flex gap-1.5 mt-2 border-t border-rule pt-2">
        <button
          className="flex-1 rounded px-1.5 py-1 text-xs font-medium text-ink-2 hover:bg-[#f4f5f8] hover:text-ink-text"
          onClick={() => onChange([])}
        >
          Clear
        </button>
        <button
          className="flex-1 rounded px-1.5 py-1 text-xs font-medium text-white bg-ink-700"
          onClick={onClose}
        >
          Apply
        </button>
      </div>
    </div>
  );
}

/* ---------------- Chip ---------------- */
function Chip({ children, onRemove, muted }: { children: React.ReactNode; onRemove?: () => void; muted?: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border pl-2.5 pr-2 py-0.5 text-[11.5px] font-medium",
        muted
          ? "text-ink-3 bg-[#f4f5f8] border-rule cursor-pointer"
          : "text-ink-700 bg-ink-50 border-ink-100"
      )}
      onClick={muted ? onRemove : undefined}
    >
      {children}
      {!muted && onRemove && (
        <button
          onClick={onRemove}
          className="h-3.5 w-3.5 rounded-full grid place-items-center text-ink-700 hover:bg-ink-700/10"
          aria-label="Remove"
        >
          {Icon.x}
        </button>
      )}
    </span>
  );
}

/* ---------------- JSONView ---------------- */
function JSONView({ obj }: { obj: any }) {
  const html = useMemo(() => {
    const str = JSON.stringify(obj, null, 2) || "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(
        /("(\\u[\da-fA-F]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(\.\d+)?([eE][+-]?\d+)?)/g,
        (m) => {
          let cls = "n";
          if (/^"/.test(m)) cls = /:$/.test(m) ? "k" : "s";
          else if (/true|false/.test(m)) cls = "b";
          else if (/null/.test(m)) cls = "b";
          return `<span class="${cls}">${m}</span>`;
        }
      );
  }, [obj]);
  return (
    <pre
      className="json-pre font-mono text-[11.5px] leading-[1.55] rounded-lg p-3.5 max-h-[280px] overflow-auto whitespace-pre"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

/* ---------------- Toast ---------------- */
function Toast({ msg }: { msg: string }) {
  return (
    <div className={cn("toast-wrap", msg && "show")}>
      <div className="inline-flex items-center gap-2.5 bg-ink-text text-white px-3.5 py-2.5 rounded-[10px] text-[13px] font-medium shadow-soft-lg">
        <span className="text-aqua-300">{Icon.check}</span>
        {msg}
      </div>
    </div>
  );
}

/* ---------------- Section header ---------------- */
function SectionHead({ title, hint, className }: { title: string; hint?: string; className?: string }) {
  return (
    <div className={cn("flex items-baseline gap-3 mt-7 mb-3 mx-0.5", className)}>
      <h2 className="m-0 text-[13px] font-semibold tracking-[0.02em] uppercase text-ink-2">{title}</h2>
      {hint && <span className="text-xs text-ink-3">{hint}</span>}
    </div>
  );
}

/* ---------------- SKU Ledger ---------------- */
function SKULedger({ rows }: { rows: TopSKU[] }) {
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<{ key: keyof TopSKU; dir: "asc" | "desc" }>({ key: "hits", dir: "desc" });

  const filtered = useMemo(() => {
    const ql = q.trim().toLowerCase();
    let r = rows.filter(
      (x) => !ql || x.sku.toLowerCase().includes(ql) || (x.product || "").toLowerCase().includes(ql)
    );
    r = [...r].sort((a, b) => {
      const k = sort.key;
      const d = sort.dir === "asc" ? 1 : -1;
      const av = a[k] as any, bv = b[k] as any;
      if (typeof av === "number") return (av - bv) * d;
      return String(av).localeCompare(String(bv)) * d;
    });
    return r;
  }, [rows, q, sort]);

  const SortHead = ({ col, children, align }: { col: keyof TopSKU; children: React.ReactNode; align?: "right" }) => (
    <th
      className={cn(
        "sticky top-0 z-[1] bg-[#fafbfc] text-ink-3 font-semibold text-[11px] uppercase tracking-[0.06em] py-2.5 px-3.5 border-b border-rule select-none cursor-pointer",
        align === "right" && "text-right",
        sort.key === col && "text-ink-text"
      )}
      onClick={() =>
        setSort((s) => ({ key: col, dir: s.key === col && s.dir === "desc" ? "asc" : "desc" }))
      }
    >
      {children}{" "}
      <span className="inline-block ml-1 text-ink-4">
        {sort.key === col ? (sort.dir === "desc" ? "↓" : "↑") : "↕"}
      </span>
    </th>
  );

  return (
    <div className="card-outline flex flex-col min-h-0 col-span-12 lg:col-span-7 bg-card border border-rule rounded-[10px] shadow-soft-sm">
      <div className="flex items-center gap-2.5 p-3.5 px-4 border-b border-rule">
        <h3 className="m-0 text-sm font-semibold">Top Retrieved SKUs</h3>
        <span className="text-xs text-ink-3 font-medium px-1.5 py-0.5 bg-[#f1f3f6] rounded-full">
          {filtered.length.toLocaleString()} / {rows.length.toLocaleString()}
        </span>
        <div className="ml-auto relative">
          <span className="absolute left-2 top-1/2 -translate-y-1/2 text-ink-3">{Icon.search}</span>
          <input
            className="w-56 pl-7 pr-2.5 py-1.5 border border-rule rounded-md text-[12.5px] bg-card text-ink-text focus:outline-none focus:border-ink-700 focus:ring-[3px] focus:ring-ink-100"
            placeholder="Search SKU or product…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
        </div>
      </div>
      <div className="overflow-auto max-h-[520px]">
        <table className="w-full border-separate border-spacing-0 text-[12.5px]">
          <thead>
            <tr>
              <SortHead col="sku">SKU</SortHead>
              <SortHead col="product">Product</SortHead>
              <SortHead col="hits" align="right">Hits</SortHead>
              <SortHead col="frequency" align="right">% of queries</SortHead>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && (
              <tr>
                <td colSpan={4} className="py-8 text-center text-ink-3 text-[13px]">
                  {rows.length === 0 ? "No SKU data yet — run some queries to populate." : "No SKUs match your search."}
                </td>
              </tr>
            )}
            {filtered.map((r) => (
              <tr key={r.sku} className="hover:bg-[#f7f9fc] transition-colors">
                <td className="py-2.5 px-3.5 border-b border-[#f0f1f4] font-mono text-[12px] text-ink-700 font-medium">
                  {r.sku}
                </td>
                <td className="py-2.5 px-3.5 border-b border-[#f0f1f4] text-ink-text font-medium">
                  {r.product || "—"}
                </td>
                <td className="py-2.5 px-3.5 border-b border-[#f0f1f4] text-right text-ink-text font-medium tabular-nums">
                  {r.hits.toLocaleString()}
                </td>
                <td className="py-2.5 px-3.5 border-b border-[#f0f1f4] text-right text-ink-text font-medium tabular-nums">
                  {r.frequency}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center gap-2.5 py-2.5 px-3.5 border-t border-rule text-xs text-ink-3">
        <span>Top {Math.min(rows.length, filtered.length)} from live traces · rebuilt on refresh</span>
      </div>
    </div>
  );
}

/* ---------------- Recent Queries ---------------- */
function RecentQueries({
  rows, onOpen, selectedId,
}: {
  rows: RecentQuery[];
  onOpen: (r: RecentQuery) => void;
  selectedId: string | null;
}) {
  const [q, setQ] = useState("");
  const [intentFilter, setIntentFilter] = useState<string[]>([]);
  const [open, setOpen] = useState(false);

  const uniqueIntents = useMemo(() => {
    const set = new Set<string>();
    rows.forEach((r) => r.intent && set.add(r.intent));
    return Array.from(set).sort();
  }, [rows]);

  const filtered = useMemo(() => {
    const ql = q.trim().toLowerCase();
    return rows
      .filter(
        (r) =>
          (intentFilter.length === 0 || intentFilter.includes(r.intent)) &&
          (!ql || r.query.toLowerCase().includes(ql) || r.id.toLowerCase().includes(ql))
      )
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }, [rows, q, intentFilter]);

  return (
    <div className="flex flex-col min-h-0 col-span-12 lg:col-span-5 bg-card border border-rule rounded-[10px] shadow-soft-sm">
      <div className="flex items-center gap-2.5 p-3.5 px-4 border-b border-rule">
        <h3 className="m-0 text-sm font-semibold">Recent Queries</h3>
        <span className="text-xs text-ink-3 font-medium px-1.5 py-0.5 bg-[#f1f3f6] rounded-full">
          {filtered.length.toLocaleString()}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <div className="relative">
            <span className="absolute left-2 top-1/2 -translate-y-1/2 text-ink-3">{Icon.search}</span>
            <input
              className="w-[180px] pl-7 pr-2.5 py-1.5 border border-rule rounded-md text-[12.5px] bg-card text-ink-text focus:outline-none focus:border-ink-700 focus:ring-[3px] focus:ring-ink-100"
              placeholder="Search queries…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
          </div>
          <div className="relative">
            <button
              className={cn(
                "inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium border",
                intentFilter.length > 0
                  ? "bg-ink-50 text-ink-700 border-ink-100"
                  : "bg-card text-ink-2 border-rule hover:border-rule-strong hover:text-ink-text"
              )}
              onClick={() => setOpen((v) => !v)}
            >
              {Icon.filter}Intent{intentFilter.length > 0 && ` · ${intentFilter.length}`}
            </button>
            {open && (
              <FilterMenu
                label="Filter by intent"
                options={uniqueIntents.map((v) => ({ value: v, label: INTENT_LABEL[v] || v }))}
                selected={intentFilter}
                onChange={setIntentFilter}
                onClose={() => setOpen(false)}
              />
            )}
          </div>
        </div>
      </div>

      {intentFilter.length > 0 && (
        <div className="flex gap-1.5 px-4 pt-2.5 flex-wrap">
          {intentFilter.map((t) => (
            <Chip key={t} onRemove={() => setIntentFilter(intentFilter.filter((v) => v !== t))}>
              {INTENT_LABEL[t] || t}
            </Chip>
          ))}
          <Chip muted onRemove={() => setIntentFilter([])}>Clear all</Chip>
        </div>
      )}

      <div className="overflow-auto max-h-[520px]">
        <table className="w-full border-separate border-spacing-0 text-[12.5px]">
          <thead>
            <tr>
              <th className="sticky top-0 z-[1] bg-[#fafbfc] text-ink-3 font-semibold text-[11px] uppercase tracking-[0.06em] py-2.5 px-3.5 border-b border-rule text-left" style={{ width: 130 }}>Intent</th>
              <th className="sticky top-0 z-[1] bg-[#fafbfc] text-ink-3 font-semibold text-[11px] uppercase tracking-[0.06em] py-2.5 px-3.5 border-b border-rule text-left">Query</th>
              <th className="sticky top-0 z-[1] bg-[#fafbfc] text-ink-3 font-semibold text-[11px] uppercase tracking-[0.06em] py-2.5 px-3.5 border-b border-rule text-right" style={{ width: 90 }}>Time</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && (
              <tr>
                <td colSpan={3} className="py-8 text-center text-ink-3 text-[13px]">
                  {rows.length === 0 ? "No queries yet — they'll appear here as they arrive." : "No queries match your filters."}
                </td>
              </tr>
            )}
            {filtered.map((r) => {
              const b = intentBadge(r.intent);
              const selected = selectedId === r.id;
              return (
                <tr
                  key={r.id}
                  className={cn(
                    "cursor-pointer transition-colors",
                    selected ? "bg-ink-50" : "hover:bg-[#f7f9fc]"
                  )}
                  onClick={() => onOpen(r)}
                >
                  <td className="py-2.5 px-3.5 border-b border-[#f0f1f4]">
                    <span
                      className={cn(
                        "inline-flex items-center rounded-full px-2 py-0.5 text-[11.5px] font-medium border capitalize",
                        b.cls
                      )}
                    >
                      {b.label}
                    </span>
                  </td>
                  <td
                    className="py-2.5 px-3.5 border-b border-[#f0f1f4] text-ink-text truncate max-w-[380px] hover:text-ink-700"
                    title={r.query}
                  >
                    {r.query}
                  </td>
                  <td className="py-2.5 px-3.5 border-b border-[#f0f1f4] text-right text-ink-3 text-xs tabular-nums">
                    {timeAgoISO(r.timestamp)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="flex items-center gap-2.5 py-2.5 px-3.5 border-t border-rule text-xs text-ink-3">
        <span>Click any row to open the trace drawer</span>
      </div>
    </div>
  );
}

/* ---------------- Drawer ---------------- */
function Drawer({ query, onClose }: { query: RecentQuery | null; onClose: () => void }) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!query) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [query, onClose]);

  const meta = query && {
    trace_id: query.id,
    timestamp: query.timestamp,
    intent: query.intent,
    intent_confidence: query.intent_confidence,
    latency_sec: query.latency,
    path: query.path,
    tools_used: query.tools_used,
    status: query.status,
    input: query.rawInput,
    output: query.rawOutput,
  };

  function copy() {
    if (!meta) return;
    try { navigator.clipboard?.writeText(JSON.stringify(meta, null, 2)); } catch {}
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  }

  const b = query ? intentBadge(query.intent) : null;

  const steps = query
    ? [
        {
          t: "Intent classification",
          tag: "intent.classify",
          d: `${b?.label} · confidence ${(query.intent_confidence || 0).toFixed(2)}`,
        },
        {
          t: "Retrieval",
          tag: query.path === "tools" ? "tool_loop" : "hybrid_search",
          d: query.rawOutput?.retrieval_success === false
            ? "No results retrieved"
            : `Retrieved ${query.rawOutput?.retrieved_count ?? "?"} chunks · latency ${
                Number(query.rawOutput?.retrieval_latency_ms ?? 0).toFixed(0)
              }ms`,
        },
        {
          t: "Answer compose",
          tag: "llm.compose",
          d: `${query.latency_known ? `${query.latency.toFixed(2)}s` : "—"} total · ${
            query.path === "tools" ? "tool-calling path" : "streaming path"
          }`,
        },
      ]
    : [];

  const citedSkus = query
    ? Object.entries((query.rawOutput?.sku_counts_in_answer as Record<string, number>) || {}).map(
        ([sku, count]) => ({
          sku,
          name: (query.rawOutput?.sku_product_names as Record<string, string>)?.[sku] || sku,
          count,
        })
      )
    : [];

  return (
    <>
      <div className={cn("scrim", query && "open")} onClick={onClose} />
      <aside className={cn("drawer", query && "open")} aria-hidden={!query}>
        {query && b && (
          <>
            <div className="flex items-center gap-2.5 p-4 px-5 border-b border-rule">
              <h3 className="m-0 text-sm font-semibold">Query Detail</h3>
              <span
                className={cn(
                  "inline-flex items-center rounded-full px-2 py-0.5 text-[11.5px] font-medium border capitalize",
                  b.cls
                )}
              >
                {b.label}
              </span>
              <div className="flex-1" />
              <button
                className="h-8 w-8 rounded-md border border-rule bg-card grid place-items-center text-ink-2 hover:border-rule-strong hover:text-ink-text"
                onClick={onClose}
                aria-label="Close"
              >
                {Icon.x}
              </button>
            </div>

            <div className="flex-1 overflow-y-auto">
              <div className="p-4 px-5 border-b border-rule">
                <h4 className="m-0 mb-2.5 text-[11px] font-semibold text-ink-3 uppercase tracking-[0.08em]">
                  Query Overview
                </h4>
                <div className="bg-ink-50 border border-ink-100 text-ink-900 px-3 py-2.5 rounded-lg text-[13px] font-medium leading-snug">
                  {query.query}
                </div>
                <div className="grid grid-cols-[110px_1fr] gap-x-3.5 gap-y-1.5 mt-2.5 text-[12.5px]">
                  <div className="text-ink-3">Timestamp</div>
                  <div className="text-ink-text font-mono text-[12px]">
                    {new Date(query.timestamp).toLocaleString()}
                  </div>
                  <div className="text-ink-3">Trace ID</div>
                  <div className="text-ink-text font-mono text-[12px]">{query.id}</div>
                  <div className="text-ink-3">Latency</div>
                  <div className="text-ink-text font-mono text-[12px]">
                    {query.latency_known ? `${query.latency.toFixed(2)} s` : "—"}
                  </div>
                  <div className="text-ink-3">Path</div>
                  <div className="text-ink-text font-mono text-[12px]">{query.path}</div>
                </div>
              </div>

              <div className="p-4 px-5 border-b border-rule">
                <h4 className="m-0 mb-2.5 text-[11px] font-semibold text-ink-3 uppercase tracking-[0.08em]">
                  Processing Path
                </h4>
                <div className="flex flex-col gap-2.5 border-l-2 border-rule pl-3.5 my-1.5">
                  {steps.map((s, i) => (
                    <div key={i} className="relative text-[12.5px]">
                      <div className="absolute -left-[19px] top-1.5 h-2 w-2 rounded-full bg-aqua-500" />
                      <div className="font-semibold text-ink-text flex gap-2 items-baseline">
                        {s.t}
                        <small className="font-medium text-ink-3 font-mono">{s.tag}</small>
                      </div>
                      <div className="text-ink-2 mt-0.5">{s.d}</div>
                    </div>
                  ))}
                </div>

                {query.tools_used.length > 0 && (
                  <div className="mt-3.5">
                    <div className="text-[11px] font-semibold text-ink-3 uppercase tracking-[0.06em] mb-2">
                      Tools Used
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {query.tools_used.map((t, i) => (
                        <span
                          key={`${t}-${i}`}
                          className="font-mono text-[11.5px] px-2 py-0.5 bg-[#f3f4f7] border border-rule rounded text-ink-text"
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {citedSkus.length > 0 && (
                  <div className="mt-3.5">
                    <div className="text-[11px] font-semibold text-ink-3 uppercase tracking-[0.06em] mb-2">
                      SKUs Cited ({citedSkus.length})
                    </div>
                    <div className="flex flex-col gap-1.5">
                      {citedSkus.map((s) => (
                        <div
                          key={s.sku}
                          className="flex items-center gap-2.5 p-2 px-2.5 border border-rule rounded-md text-[12.5px] bg-[#fafbfc]"
                        >
                          <span className="font-mono text-ink-700 font-medium">{s.sku}</span>
                          <span className="text-ink-text truncate">{s.name}</span>
                          <span className="ml-auto text-ink-3 tabular-nums">cited {s.count}×</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="p-4 px-5">
                <div className="flex items-center mb-2.5">
                  <h4 className="m-0 text-[11px] font-semibold text-ink-3 uppercase tracking-[0.08em]">
                    Raw Metadata
                  </h4>
                  <button
                    className={cn(
                      "ml-auto inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11.5px] font-medium",
                      copied
                        ? "text-good border border-[#cfeadd] bg-good-bg"
                        : "text-ink-2 border border-rule bg-card hover:border-rule-strong hover:text-ink-text"
                    )}
                    onClick={copy}
                  >
                    {copied ? Icon.check : Icon.copy}
                    {copied ? "Copied" : "Copy JSON"}
                  </button>
                </div>
                <JSONView obj={meta} />
              </div>
            </div>
          </>
        )}
      </aside>
    </>
  );
}

/* ---------------- Page ---------------- */
export default function Page() {
  const router = useRouter();
  const [range, setRange] = useState<Range>("7d");
  const [data, setData] = useState<AnalyticsPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loggingOut, setLoggingOut] = useState(false);
  const [openQuery, setOpenQuery] = useState<RecentQuery | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [notifOpen, setNotifOpen] = useState(false);
  const [brandMode, setBrandMode] = useState<"hits" | "share">("hits");
  const [subcatMode, setSubcatMode] = useState<"hits" | "share">("hits");
  const [toast, setToast] = useState("");
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const flash = (msg: string) => {
    setToast(msg);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(""), 2200);
  };

  const LoadingSkeleton = () => (
    <div className="splash-overlay" role="status" aria-live="polite" aria-label="Loading analytics">
      <div className="splash-card splash-card-compact">
        <div className="skeleton skeleton-icon" aria-hidden="true" />
        <div className="skeleton skeleton-title" aria-hidden="true" />
        <div className="skeleton skeleton-subtitle" aria-hidden="true" />
      </div>
    </div>
  );

  const fetchData = async (r: Range) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/admin-api/analytics?range=${r}`);
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || "Failed to fetch analytics");
      }
      const j = (await res.json()) as AnalyticsPayload;
      setData(j);
    } catch (e: any) {
      setError(e.message || "Unknown error");
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(range); }, [range]);

  const onLogout = async () => {
    if (loggingOut) return;
    setLoggingOut(true);
    try { await signOut(); } catch {}
    router.replace("/login");
    router.refresh();
    setLoggingOut(false);
  };

  const kpis = data?.kpis;
  const volume = data?.charts.volume || [];
  const topBrands = data?.charts.topBrands || [];
  const topSubcats = data?.charts.topSubcategories || [];
  const topCats = data?.charts.topCategories || [];
  const intents = data?.charts.intent || [];
  const languages = (data?.charts.languages || []).map((l) => ({ ...l, color: "#37BAD1" }));
  const topSKUs = data?.topSKUs || [];
  const recent = data?.recentQueries || [];

  const brandSum = topBrands.reduce((s, d) => s + d.value, 0) || 1;
  const subcatSum = topSubcats.reduce((s, d) => s + d.value, 0) || 1;
  const brandData = brandMode === "share"
    ? topBrands.map((d) => ({ ...d, value: +((d.value / brandSum) * 100).toFixed(1) }))
    : topBrands;
  const subcatData = subcatMode === "share"
    ? topSubcats.map((d) => ({ ...d, value: +((d.value / subcatSum) * 100).toFixed(1) }))
    : topSubcats;

  // Notifications derived from live state.
  const notifs = useMemo(() => {
    if (!kpis) return [];
    const items: { id: string; level: "crit" | "warn" | ""; title: string; desc: string; t: string }[] = [];
    if (kpis.p95_latency_sec > 5) {
      items.push({
        id: "p95",
        level: "crit",
        title: "p95 latency above threshold",
        desc: `p95 ${kpis.p95_latency_sec.toFixed(2)}s over last ${range} (threshold 5s)`,
        t: "now",
      });
    }
    if (kpis.fallback_rate >= 10) {
      items.push({
        id: "fallback",
        level: "warn",
        title: "Elevated fallback rate",
        desc: `${kpis.fallback_rate}% of queries fell back to system defaults`,
        t: "now",
      });
    }
    if (kpis.zero_result_rate >= 10) {
      items.push({
        id: "zero",
        level: "warn",
        title: "Zero-result rate elevated",
        desc: `${kpis.zero_result_rate}% of queries returned no chunks`,
        t: "now",
      });
    }
    if (kpis.low_confidence_rate >= 20) {
      items.push({
        id: "lowconf",
        level: "",
        title: "Low-confidence queries detected",
        desc: `${kpis.low_confidence_rate}% of queries classified with confidence < 0.6`,
        t: "now",
      });
    }
    if (items.length === 0) {
      items.push({
        id: "ok",
        level: "",
        title: "All systems nominal",
        desc: `p50 ${kpis.p50_latency_sec.toFixed(2)}s · fallback ${kpis.fallback_rate}% · zero-result ${kpis.zero_result_rate}%`,
        t: "now",
      });
    }
    return items;
  }, [kpis, range]);

  const unreadNotifs = notifs.filter((n) => n.level !== "").length;

  const sparkTotal = buildSpark(volume, "total");
  const sparkPurchase = buildSpark(volume, "purchase_rate");
  const sparkBasket = buildSpark(volume, "basket_build");
  const sparkCompare = buildSpark(volume, "price_compare");

  return (
    <div className="min-h-screen">
      {loading && !data && <LoadingSkeleton />}
      {/* Top bar */}
      <div className="sticky top-0 z-20 bg-[rgba(247,248,250,0.85)] backdrop-blur border-b border-rule">
        <div className="mx-auto w-full max-w-[1600px] px-7 py-3.5 flex items-center gap-4">
        <div className="text-[12.5px] text-ink-3">
          Analytics <span className="px-1.5 text-ink-4">/</span>
          <b className="text-ink-text font-semibold">AI Query Performance</b>
        </div>
        <div className="flex-1" />
        <div className="inline-flex items-center gap-2 px-2.5 py-1.5 border border-rule rounded-md bg-card text-[12.5px] font-medium text-ink-2">
          <span className={cn("h-1.5 w-1.5 rounded-full", loading ? "bg-ink-3" : "bg-aqua-500")} />
          {new Date().toLocaleString("en-US", { month: "short", day: "numeric" })} · {loading ? "loading" : "live"}
        </div>
        <div className="inline-flex bg-card border border-rule rounded-md p-[3px]">
          {(["24h", "7d", "30d"] as Range[]).map((r) => (
            <button
              key={r}
              className={cn(
                "px-2.5 py-1 rounded text-xs font-medium",
                range === r ? "bg-ink-700 text-white" : "text-ink-2 bg-transparent"
              )}
              onClick={() => setRange(r)}
            >
              {r}
            </button>
          ))}
        </div>
        <button
          className="h-8 w-8 rounded-md border border-rule bg-card grid place-items-center text-ink-2 hover:border-rule-strong hover:text-ink-text disabled:opacity-40"
          title="Refresh"
          onClick={() => fetchData(range)}
          disabled={loading}
        >
          <span className={cn(loading && "animate-spin")}>{Icon.refresh}</span>
        </button>
        <div className="relative">
          <button
            className="h-8 w-8 rounded-md border border-rule bg-card grid place-items-center text-ink-2 hover:border-rule-strong hover:text-ink-text"
            title="Export"
            onClick={() => { setExportOpen((v) => !v); setNotifOpen(false); }}
          >
            {Icon.dl}
          </button>
          <Popover open={exportOpen} onClose={() => setExportOpen(false)}>
            <div className="flex items-center gap-2.5 py-3 px-3.5 border-b border-rule">
              <h4 className="m-0 text-[13px] font-semibold">Export current view</h4>
            </div>
            <div className="py-1">
              {([
                ["CSV", "Tables & KPI values", Icon.csv],
                ["PDF", "Print-ready snapshot", Icon.pdf],
                ["PNG", "Charts as image", Icon.png],
                ["JSON", "Raw data + metadata", Icon.copy],
                ["Copy share link", "Filtered view · expires in 7d", Icon.link],
              ] as const).map(([t, sub, ic]) => (
                <div
                  key={t}
                  className="flex items-center gap-2.5 px-3.5 py-2.5 text-[13px] cursor-pointer hover:bg-[#f4f5f8]"
                  onClick={() => { setExportOpen(false); flash(`Exporting current view as ${t}…`); }}
                >
                  <span className="h-7 w-7 rounded-md bg-ink-50 text-ink-700 grid place-items-center flex-none">{ic}</span>
                  <span className="flex-1">
                    {t}
                    <small className="block text-ink-3 text-[11.5px] mt-0.5">{sub}</small>
                  </span>
                  {Icon.chev}
                </div>
              ))}
            </div>
            <div className="py-2 px-3.5 border-t border-rule text-xs text-ink-3">
              Range: last {range}
            </div>
          </Popover>
        </div>
        <div className="relative">
          <div className="relative">
            <button
              className="h-8 w-8 rounded-md border border-rule bg-card grid place-items-center text-ink-2 hover:border-rule-strong hover:text-ink-text"
              title="Notifications"
              onClick={() => { setNotifOpen((v) => !v); setExportOpen(false); }}
            >
              {Icon.bell}
            </button>
            {unreadNotifs > 0 && (
              <span className="absolute -top-0.5 -right-0.5 h-[9px] w-[9px] rounded-full bg-aqua-500 border-2 border-canvas" />
            )}
          </div>
          <Popover open={notifOpen} onClose={() => setNotifOpen(false)} className="w-[360px]">
            <div className="flex items-center gap-2.5 py-3 px-3.5 border-b border-rule">
              <h4 className="m-0 text-[13px] font-semibold">Notifications</h4>
              {unreadNotifs > 0 && (
                <span className="text-[11px] px-1.5 py-0.5 bg-[#f1f3f6] rounded-full text-ink-2">
                  {unreadNotifs} new
                </span>
              )}
            </div>
            <div className="max-h-[380px] overflow-y-auto">
              {notifs.map((n) => (
                <div
                  key={n.id}
                  className="grid grid-cols-[8px_1fr_auto] gap-2.5 items-start py-3 px-3.5 border-b border-[#f0f1f4] last:border-b-0 hover:bg-[#fafbfc]"
                >
                  <span
                    className={cn(
                      "h-2 w-2 rounded-full mt-1.5",
                      n.level === "crit" ? "bg-bad" : n.level === "warn" ? "bg-[#d68800]" : "bg-aqua-500"
                    )}
                  />
                  <div>
                    <div className="text-[13px] font-medium text-ink-text">{n.title}</div>
                    <div className="text-xs text-ink-2 mt-0.5 leading-snug">{n.desc}</div>
                  </div>
                  <span className="text-[11px] text-ink-3 tabular-nums whitespace-nowrap">{n.t}</span>
                </div>
              ))}
            </div>
            <div className="py-2 px-3.5 border-t border-rule text-xs text-ink-3 flex">
              <span>Derived from live telemetry</span>
            </div>
          </Popover>
        </div>
        <button
          className="h-8 px-2.5 rounded-md border border-rule bg-card inline-flex items-center gap-1.5 text-[12.5px] text-ink-2 hover:border-rule-strong hover:text-ink-text disabled:opacity-50"
          onClick={onLogout}
          disabled={loggingOut}
          title="Log out"
        >
          {Icon.logout}
          {loggingOut ? "…" : "Logout"}
        </button>
        </div>
      </div>

      {/* Content */}
      <div className="px-7 pt-6 pb-12 max-w-[1600px] mx-auto w-full">
        {error && !data && (
          <div className="mb-4 rounded-md border border-bad/30 bg-bad-bg text-bad px-3.5 py-2.5 text-[12.5px]">
            {error} — no data to display yet. Run some chatbot queries, then refresh.
          </div>
        )}

        {/* Primary KPIs */}
        <SectionHead
          title="Primary Metrics"
          hint="Signals used to drive weekly decisions"
          className="first:mt-1"
        />
        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-12 md:col-span-4">
            <KPI
              primary
              label="Purchase Intent"
              value={kpis ? `${kpis.purchase_intent_rate}` : "—"}
              unit={kpis ? "%" : undefined}
              sub={kpis?.top_brand ? `Top brand: ${kpis.top_brand}` : "Share of purchase-like queries"}
              spark={sparkPurchase}
            />
          </div>
          <div className="col-span-12 md:col-span-4">
            <KPI
              primary
              label="Total Queries"
              value={kpis ? formatNum(kpis.total_queries) : "—"}
              sub={`Range: last ${range}`}
              spark={sparkTotal}
            />
          </div>
          <div className="col-span-12 md:col-span-4">
            <KPI
              primary
              label="SKU Citation Rate"
              value={kpis ? `${kpis.sku_hit_rate}` : "—"}
              unit={kpis ? "%" : undefined}
              sub={kpis?.top_category ? `Top category: ${kpis.top_category}` : "Queries citing ≥1 SKU"}
              sparkAlt
              spark={sparkTotal}
            />
          </div>
        </div>

        <SectionHead title="Secondary Metrics" hint="Supporting behaviours — de-emphasized" />
        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-12 md:col-span-6">
            <KPI
              label="Basket Builds"
              value={kpis ? formatNum(kpis.basket_queries) : "—"}
              sub={kpis && kpis.avg_basket_budget_eur > 0
                ? `Avg budget €${kpis.avg_basket_budget_eur.toFixed(2)}`
                : "auto-compiled baskets"}
              spark={sparkBasket}
            />
          </div>
          <div className="col-span-12 md:col-span-6">
            <KPI
              label="Comparison Requests"
              value={kpis ? formatNum(kpis.comparison_queries) : "—"}
              sub={kpis
                ? `p50 ${kpis.p50_latency_sec.toFixed(2)}s · p95 ${kpis.p95_latency_sec.toFixed(2)}s`
                : "compare-intent queries"}
              spark={sparkCompare}
            />
          </div>
        </div>

        {/* Charts */}
        <SectionHead title="Graph Insights" hint="Retrieval and query distribution" />
        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-12 lg:col-span-6 bg-card border border-rule rounded-[10px] shadow-soft-sm p-5 flex flex-col gap-3.5">
            <div className="flex items-start gap-3">
              <div>
                <h3 className="m-0 text-sm font-semibold tracking-[-0.01em] text-ink-text">Most Retrieved Brands</h3>
                <div className="text-ink-3 text-xs mt-0.5">Top brands cited in retrieval · last {range}</div>
              </div>
              <div className="ml-auto">
                <div className="inline-flex bg-[#f3f4f7] rounded-md p-0.5">
                  {(["hits", "share"] as const).map((m) => (
                    <button
                      key={m}
                      className={cn(
                        "px-2.5 py-1 rounded text-[11.5px] font-medium capitalize",
                        brandMode === m
                          ? "bg-card text-ink-text shadow-soft-sm"
                          : "text-ink-2 bg-transparent"
                      )}
                      onClick={() => setBrandMode(m)}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <HBars data={brandData} suffix={brandMode === "share" ? "%" : ""} />
          </div>

          <div className="col-span-12 lg:col-span-6 bg-card border border-rule rounded-[10px] shadow-soft-sm p-5 flex flex-col gap-3.5">
            <div className="flex items-start gap-3">
              <div>
                <h3 className="m-0 text-sm font-semibold tracking-[-0.01em] text-ink-text">Top Subcategories</h3>
                <div className="text-ink-3 text-xs mt-0.5">Subcategories appearing most in retrieval</div>
              </div>
              <div className="ml-auto">
                <div className="inline-flex bg-[#f3f4f7] rounded-md p-0.5">
                  {(["hits", "share"] as const).map((m) => (
                    <button
                      key={m}
                      className={cn(
                        "px-2.5 py-1 rounded text-[11.5px] font-medium capitalize",
                        subcatMode === m
                          ? "bg-card text-ink-text shadow-soft-sm"
                          : "text-ink-2 bg-transparent"
                      )}
                      onClick={() => setSubcatMode(m)}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <HBars data={subcatData} alt suffix={subcatMode === "share" ? "%" : ""} />
          </div>

          <div className="col-span-12 md:col-span-6 lg:col-span-4 bg-card border border-rule rounded-[10px] shadow-soft-sm p-5 flex flex-col gap-3.5">
            <div className="flex items-start">
              <div>
                <h3 className="m-0 text-sm font-semibold tracking-[-0.01em] text-ink-text">Category Distribution</h3>
                <div className="text-ink-3 text-xs mt-0.5">Share of total retrieval volume</div>
              </div>
            </div>
            <div className="grid grid-cols-[140px_1fr] gap-4 items-center">
              <Donut data={topCats} />
              <Legend data={topCats} />
            </div>
          </div>

          <div className="col-span-12 md:col-span-6 lg:col-span-4 bg-card border border-rule rounded-[10px] shadow-soft-sm p-5 flex flex-col gap-3.5">
            <div className="flex items-start">
              <div>
                <h3 className="m-0 text-sm font-semibold tracking-[-0.01em] text-ink-text">Intent Breakdown</h3>
                <div className="text-ink-3 text-xs mt-0.5">Classifier output share</div>
              </div>
            </div>
            <div className="grid grid-cols-[140px_1fr] gap-4 items-center">
              <Donut data={intents.map((i) => ({ ...i, name: INTENT_LABEL[i.name] || i.name }))} />
              <Legend data={intents.map((i) => ({ ...i, name: INTENT_LABEL[i.name] || i.name }))} />
            </div>
          </div>

          <div className="col-span-12 md:col-span-6 lg:col-span-4 bg-card border border-rule rounded-[10px] shadow-soft-sm p-5 flex flex-col gap-3.5">
            <div className="flex items-start">
              <div>
                <h3 className="m-0 text-sm font-semibold tracking-[-0.01em] text-ink-text">Query Language</h3>
                <div className="text-ink-3 text-xs mt-0.5">Top languages served</div>
              </div>
            </div>
            <HBars data={languages} alt />
          </div>

          <div className="col-span-12 bg-card border border-rule rounded-[10px] shadow-soft-sm p-5 flex flex-col gap-3.5">
            <div className="flex items-start">
              <div>
                <h3 className="m-0 text-sm font-semibold tracking-[-0.01em] text-ink-text">Query Volume by Hour</h3>
                <div className="text-ink-3 text-xs mt-0.5">Aggregated across last {range} · UTC</div>
              </div>
            </div>
            {volume.length > 0
              ? <HourBars data={volume} />
              : <div className="py-10 text-center text-ink-3 text-sm">No volume data for this range.</div>}
          </div>
        </div>

        {/* Tables */}
        <SectionHead title="Data Tables" hint="Click any query to open the trace drawer" />
        <div className="grid grid-cols-12 gap-4">
          <SKULedger rows={topSKUs} />
          <RecentQueries rows={recent} onOpen={setOpenQuery} selectedId={openQuery?.id || null} />
        </div>
      </div>

      <Drawer query={openQuery} onClose={() => setOpenQuery(null)} />
      <Toast msg={toast} />
    </div>
  );
}
