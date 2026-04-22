import { NextResponse } from "next/server";
import { langfuse } from "@/lib/langfuse";

/**
 * GET /admin-api/analytics
 *
 * Aggregates Langfuse traces into the dashboard payload. All fields below are
 * derived from what backend/app.py logs via span.update():
 *
 *   input:  query, intent, intent_confidence, user_language
 *   output: retrieval_latency_ms, retrieval_success, retrieved_count,
 *           retrieved_brands, retrieved_categories, retrieved_subcategories,
 *           retrieved_skus, sku_product_names, sku_counts_in_answer,
 *           answer, fallback_triggered, no_result, tool_trace (tools path)
 *   trace:  latency (seconds, float), timestamp (ISO)
 *
 * If a field is missing (older traces, OCR-era data), the aggregation silently
 * skips it — dashboards still render, empty charts are replaced with mocks
 * client-side.
 */
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const range = searchParams.get("range") || "24h";

  const now = new Date();
  const from = new Date();
  if (range === "24h") from.setHours(now.getHours() - 24);
  else if (range === "7d") from.setDate(now.getDate() - 7);
  else if (range === "30d") from.setDate(now.getDate() - 30);

  try {
    const tracesResponse = await langfuse.api.traceList({
      limit: 100,
      fromTimestamp: from.toISOString(),
      toTimestamp: now.toISOString(),
    });

    const traces = tracesResponse.data;

    if (!traces || traces.length === 0) {
      return NextResponse.json({ error: "No traces found for this period" }, { status: 404 });
    }

    // ---------------- Per-hour buckets ----------------
    const hourlyVolume: Record<
      string,
      {
        hour: string;
        product_search: number;
        product_recommendation: number;
        price_compare: number;
        basket_build: number;
        barcode_lookup: number;
        order_status: number;
        general_qa: number;
        other: number;
      }
    > = {};
    const confidenceSequence: Record<string, { hour: string; confidence: number; count: number }> = {};
    for (let i = 0; i < 24; i++) {
      const h = i.toString().padStart(2, "0");
      hourlyVolume[h] = {
        hour: h,
        product_search: 0,
        product_recommendation: 0,
        price_compare: 0,
        basket_build: 0,
        barcode_lookup: 0,
        order_status: 0,
        general_qa: 0,
        other: 0,
      };
      confidenceSequence[h] = { hour: h, confidence: 0, count: 0 };
    }

    // ---------------- Aggregators ----------------
    const intentCounts: Record<string, number> = {};
    const languageCounts: Record<string, number> = {};
    const latencyBuckets: Record<string, number> = { "<2s": 0, "2-5s": 0, "5-10s": 0, "10-20s": 0, "20-30s": 0, ">30s": 0 };
    const topSKUsMap: Record<string, { product: string; hits: number }> = {};

    const brandCounts: Record<string, number> = {};
    const categoryCounts: Record<string, number> = {};
    const subcategoryCounts: Record<string, number> = {};
    const toolCounts: Record<string, number> = {};
    const pathCounts: Record<string, number> = { stream: 0, tools: 0 };

    const latenciesSec: number[] = [];
    let zeroResults = 0;
    let fallbackTriggered = 0;
    let lowConfidence = 0; // intent_confidence < 0.6
    let totalConfidence = 0;
    let skuHits = 0;
    let purchaseIntent = 0;
    let basketQueries = 0;
    let basketBudgetSum = 0;
    let basketBudgetCount = 0;
    let comparisonQueries = 0;

    traces.forEach((trace: any) => {
      let inputData: any = {};
      let outputData: any = {};
      try {
        inputData = typeof trace.input === "string" ? JSON.parse(trace.input) : trace.input || {};
      } catch (e) {
        console.warn(`[analytics] failed to parse trace.input for ${trace.id}:`, e);
      }
      try {
        outputData = typeof trace.output === "string" ? JSON.parse(trace.output) : trace.output || {};
      } catch (e) {
        console.warn(`[analytics] failed to parse trace.output for ${trace.id}:`, e);
      }
      const toolTrace = outputData.tool_trace;

      // ---- latency (Langfuse: seconds, float) ----
      const latencyInSec =
        (typeof trace.latency === "number" ? trace.latency : null) ??
        (typeof trace.duration === "number" ? trace.duration : 0);
      latenciesSec.push(latencyInSec);
      if (latencyInSec < 2) latencyBuckets["<2s"]++;
      else if (latencyInSec < 5) latencyBuckets["2-5s"]++;
      else if (latencyInSec < 10) latencyBuckets["5-10s"]++;
      else if (latencyInSec < 20) latencyBuckets["10-20s"]++;
      else if (latencyInSec < 30) latencyBuckets["20-30s"]++;
      else latencyBuckets[">30s"]++;

      // ---- intent & language ----
      const intent = inputData.intent || "other";
      intentCounts[intent] = (intentCounts[intent] || 0) + 1;
      const langRaw = inputData.user_language;
      const lang = typeof langRaw === "string" ? langRaw.trim() : "";
      if (lang && lang.toLowerCase() !== "unknown") {
        languageCounts[lang] = (languageCounts[lang] || 0) + 1;
      }

      // ---- confidence ----
      const conf = typeof inputData.intent_confidence === "number" ? inputData.intent_confidence : 0;
      totalConfidence += conf;
      if (conf > 0 && conf < 0.6) lowConfidence++;

      // ---- SKUs in answer (for KPI: SKU hit rate) ----
      const skuCounts = outputData.sku_counts_in_answer || {};
      const skuProductNamesRaw = outputData.sku_product_names || {};
      const skuProductNames: Record<string, string> = {};
      if (skuProductNamesRaw && typeof skuProductNamesRaw === "object") {
        for (const [k, v] of Object.entries(skuProductNamesRaw as Record<string, unknown>)) {
          const skuKey = String(k || "").trim();
          const name = typeof v === "string" ? v.trim() : String(v || "").trim();
          if (skuKey && name) skuProductNames[skuKey] = name;
        }
      }
      if (Object.keys(skuCounts).length > 0) skuHits++;

      // ---- SKUs retrieved (for "Top SKUs retrieved" table) ----
      let retrievedSkus: string[] = [];
      if (Array.isArray(outputData.retrieved_skus)) {
        retrievedSkus = (outputData.retrieved_skus as unknown[])
          .map((v) => String(v || "").trim())
          .filter(Boolean);
      }
      // Fallback for older tool-loop traces without retrieved_skus:
      // infer from tool_trace get_product calls (sku arguments).
      if (retrievedSkus.length === 0 && Array.isArray(toolTrace)) {
        for (const step of toolTrace) {
          if (!step || typeof step !== "object") continue;
          const stepName = String((step as any).name || "").trim();
          if (stepName !== "get_product") continue;
          const args = (step as any).arguments;
          const skuKey = args && typeof args === "object" ? String(args.sku || "").trim() : "";
          if (skuKey) retrievedSkus.push(skuKey);
        }
      }

      if (retrievedSkus.length > 0) {
        for (const sku of retrievedSkus) {
          const skuKey = String(sku || "").trim();
          if (!skuKey) continue;
          const name = skuProductNames[skuKey] || skuProductNames[skuKey.toUpperCase()] || skuKey;
          if (!topSKUsMap[skuKey]) topSKUsMap[skuKey] = { product: name, hits: 0 };
          topSKUsMap[skuKey].hits += 1;
        }
      } else {
        // Final fallback: use cited SKUs so the table isn't empty.
        Object.entries(skuCounts).forEach(([sku, count]: [string, any]) => {
          const skuKey = String(sku || "").trim();
          if (!skuKey) return;
          const name = skuProductNames[skuKey] || skuProductNames[skuKey.toUpperCase()] || skuKey;
          if (!topSKUsMap[skuKey]) topSKUsMap[skuKey] = { product: name, hits: 0 };
          topSKUsMap[skuKey].hits += typeof count === "number" ? count : 1;
        });
      }

      // ---- retrieved_brands / categories / subcategories ----
      const brandsMap = outputData.retrieved_brands || {};
      if (brandsMap && typeof brandsMap === "object") {
        for (const [b, n] of Object.entries(brandsMap as Record<string, unknown>)) {
          const key = String(b || "").trim();
          if (!key) continue;
          brandCounts[key] = (brandCounts[key] || 0) + (Number(n) || 0);
        }
      }
      const catsMap = outputData.retrieved_categories || {};
      if (catsMap && typeof catsMap === "object") {
        for (const [c, n] of Object.entries(catsMap as Record<string, unknown>)) {
          const key = String(c || "").trim();
          if (!key) continue;
          categoryCounts[key] = (categoryCounts[key] || 0) + (Number(n) || 0);
        }
      }
      const subcatsMap = outputData.retrieved_subcategories || {};
      if (subcatsMap && typeof subcatsMap === "object") {
        for (const [sc, n] of Object.entries(subcatsMap as Record<string, unknown>)) {
          const key = String(sc || "").trim();
          if (!key) continue;
          subcategoryCounts[key] = (subcategoryCounts[key] || 0) + (Number(n) || 0);
        }
      }

      // ---- tool_trace (only present on tools path) ----
      let usedCompareTool = false;
      let usedBasketTool = false;
      let budgetEur: number | null = null;
      if (Array.isArray(toolTrace) && toolTrace.length > 0) {
        pathCounts.tools++;
        for (const step of toolTrace) {
          const name = step && typeof step === "object" ? String(step.name || "") : "";
          if (!name) continue;
          toolCounts[name] = (toolCounts[name] || 0) + 1;
          if (name === "compare_products") usedCompareTool = true;
          if (name === "build_budget_basket") {
            usedBasketTool = true;
            const args = step && typeof step === "object" ? (step as any).arguments : null;
            const b = args && typeof args === "object" ? (args as any).budget_eur : null;
            if (typeof b === "number" && Number.isFinite(b) && b > 0) budgetEur = b;
          }
        }
      } else {
        pathCounts.stream++;
      }

      // ---- retrieval health flags ----
      if (outputData.no_result === true) zeroResults++;
      if (outputData.fallback_triggered === true) fallbackTriggered++;

      // ---- business KPIs ----
      const purchaseIntents = new Set([
        "product_search",
        "product_recommendation",
        "price_compare",
        "basket_build",
        "barcode_lookup",
      ]);
      if (purchaseIntents.has(intent)) purchaseIntent++;
      if (intent === "price_compare" || usedCompareTool) comparisonQueries++;
      if (intent === "basket_build" || usedBasketTool) basketQueries++;
      if (budgetEur != null) {
        basketBudgetSum += budgetEur;
        basketBudgetCount += 1;
      }

      // ---- hourly buckets ----
      const hour = new Date(trace.timestamp).getUTCHours().toString().padStart(2, "0");
      if (hourlyVolume[hour]) {
        if ((hourlyVolume[hour] as any)[intent] !== undefined) (hourlyVolume[hour] as any)[intent]++;
        else hourlyVolume[hour].other++;
      }
      if (confidenceSequence[hour]) {
        confidenceSequence[hour].confidence += conf;
        confidenceSequence[hour].count++;
      }
    });

    const total = traces.length;

    // ---------------- Latency percentiles (sorted) ----------------
    const sortedLatencies = [...latenciesSec].filter((v) => v >= 0).sort((a, b) => a - b);
    const pctl = (p: number) => {
      if (sortedLatencies.length === 0) return 0;
      const idx = Math.min(sortedLatencies.length - 1, Math.floor(p * sortedLatencies.length));
      return sortedLatencies[idx];
    };
    const p50 = pctl(0.5);
    const p95 = pctl(0.95);

    // ---------------- Confidence sequence ----------------
    const finalConfidence = Object.values(confidenceSequence).map((h) => ({
      hour: h.hour,
      confidence: h.count > 0 ? h.confidence / h.count : null,
    }));

    // ---------------- Sorted top-N helpers ----------------
    const sortTopN = (m: Record<string, number>, n: number) =>
      Object.entries(m)
        .map(([name, value]) => ({ name, value }))
        .sort((a, b) => b.value - a.value)
        .slice(0, n);

    const brandPalette = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#6366f1", "#f97316", "#84cc16"];
    const withColors = (rows: { name: string; value: number }[]) =>
      rows.map((r, i) => ({ ...r, color: brandPalette[i % brandPalette.length] }));

    return NextResponse.json({
      kpis: {
        total_queries: total,
        purchase_intent_rate: total > 0 ? Math.round((purchaseIntent / total) * 100) : 0,
        basket_queries: basketQueries,
        avg_basket_budget_eur: basketBudgetCount > 0 ? Number((basketBudgetSum / basketBudgetCount).toFixed(2)) : 0,
        comparison_queries: comparisonQueries,
        sku_hit_rate: total > 0 ? Math.round((skuHits / total) * 100) : 0,
        top_brand: sortTopN(brandCounts, 1)[0]?.name || null,
        top_category: sortTopN(categoryCounts, 1)[0]?.name || null,

        // Keep technical metrics for deeper debugging views/charts.
        p50_latency_sec: Number(p50.toFixed(2)),
        p95_latency_sec: Number(p95.toFixed(2)),
        zero_result_rate: total > 0 ? Math.round((zeroResults / total) * 100) : 0,
        fallback_rate: total > 0 ? Math.round((fallbackTriggered / total) * 100) : 0,
        avg_confidence: total > 0 ? Number((totalConfidence / total).toFixed(2)) : 0,
        low_confidence_rate: total > 0 ? Math.round((lowConfidence / total) * 100) : 0,
      },
      charts: {
        volume: Object.values(hourlyVolume),
        latency: Object.entries(latencyBuckets).map(([bucket, count]) => ({
          bucket,
          count,
          color:
            bucket.includes("<") || bucket.includes("2-5s")
              ? "#10b981"
              : bucket.includes("5-10s")
              ? "#f59e0b"
              : "#ef4444",
        })),
        confidence: finalConfidence,
        intent: Object.entries(intentCounts).map(([name, value]) => ({
          name,
          value,
          color:
            name === "product_search"
              ? "#3b82f6"
              : name === "barcode_lookup"
              ? "#10b981"
              : name === "product_recommendation"
              ? "#06b6d4"
              : name === "price_compare"
              ? "#f97316"
              : name === "basket_build"
              ? "#ec4899"
              : name === "order_status"
              ? "#f59e0b"
              : name === "general_qa"
              ? "#a855f7"
              : "#94a3b8",
        })),
        languages: Object.entries(languageCounts).map(([name, value]) => ({ name, value })),
        topBrands: withColors(sortTopN(brandCounts, 10)),
        topCategories: withColors(sortTopN(categoryCounts, 10)),
        topSubcategories: withColors(sortTopN(subcategoryCounts, 10)),
        toolUsage: withColors(sortTopN(toolCounts, 10)),
        pathDistribution: withColors(
          Object.entries(pathCounts)
            .filter(([, v]) => v > 0)
            .map(([name, value]) => ({ name, value }))
        ),
      },
      topSKUs: Object.entries(topSKUsMap)
        .map(([sku, v]) => ({
          sku,
          product: v.product,
          hits: v.hits,
          frequency: Math.round((v.hits / total) * 100),
        }))
        .sort((a, b) => b.hits - a.hits)
        .slice(0, 5),
      recentQueries: traces.map((t: any) => {
        let input: any = {};
        let output: any = {};
        try {
          input = typeof t.input === "string" ? JSON.parse(t.input) : t.input || {};
        } catch (e) {
          console.warn(`[analytics] recentQueries: failed to parse input for ${t.id}:`, e);
        }
        try {
          output = typeof t.output === "string" ? JSON.parse(t.output) : t.output || {};
        } catch (e) {
          console.warn(`[analytics] recentQueries: failed to parse output for ${t.id}:`, e);
        }
        const latencyInSec =
          (typeof t.latency === "number" ? t.latency : null) ??
          (typeof t.duration === "number" ? t.duration : 0);
        const hasToolTrace = Array.isArray(output.tool_trace) && output.tool_trace.length > 0;
        return {
          id: t.id,
          timestamp: t.timestamp,
          query: input.query || "Unknown Query",
          intent: input.intent || "other",
          intent_confidence: input.intent_confidence || 0,
          latency: latencyInSec,
          latency_known: typeof t.latency === "number" && t.latency > 0,
          answer: output.answer || "No response recorded",
          path: hasToolTrace ? "tools" : "stream",
          tools_used: hasToolTrace ? output.tool_trace.map((s: any) => s?.name).filter(Boolean) : [],
          status: output.no_result === true ? "zero_result" : latencyInSec > 15 ? "critical" : "success",
          rawInput: input,
          rawOutput: output,
        };
      }),
    });
  } catch (error: any) {
    console.error("Langfuse Analytics Error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
