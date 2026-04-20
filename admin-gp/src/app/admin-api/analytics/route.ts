import { NextResponse } from "next/server";
import { langfuse } from "@/lib/langfuse";

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

    // Initialize sequential 24-hour volume and confidence arrays
    const hourlyVolume: Record<string, any> = {};
    const confidenceSequence: Record<string, { hour: string; confidence: number; count: number }> = {};
    
    for (let i = 0; i < 24; i++) {
      const h = i.toString().padStart(2, '0');
      hourlyVolume[h] = { hour: h, product_search: 0, barcode_lookup: 0, price_check: 0, other: 0 };
      confidenceSequence[h] = { hour: h, confidence: 0, count: 0 };
    }

    let totalLatency = 0;
    let totalConfidence = 0;
    let skuHits = 0;
    let zeroResults = 0;
    const intentCounts: Record<string, number> = {};
    const languageCounts: Record<string, number> = {};
    const latencyBuckets = { "<2s": 0, "2-5s": 0, "5-10s": 0, "10-20s": 0, "20-30s": 0, ">30s": 0 };
    const topSKUsMap: Record<string, { product: string; hits: number }> = {};

    traces.forEach((trace: any) => {
      let inputData: any = {};
      let outputData: any = {};
      try { inputData = typeof trace.input === 'string' ? JSON.parse(trace.input) : trace.input || {}; } catch(e) {}
      try { outputData = typeof trace.output === 'string' ? JSON.parse(trace.output) : trace.output || {}; } catch(e) {}

      // Latency
      const latencyInSec = trace.duration || 0;
      totalLatency += latencyInSec;
      if (latencyInSec < 2) latencyBuckets["<2s"]++;
      else if (latencyInSec < 5) latencyBuckets["2-5s"]++;
      else if (latencyInSec < 10) latencyBuckets["5-10s"]++;
      else if (latencyInSec < 20) latencyBuckets["10-20s"]++;
      else if (latencyInSec < 30) latencyBuckets["20-30s"]++;
      else latencyBuckets[">30s"]++;

      // Intent & Language
      const intent = inputData.intent || "other";
      intentCounts[intent] = (intentCounts[intent] || 0) + 1;
      const langRaw = inputData.user_language;
      const lang = typeof langRaw === "string" ? langRaw.trim() : "";
      if (lang && lang.toLowerCase() !== "unknown") {
        languageCounts[lang] = (languageCounts[lang] || 0) + 1;
      }

      // Confidence
      const conf = inputData.intent_confidence || 0;
      totalConfidence += conf;

      // SKUs
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
      if (Object.keys(skuCounts).length > 0) skuHits++; else zeroResults++;
      Object.entries(skuCounts).forEach(([sku, count]: [string, any]) => {
        const skuKey = String(sku || "").trim();
        if (!skuKey) return;
        const name = skuProductNames[skuKey] || skuProductNames[skuKey.toUpperCase()] || skuKey;
        if (!topSKUsMap[skuKey]) topSKUsMap[skuKey] = { product: name, hits: 0 };
        topSKUsMap[skuKey].hits += (typeof count === 'number' ? count : 1);
      });

      // Sequential Data Mapping
      const hour = new Date(trace.timestamp).getUTCHours().toString().padStart(2, '0');
      if (hourlyVolume[hour]) {
        if (hourlyVolume[hour][intent] !== undefined) hourlyVolume[hour][intent]++;
        else hourlyVolume[hour].other++;
      }
      if (confidenceSequence[hour]) {
        confidenceSequence[hour].confidence += conf;
        confidenceSequence[hour].count++;
      }
    });

    const total = traces.length;

    // Finalize confidence sequence
    const finalConfidence = Object.values(confidenceSequence).map(h => ({
      hour: h.hour,
      confidence: h.count > 0 ? h.confidence / h.count : null
    }));

    return NextResponse.json({
      charts: {
        volume: Object.values(hourlyVolume),
        latency: Object.entries(latencyBuckets).map(([bucket, count]) => ({ bucket, count, color: bucket.includes('<') || bucket.includes('2-5s') ? '#10b981' : bucket.includes('5-10s') ? '#f59e0b' : '#ef4444' })),
        confidence: finalConfidence,
        intent: Object.entries(intentCounts).map(([name, value]) => ({ name, value, color: name === 'product_search' ? '#3b82f6' : name === 'barcode_lookup' ? '#10b981' : '#f59e0b' })),
        languages: Object.entries(languageCounts).map(([name, value]) => ({ name, value }))
      },
      topSKUs: Object.entries(topSKUsMap)
        .map(([sku, v]) => ({ sku, product: v.product, hits: v.hits, frequency: Math.round((v.hits / total) * 100) }))
        .sort((a, b) => b.hits - a.hits)
        .slice(0, 5),
      recentQueries: traces.map((t: any) => {
        let input: any = {};
        let output: any = {};
        try { input = typeof t.input === 'string' ? JSON.parse(t.input) : t.input || {}; } catch(e) {}
        try { output = typeof t.output === 'string' ? JSON.parse(t.output) : t.output || {}; } catch(e) {}
        return {
          id: t.id,
          timestamp: t.timestamp,
          query: input.query || "Unknown Query",
          intent: input.intent || "other",
          intent_confidence: input.intent_confidence || 0,
          latency: t.duration || 0,
          answer: output.answer || "No response recorded",
          status: (t.duration || 0) > 15 ? 'critical' : 'success',
          rawInput: input,
          rawOutput: output
        };
      })
    });

  } catch (error: any) {
    console.error("Langfuse Analytics Error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
