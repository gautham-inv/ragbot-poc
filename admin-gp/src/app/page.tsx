"use client";

import React, { useState, useEffect } from 'react';
import {
  QueryVolumeChart,
  LatencyDistributionChart,
  IntentDonutChart,
  LanguageDistributionChart,
  ConfidenceLineChart,
  TopBrandsChart,
  TopCategoriesChart,
  TopSubcategoriesChart,
  ToolUsageChart,
  PathDistributionChart
} from '@/components/DashboardCharts';
import { TopSKUsTable, RecentQueriesList } from '@/components/DataTables';
import { StatCard } from '@/components/StatCard';
import {
  queryVolumeData as mockVolume,
  latencyDistributionData as mockLatency,
  intentBreakdownData as mockIntent,
  topSKUs as mockTopSKUs,
  recentQueries as mockRecent,
  languageDistributionData as mockLanguages
} from '@/lib/mockData';
import { signOut } from '@/lib/auth-client';
import { AlertCircle, LogOut, RefreshCcw } from 'lucide-react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';

export default function Dashboard() {
  const router = useRouter();
  const [timeRange, setTimeRange] = useState('24h');
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loggingOut, setLoggingOut] = useState(false);
  const [showSplash, setShowSplash] = useState(true);

  const fetchData = async (range: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/admin-api/analytics?range=${range}`);
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || "Failed to fetch analytics");
      }
      const jsonData = await response.json();
      setData(jsonData);
    } catch (err: any) {
      console.error(err);
      setError(err.message);
      // Fallback to mock data if no live data is available yet
      // This helps the user see the UI even if they haven't run any queries yet
    } finally {
      setLoading(false);
      setShowSplash(false);
    }
  };

  useEffect(() => {
    fetchData(timeRange);
  }, [timeRange]);

  const onLogout = async () => {
    if (loggingOut) return;
    setLoggingOut(true);
    try {
      await signOut();
    } catch (err) {
      console.error(err);
    } finally {
      router.replace("/login");
      router.refresh();
      setLoggingOut(false);
    }
  };

  const charts = data?.charts || {
    volume: mockVolume,
    latency: mockLatency,
    intent: mockIntent,
    languages: mockLanguages
  };
  const topSKUs = data?.topSKUs || mockTopSKUs;
  const recent = data?.recentQueries || mockRecent;

  // KPI strip — only shown when live data is available.
  const kpis = data?.kpis;

  // New chart sections — rendered only when live data is present so that
  // the existing mock-data widgets above don't change behavior.
  const hasConfidence = Array.isArray(charts?.confidence) && charts.confidence.some((pt: any) => pt?.confidence != null);
  const hasBrands = Array.isArray(charts?.topBrands) && charts.topBrands.length > 0;
  const hasCategories = Array.isArray(charts?.topCategories) && charts.topCategories.length > 0;
  const hasSubcategories = Array.isArray(charts?.topSubcategories) && charts.topSubcategories.length > 0;
  const hasTools = Array.isArray(charts?.toolUsage) && charts.toolUsage.length > 0;
  const hasPaths = Array.isArray(charts?.pathDistribution) && charts.pathDistribution.length > 0;

  return (
    <main className="dashboard-container">
      {showSplash ? (
        <div className="splash-overlay" role="status" aria-live="polite" aria-label="Loading analytics">
          <div className="splash-card">
            <div className="spinner" aria-hidden="true" />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="heading-2">Loading analytics</div>
              <div className="text-sm text-secondary" style={{ marginTop: '0.25rem' }}>
                Fetching recent traces and aggregations.
              </div>
            </div>
          </div>
        </div>
      ) : null}
      {/* Header */}
      <header className="flex flex-wrap justify-between items-center gap-4 mb-8" style={{ marginBottom: '2rem' }}>
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Image src="/paw.jpg" alt="Gloria Pets" width={26} height={26} priority />
            <h1 className="heading-1" style={{ minWidth: 0 }}>GloriaPets - Catalog Bot Analytics</h1>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <div className={`flex items-center gap-1 ${loading ? 'text-secondary' : 'text-success'}`}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: loading ? 'var(--secondary)' : 'currentColor' }}></div>
              <span>{loading ? 'Fetching live traces...' : 'Langfuse traces (Live)'}</span>
            </div>
            <span className="text-secondary">|</span>
            <span className="text-secondary">default environment</span>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          {error && (
            <div className="flex items-center gap-1 text-xs text-critical bg-critical" style={{ background: 'rgba(239, 68, 68, 0.1)', padding: '0.25rem 0.75rem', borderRadius: '4px' }}>
              <AlertCircle size={14} />
              <span>{error} (Showing Mock Data)</span>
            </div>
          )}
          <button
            onClick={() => fetchData(timeRange)}
            className="btn-base btn-secondary-premium"
          >
            <RefreshCcw size={14} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
          <button
            onClick={onLogout}
            disabled={loggingOut}
            className="btn-base btn-secondary-premium"
            title="Log out"
          >
            <LogOut size={14} />
            {loggingOut ? "Logging out..." : "Logout"}
          </button>
          <div className="toggle-group">
            {['24h', '7d', '30d'].map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`btn-toggle ${timeRange === range ? 'active' : ''}`}
              >
                {range}
              </button>
            ))}
          </div>
        </div>
      </header>

      <div style={{ opacity: loading ? 0.6 : 1, transition: 'opacity 0.3s' }}>
        {/* KPI strip — only rendered when we have live data */}
        {kpis && (
          <section
            className="grid mb-8"
            style={{
              marginBottom: '1.5rem',
              gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))',
              gap: '1rem'
            }}
          >
            <StatCard
              label="Total queries"
              value={kpis.total_queries ?? 0}
              subValue={timeRange}
              color="#3b82f6"
            />
            <StatCard
              label="Purchase-intent rate"
              value={`${kpis.purchase_intent_rate ?? 0}%`}
              subValue={`Top brand ${kpis.top_brand || '-'}`}
              color="#06b6d4"
            />
            <StatCard
              label="Basket builds"
              value={kpis.basket_queries ?? 0}
              subValue={`Avg budget €${Number(kpis.avg_basket_budget_eur ?? 0).toFixed(2)}`}
              color="#ec4899"
            />
            <StatCard
              label="Comparison requests"
              value={kpis.comparison_queries ?? 0}
              subValue={`Top category ${kpis.top_category || '-'}`}
              color="#f97316"
            />
            <StatCard
              label="Catalog coverage"
              value={`${Math.max(0, 100 - (kpis.zero_result_rate ?? 0))}%`}
              subValue="Queries with results"
              color="#10b981"
            />
            <StatCard
              label="SKU coverage"
              value={`${kpis.sku_hit_rate ?? 0}%`}
              subValue="Answers that cited at least one SKU"
              color="#8b5cf6"
            />
          </section>
        )}

        {/* ===== TOP ===== Recent queries + top-SKUs */}
        <section className="grid data-grid mb-8" style={{ marginBottom: '1.5rem' }}>
          <TopSKUsTable data={topSKUs} />
          <RecentQueriesList data={recent} />
        </section>

        {/* ===== TOP ===== Most-retrieved brands + category demand */}
        {(hasBrands || hasCategories) && (
          <section className="grid data-grid mb-8" style={{ marginBottom: '1.5rem' }}>
            {hasBrands && <TopBrandsChart data={charts.topBrands} title="Most-retrieved brands" />}
            {hasCategories && <TopCategoriesChart data={charts.topCategories} title="Category demand" />}
          </section>
        )}

        {/* ===== TOP ===== Top subcategories */}
        {hasSubcategories && (
          <section className="grid mb-8" style={{ marginBottom: '1.5rem' }}>
            <TopSubcategoriesChart data={charts.topSubcategories} title="Top subcategories" />
          </section>
        )}

        {/* ===== MIDDLE ===== Query volume + intent breakdown */}
        <section className="grid data-grid mb-8" style={{ marginBottom: '1.5rem' }}>
          <QueryVolumeChart data={charts.volume} title="Query volume by hour" />
          <IntentDonutChart data={charts.intent} title="Intent breakdown" />
        </section>

        {/* ===== MIDDLE ===== Languages + confidence */}
        <section className="grid data-grid mb-8" style={{ marginBottom: '1.5rem' }}>
          <LanguageDistributionChart data={charts.languages} title="Query languages" />
          {hasConfidence && (
            <ConfidenceLineChart data={charts.confidence} title="Intent-classifier confidence by hour" />
          )}
        </section>

        {/* ===== BOTTOM ===== Latency distribution + tool usage */}
        <section className="grid data-grid mb-8" style={{ marginBottom: '1.5rem' }}>
          <LatencyDistributionChart data={charts.latency} title="Latency distribution" />
          {hasTools && <ToolUsageChart data={charts.toolUsage} title="Tool usage (LLM routing)" />}
        </section>

        {/* ===== BOTTOM ===== Path distribution (stream vs tools) */}
        {hasPaths && (
          <section className="grid mb-8" style={{ marginBottom: '1.5rem' }}>
            <PathDistributionChart data={charts.pathDistribution} title="Retrieval path (stream vs tools)" />
          </section>
        )}

      </div>

      <footer className="mt-12 text-center text-xs text-secondary" style={{ marginTop: '3rem', paddingBottom: '2rem' }}>
        Gloria Pets Analytics Dashboard | Powered by Langfuse Traces | Live Connection: {data ? 'Enabled' : 'Simulated'}
      </footer>

      <style jsx global>{`
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </main>
  );
}
