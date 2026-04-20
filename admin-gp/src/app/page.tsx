"use client";

import React, { useState, useEffect } from 'react';
import {
  QueryVolumeChart,
  LatencyDistributionChart,
  IntentDonutChart,
  LanguageDistributionChart
} from '@/components/DashboardCharts';
import { TopSKUsTable, RecentQueriesList } from '@/components/DataTables';
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

  return (
    <main className="dashboard-container">
      {/* Header */}
      <header className="flex justify-between items-center mb-8" style={{ marginBottom: '2rem' }}>
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Image src="/paw.jpg" alt="Gloria Pets" width={26} height={26} priority />
            <h1 className="heading-1">GloriaPets - Catalog Bot Analytics</h1>
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

        <div className="flex items-center gap-4">
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
        {/* Main Charts Row */}
        <section className="grid charts-grid mb-8" style={{ marginBottom: '1.5rem' }}>
          <QueryVolumeChart data={charts.volume} title="Query volume by hour" />
          <LatencyDistributionChart data={charts.latency} title="Latency distribution" />
        </section>

        {/* Secondary Charts Row: Intent & Language */}
        <section className="grid data-grid mb-8" style={{ marginBottom: '1.5rem' }}>
          <IntentDonutChart data={charts.intent} title="Intent breakdown" />
          <LanguageDistributionChart data={charts.languages} title="Query languages" />
        </section>

        {/* Data Row */}
        <section className="grid data-grid mb-8" style={{ marginBottom: '1.5rem' }}>
          <TopSKUsTable data={topSKUs} />
          <RecentQueriesList data={recent} />
        </section>

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
