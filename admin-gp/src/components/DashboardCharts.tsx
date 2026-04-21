"use client";

import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  AreaChart, Area, PieChart, Pie, Cell
} from 'recharts';

interface ChartProps {
  data: any[];
  title: string;
}

export const QueryVolumeChart: React.FC<ChartProps> = ({ data, title }) => (
  <div className="card" style={{ height: '400px', display: 'flex', flexDirection: 'column' }}>
    <h3 className="heading-2 mb-4" style={{ marginBottom: '1rem' }}>{title}</h3>
    <div style={{ flex: 1, minHeight: 0 }}>
      <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
          <XAxis 
            dataKey="hour" 
            axisLine={false} 
            tickLine={false} 
            tick={{ fontSize: 10, fill: '#64748b' }}
            interval={2}
          />
          <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#64748b' }} />
          <Tooltip
            contentStyle={{ background: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
          />
          <Legend iconType="circle" wrapperStyle={{ paddingTop: '10px', fontSize: '11px' }} />
          <Bar dataKey="product_search" stackId="a" fill="#3b82f6" radius={[0, 0, 0, 0]} />
          <Bar dataKey="barcode_lookup" stackId="a" fill="#10b981" radius={[0, 0, 0, 0]} />
          <Bar dataKey="price_check" stackId="a" fill="#f59e0b" radius={[0, 0, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  </div>
);

export const LatencyDistributionChart: React.FC<ChartProps> = ({ data, title }) => (
  <div className="card" style={{ height: '400px', display: 'flex', flexDirection: 'column' }}>
    <h3 className="heading-2 mb-4" style={{ marginBottom: '1rem' }}>{title}</h3>
    <div style={{ flex: 1, minHeight: 0 }}>
      <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
        <BarChart data={data} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
          <XAxis type="number" hide />
          <YAxis dataKey="bucket" type="category" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#64748b' }} width={60} />
          <Tooltip cursor={{ fill: 'transparent' }} />
          <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={30}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  </div>
);

export const ConfidenceLineChart: React.FC<ChartProps> = ({ data, title }) => (
  <div className="card" style={{ height: '300px', display: 'flex', flexDirection: 'column' }}>
    <h3 className="heading-2 mb-4" style={{ marginBottom: '1rem' }}>{title}</h3>
    <div style={{ flex: 1, minHeight: 0 }}>
      <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorConf" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
          <XAxis 
            dataKey="hour" 
            axisLine={false} 
            tickLine={false} 
            tick={{ fontSize: 10, fill: '#64748b' }}
            interval={2}
          />
          <YAxis domain={[0.7, 1]} axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#64748b' }} />
          <Tooltip />
          <Area 
            type="monotone" 
            dataKey="confidence" 
            stroke="#3b82f6" 
            strokeWidth={3} 
            fillOpacity={1} 
            fill="url(#colorConf)" 
            connectNulls={true}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  </div>
);

export const IntentDonutChart: React.FC<ChartProps> = ({ data, title }) => (
  <div className="card" style={{ height: '300px', display: 'flex', flexDirection: 'column' }}>
    <h3 className="heading-2 mb-4" style={{ marginBottom: '1rem' }}>{title}</h3>
    <div style={{ flex: 1, minHeight: 0 }}>
      <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
        <PieChart>
          <Pie
            data={data.filter((entry) => String(entry?.name ?? '').toLowerCase() !== 'other')}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={80}
            paddingAngle={5}
            dataKey="value"
          >
            {data.filter((entry) => String(entry?.name ?? '').toLowerCase() !== 'other').map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip />
          <Legend verticalAlign="middle" align="right" layout="vertical" iconType="circle" wrapperStyle={{ fontSize: '11px' }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  </div>
);

export const LanguageDistributionChart: React.FC<ChartProps> = ({ data, title }) => (
  <div className="card" style={{ height: '300px', display: 'flex', flexDirection: 'column' }}>
    <h3 className="heading-2 mb-4" style={{ marginBottom: '1rem' }}>{title}</h3>
    <div style={{ flex: 1, minHeight: 0 }}>
      <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
        <BarChart data={data} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
          <XAxis type="number" hide />
          <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#64748b' }} width={80} />
          <Tooltip cursor={{ fill: 'transparent' }} />
          <Bar dataKey="value" fill="var(--primary)" radius={[0, 4, 4, 0]} barSize={20} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  </div>
);

/**
 * Horizontal bar chart tuned for top-N categorical aggregations
 * (retrieved brands, subcategories, tool-usage counts, etc.).
 *
 * Expected data shape: [{ name, value, color? }].
 */
const HORIZ_PALETTE = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#6366f1', '#f97316', '#84cc16'];

export const TopBrandsChart: React.FC<ChartProps> = ({ data, title }) => (
  <div className="card" style={{ height: '360px', display: 'flex', flexDirection: 'column' }}>
    <h3 className="heading-2 mb-4" style={{ marginBottom: '1rem' }}>{title}</h3>
    <div style={{ flex: 1, minHeight: 0 }}>
      <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 16, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
          <XAxis type="number" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#64748b' }} />
          <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#64748b' }} width={140} />
          <Tooltip cursor={{ fill: 'transparent' }} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={18}>
            {data.map((entry, i) => (
              <Cell key={`cell-${i}`} fill={entry?.color || HORIZ_PALETTE[i % HORIZ_PALETTE.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  </div>
);

export const TopSubcategoriesChart: React.FC<ChartProps> = ({ data, title }) => (
  <div className="card" style={{ height: '360px', display: 'flex', flexDirection: 'column' }}>
    <h3 className="heading-2 mb-4" style={{ marginBottom: '1rem' }}>{title}</h3>
    <div style={{ flex: 1, minHeight: 0 }}>
      <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 16, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
          <XAxis type="number" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#64748b' }} />
          <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#64748b' }} width={160} />
          <Tooltip cursor={{ fill: 'transparent' }} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={16}>
            {data.map((entry, i) => (
              <Cell key={`cell-${i}`} fill={entry?.color || HORIZ_PALETTE[i % HORIZ_PALETTE.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  </div>
);

/**
 * Donut renderer reused for {category / tool / path} distributions.
 * Filters zero-value entries and uses the `color` field if provided by the API.
 */
const DonutChart: React.FC<ChartProps & { innerRadius?: number; outerRadius?: number }> = ({
  data,
  title,
  innerRadius = 55,
  outerRadius = 80,
}) => {
  const safe = (data || []).filter((d) => Number(d?.value) > 0);
  return (
    <div className="card" style={{ height: '320px', display: 'flex', flexDirection: 'column' }}>
      <h3 className="heading-2 mb-4" style={{ marginBottom: '1rem' }}>{title}</h3>
      <div style={{ flex: 1, minHeight: 0 }}>
        <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
          <PieChart>
            <Pie data={safe} cx="50%" cy="50%" innerRadius={innerRadius} outerRadius={outerRadius} paddingAngle={4} dataKey="value">
              {safe.map((entry, i) => (
                <Cell key={`cell-${i}`} fill={entry?.color || HORIZ_PALETTE[i % HORIZ_PALETTE.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend verticalAlign="middle" align="right" layout="vertical" iconType="circle" wrapperStyle={{ fontSize: '11px' }} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export const TopCategoriesChart: React.FC<ChartProps> = (props) => <DonutChart {...props} />;
export const ToolUsageChart: React.FC<ChartProps> = (props) => <DonutChart {...props} />;
export const PathDistributionChart: React.FC<ChartProps> = (props) => <DonutChart {...props} innerRadius={50} outerRadius={75} />;
