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
          <Bar dataKey="other" stackId="a" fill="#94a3b8" radius={[2, 2, 0, 0]} />
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
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={80}
            paddingAngle={5}
            dataKey="value"
          >
            {data.map((entry, index) => (
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
