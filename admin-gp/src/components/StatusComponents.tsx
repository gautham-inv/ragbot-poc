import React from 'react';

interface HealthItem {
  field: string;
  populated: boolean;
}

interface Alert {
  level: string;
  title: string;
  description: string;
}

export const InstrumentationHealth: React.FC<{ data: HealthItem[] }> = ({ data }) => (
  <div className="card">
    <h3 className="heading-2 mb-4" style={{ marginBottom: '1.5rem' }}>Instrumentation health</h3>
    <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '1rem' }}>
      {data.map((item, idx) => (
        <div key={idx} className="flex items-center gap-2 p-2 bg-surface-hover" style={{ padding: '0.75rem', borderRadius: '8px', background: 'var(--background)', border: '1px solid var(--border)' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: item.populated ? 'var(--success)' : 'var(--critical)' }}></div>
          <span className="text-xs font-medium">{item.field}</span>
          <span className="text-xs text-secondary ml-auto" style={{ marginLeft: 'auto' }}>
            {item.populated ? 'Populated' : 'Missing'}
          </span>
        </div>
      ))}
    </div>
  </div>
);

export const AlertsPanel: React.FC<{ data: Alert[] }> = ({ data }) => (
  <div className="card">
    <h3 className="heading-2 mb-4" style={{ marginBottom: '1.5rem' }}>Alerts & Issues</h3>
    <div className="flex flex-direction-column gap-3" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {data.map((alert, idx) => (
        <div key={idx} className={`p-4 flex gap-4`} style={{ padding: '1rem', borderRadius: '8px', background: 'var(--background)', border: '1px solid var(--border)' }}>
          <span className={`badge badge-${alert.level === 'critical' ? 'critical' : alert.level === 'warning' ? 'warning' : 'info'}`} style={{ height: 'fit-content' }}>
            {alert.level}
          </span>
          <div className="flex flex-direction-column gap-1" style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <h4 className="text-sm font-bold">{alert.title}</h4>
            <p className="text-xs text-secondary">{alert.description}</p>
          </div>
        </div>
      ))}
    </div>
  </div>
);
