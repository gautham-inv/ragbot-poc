import React from 'react';

interface StatCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  trend?: {
    value: string;
    isUp: boolean;
  };
  color?: string;
}

export const StatCard: React.FC<StatCardProps> = ({ label, value, subValue, trend, color }) => {
  return (
    <div className="card flex flex-direction-column justify-between items-start" style={{ 
      borderLeft: `4px solid ${color || 'var(--primary)'}`, 
      padding: '1rem 1.25rem',
      minHeight: '110px',
      gap: '0.25rem'
    }}>
      {/* Label section */}
      <span className="text-secondary uppercase font-bold tracking-wider" style={{ 
        fontSize: '0.65rem', 
        lineHeight: '1.4',
        opacity: 0.8,
        display: 'block',
        width: '100%',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        display: '-webkit-box',
        WebkitLineClamp: 2,
        WebkitBoxOrient: 'vertical'
      }}>
        {label}
      </span>
      
      {/* Value section */}
      <div className="flex items-baseline gap-2" style={{ margin: '0.25rem 0' }}>
        <span className="heading-1" style={{ fontSize: '1.65rem', lineHeight: '1.1' }}>
          {value}
        </span>
      </div>

      {/* Footer section: Trend & Subvalue */}
      <div className="flex items-center justify-between w-full mt-auto" style={{ gap: '0.5rem' }}>
        {trend && (
          <span className={`text-xs font-bold ${trend.isUp ? 'badge-success' : 'badge-critical'} badge`} style={{ 
            padding: '0.125rem 0.5rem', 
            borderRadius: '100px', 
            fontSize: '0.7rem' 
          }}>
            {trend.isUp ? '↑' : '↓'} {trend.value}
          </span>
        )}
        {subValue && (
          <span className="text-secondary text-right" style={{ 
            fontSize: '0.65rem', 
            fontStyle: 'italic',
            flex: 1,
            minWidth: 0,
            textAlign: 'right',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflowWrap: 'anywhere'
          }}>
            {subValue}
          </span>
        )}
      </div>
    </div>
  );
};
