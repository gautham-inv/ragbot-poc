import React, { useState } from 'react';
import { QueryTrace } from '@/lib/mockData';
import { ChevronLeft, ChevronRight, X, Clock, Database, MessageSquare, User } from 'lucide-react';

interface SKU {
  sku: string;
  product: string;
  hits: number;
  frequency: number;
}

export const TopSKUsTable: React.FC<{ data: SKU[] }> = ({ data }) => (
  <div className="card h-full flex flex-direction-column" style={{ display: 'flex', flexDirection: 'column' }}>
    <h3 className="heading-2 mb-4" style={{ marginBottom: '1.5rem' }}>Top SKUs retrieved</h3>
    <div style={{ flex: 1, overflow: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
        <thead>
          <tr className="text-xs text-secondary text-left uppercase font-bold" style={{ borderBottom: '1px solid var(--border)' }}>
            <th style={{ padding: '0.75rem', width: '30%' }}>SKU</th>
            <th style={{ padding: '0.75rem', width: '40%' }}>Product</th>
            <th style={{ padding: '0.75rem', width: '15%' }}>Hits</th>
            <th style={{ padding: '0.75rem', width: '15%' }}>Freq</th>
          </tr>
        </thead>
        <tbody>
          {data.map((item, idx) => (
            <tr key={idx} style={{ borderBottom: '1px solid var(--border)', fontSize: '0.875rem' }}>
              <td style={{ padding: '0.75rem', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.sku}</td>
              <td style={{ padding: '0.75rem', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.product}</td>
              <td style={{ padding: '0.75rem' }}>{item.hits}</td>
              <td style={{ padding: '0.75rem' }}>
                <div style={{ width: '100%', height: '4px', background: 'var(--border)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ width: `${item.frequency}%`, height: '100%', background: 'var(--primary)' }}></div>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

const QueryDetailModal: React.FC<{ query: QueryTrace; onClose: () => void }> = ({ query, onClose }) => {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div className="flex items-center gap-2">
            <span className={`badge badge-info uppercase text-xs`}>
              {query.intent.replace('_', ' ')}
            </span>
            <h3 className="heading-2" style={{ margin: 0 }}>Query Inspector</h3>
          </div>
          <button onClick={onClose} className="bg-transparent border-none cursor-pointer p-1 text-secondary hover:text-primary">
            <X size={20} />
          </button>
        </div>
        
        <div className="modal-body">
          <div className="grid gap-6" style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1.5rem' }}>
            {/* Conversation Flow */}
            <div className="grid gap-4">
              <div className="flex flex-direction-column gap-2">
                <div className="flex items-center gap-2 text-xs text-secondary font-bold uppercase">
                  <User size={14} /> User Query
                </div>
                <div className="p-4 border" style={{ borderRadius: '8px', borderLeft: '4px solid var(--primary)', fontSize: '0.925rem', lineHeight: 1.5, background: 'var(--surface)', borderColor: 'var(--border)' }}>
                  {query.query}
                </div>
              </div>

              <div className="flex flex-direction-column gap-2">
                <div className="flex items-center gap-2 text-xs text-secondary font-bold uppercase">
                  <MessageSquare size={14} /> Bot Response
                </div>
                <div className="p-4 border" style={{ borderRadius: '8px', borderLeft: '4px solid var(--success)', fontSize: '0.925rem', lineHeight: 1.6, background: 'var(--surface)', borderColor: 'var(--border)' }}>
                  {query.answer || "No response recorded for this trace."}
                </div>
              </div>
            </div>

            {/* Metadata Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
              <div className="p-3 border" style={{ borderRadius: '8px', background: 'var(--surface)', borderColor: 'var(--border)' }}>
                <div className="text-secondary text-xs uppercase font-bold mb-1 flex items-center gap-1">
                  <Clock size={12} /> Latency
                </div>
                <div className="font-bold">{query.latency.toFixed(2)}s</div>
              </div>
              <div className="p-3 border" style={{ borderRadius: '8px', background: 'var(--surface)', borderColor: 'var(--border)' }}>
                <div className="text-secondary text-xs uppercase font-bold mb-1 flex items-center gap-1">
                  <Database size={12} /> Confidence
                </div>
                <div className="font-bold">{(query.intent_confidence || 0).toFixed(4)}</div>
              </div>
              <div className="p-3 border" style={{ borderRadius: '8px', background: 'var(--surface)', borderColor: 'var(--border)' }}>
                <div className="text-secondary text-xs uppercase font-bold mb-1">Timestamp (UTC)</div>
                <div className="font-bold">{new Date(query.timestamp).toLocaleString()}</div>
              </div>
              <div className="p-3 border" style={{ borderRadius: '8px', background: 'var(--surface)', borderColor: 'var(--border)' }}>
                <div className="text-secondary text-xs uppercase font-bold mb-1">Trace ID</div>
                <div className="font-bold text-xs" style={{ fontFamily: 'monospace' }}>{query.id.substring(0, 8)}...</div>
              </div>
            </div>

            {/* Raw Components */}
            <div className="grid gap-2">
              <div className="text-xs text-secondary font-bold uppercase">Raw Trace Metadata</div>
              <pre className="json-block">
                {JSON.stringify({ input: query.rawInput, output: query.rawOutput }, null, 2)}
              </pre>
            </div>
          </div>
        </div>

        <div className="modal-footer">
          <button 
            onClick={() => {
              navigator.clipboard.writeText(JSON.stringify({ input: query.rawInput, output: query.rawOutput }, null, 2));
              alert("Metadata copied to clipboard!");
            }}
            className="px-4 py-2 rounded-md cursor-pointer font-medium hover:opacity-90 mr-2 bg-[var(--surface-hover)] text-[var(--foreground)] border border-[var(--border)]"
            style={{ fontSize: '0.875rem' }}
          >
            Copy JSON
          </button>
          <button 
            onClick={onClose}
            className="px-4 py-2 rounded-md cursor-pointer font-medium hover:opacity-90 bg-[var(--primary)] text-white border border-transparent"
            style={{ fontSize: '0.875rem' }}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export const RecentQueriesList: React.FC<{ data: QueryTrace[] }> = ({ data }) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedQuery, setSelectedQuery] = useState<QueryTrace | null>(null);
  const pageSize = 10;
  
  const totalPages = Math.ceil(data.length / pageSize);
  const startIndex = (currentPage - 1) * pageSize;
  const currentData = data.slice(startIndex, startIndex + pageSize);

  return (
    <>
      <div className="card h-full flex flex-direction-column" style={{ display: 'flex', flexDirection: 'column', minHeight: '520px' }}>
        <div className="flex justify-between items-center mb-4" style={{ marginBottom: '1.5rem' }}>
          <h3 className="heading-2">Recent queries</h3>
          <span className="text-xs text-secondary font-medium">Page {currentPage} of {totalPages || 1}</span>
        </div>
        
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.5rem', overflow: 'hidden' }}>
          {currentData.length > 0 ? (
            currentData.map((trace) => (
              <div 
                key={trace.id} 
                className="flex items-center justify-between p-2 hover:bg-surface-hover transition-colors cursor-pointer group" 
                style={{ borderBottom: '1px solid var(--border)', padding: '0.75rem 0.5rem', borderRadius: '6px' }}
                onClick={() => setSelectedQuery(trace)}
              >
                <div className="flex items-center gap-4" style={{ flex: 1, minWidth: 0 }}>
                  <span className={`badge badge-info uppercase text-xs`} style={{ minWidth: '95px', textAlign: 'center', flexShrink: 0 }}>
                    {trace.intent.replace('_', ' ')}
                  </span>
                  <span className="text-sm font-medium group-hover:text-primary transition-colors" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                    {trace.query}
                  </span>
                </div>
                <div className="flex items-center gap-4 ml-4" style={{ flexShrink: 0 }}>
                  <span className="text-xs text-secondary font-bold" title="Confidence Score">{(trace.intent_confidence || 0).toFixed(2)}</span>
                  <span className={`text-xs font-bold ${trace.latency > 15 ? 'text-critical' : 'text-secondary'}`}>
                    {trace.latency.toFixed(1)}s
                  </span>
                </div>
              </div>
            ))
          ) : (
            <div className="flex items-center justify-center p-8 text-secondary text-sm italic">No records found</div>
          )}
        </div>

        {/* Pagination Controls */}
        <div className="flex justify-between items-center mt-6 pt-4" style={{ borderTop: '1px solid var(--border)', marginTop: 'auto' }}>
          <button 
            onClick={(e) => { e.stopPropagation(); setCurrentPage(p => Math.max(1, p - 1)); }}
            disabled={currentPage === 1}
            className="btn-pagination btn-pagination-nav"
          >
            <ChevronLeft size={16} /> Previous
          </button>
          <div className="flex gap-2">
            {totalPages > 1 && Array.from({ length: Math.min(5, totalPages) }).map((_, i) => {
               const pageNum = i + 1;
               return (
                 <button 
                   key={pageNum}
                   onClick={(e) => { e.stopPropagation(); setCurrentPage(pageNum); }}
                   className={`btn-pagination ${currentPage === pageNum ? 'active' : ''}`}
                 >
                   {pageNum}
                 </button>
               );
            })}
          </div>
          <button 
            onClick={(e) => { e.stopPropagation(); setCurrentPage(p => Math.min(totalPages, p + 1)); }}
            disabled={currentPage === totalPages || totalPages === 0}
            className="btn-pagination btn-pagination-nav"
          >
            Next <ChevronRight size={16} />
          </button>
        </div>
      </div>

      {selectedQuery && (
        <QueryDetailModal 
          query={selectedQuery} 
          onClose={() => setSelectedQuery(null)} 
        />
      )}
    </>
  );
};
