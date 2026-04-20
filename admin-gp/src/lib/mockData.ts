export interface QueryTrace {
  id: string;
  query: string;
  answer: string;
  intent: 'product_search' | 'barcode_lookup' | 'price_check' | 'other';
  intent_confidence: number;
  latency: number;
  timestamp: string;
  skus_in_answer: Record<string, number>;
  status: 'critical' | 'warning' | 'info' | 'success';
  rawInput?: any;
  rawOutput?: any;
}

export const summaryStats = {
  totalQueries: 247,
  avgLatency:'21.3s',
  skuHitRate: '78%',
  zeroResults: '12%',
  avgConfidence: '0.89',
  truncatedAnswers: '3.2%'
};

export const queryVolumeData = [
  { hour: '00', product_search: 5, barcode_lookup: 2, price_check: 1, other: 1 },
  { hour: '02', product_search: 2, barcode_lookup: 1, price_check: 0, other: 1 },
  { hour: '04', product_search: 1, barcode_lookup: 0, price_check: 0, other: 0 },
  { hour: '06', product_search: 8, barcode_lookup: 3, price_check: 2, other: 1 },
  { hour: '08', product_search: 15, barcode_lookup: 7, price_check: 4, other: 2 },
  { hour: '10', product_search: 25, barcode_lookup: 12, price_check: 8, other: 5 },
  { hour: '12', product_search: 35, barcode_lookup: 15, price_check: 10, other: 6 },
  { hour: '14', product_search: 30, barcode_lookup: 10, price_check: 9, other: 4 },
  { hour: '16', product_search: 32, barcode_lookup: 14, price_check: 11, other: 7 },
  { hour: '18', product_search: 28, barcode_lookup: 11, price_check: 8, other: 5 },
  { hour: '20', product_search: 20, barcode_lookup: 8, price_check: 6, other: 3 },
  { hour: '22', product_search: 12, barcode_lookup: 5, price_check: 3, other: 2 },
];

export const latencyDistributionData = [
  { bucket: '<2s', count: 12, color: '#10b981' },
  { bucket: '2-5s', count: 28, color: '#10b981' },
  { bucket: '5-10s', count: 42, color: '#f59e0b' },
  { bucket: '10-20s', count: 88, color: '#ef4444' },
  { bucket: '20-30s', count: 52, color: '#ef4444' },
  { bucket: '>30s', count: 25, color: '#ef4444' },
];

export const confidenceOverTimeData = [
  { hour: '00', confidence: 0.88 },
  { hour: '02', confidence: 0.82 },
  { hour: '04', confidence: 0.86 },
  { hour: '06', confidence: 0.84 },
  { hour: '08', confidence: 0.81 },
  { hour: '10', confidence: 0.87 },
  { hour: '12', confidence: 0.91 },
  { hour: '14', confidence: 0.85 },
  { hour: '16', confidence: 0.89 },
  { hour: '18', confidence: 0.84 },
  { hour: '20', confidence: 0.87 },
  { hour: '22', confidence: 0.90 },
];

export const intentBreakdownData = [
  { name: 'Product Search', value: 63, color: '#3b82f6' },
  { name: 'Barcode Lookup', value: 21, color: '#10b981' },
  { name: 'Price Check', value: 11, color: '#f59e0b' },
];

export const languageDistributionData = [
  { name: 'es', value: 185 },
  { name: 'en', value: 42 },
  { name: 'pt', value: 15 },
  { name: 'fr', value: 5 },
];

export const topSKUs = [
  { sku: 'JU04588', product: 'KONG Wubba Zoo Koala L', hits: 31, frequency: 85 },
  { sku: 'JU04587', product: 'KONG Wubba Zoo Mandrill L', hits: 28, frequency: 72 },
  { sku: 'PT05060FX', product: 'Royal Canin Medium Adult', hits: 24, frequency: 65 },
  { sku: 'CA02219', product: 'Ferplast Cayman 80 Crate', hits: 19, frequency: 58 },
  { sku: 'JU03162', product: 'KONG Classic XL', hits: 17, frequency: 45 },
];

export const recentQueries: QueryTrace[] = [
  {
    id: '1',
    query: 'Buscar KONG Wubba Zoo',
    answer: 'He encontrado los siguientes productos KONG Wubba...',
    intent: 'product_search',
    intent_confidence: 0.95,
    latency: 21.3,
    timestamp: '2026-04-16T09:57:40',
    skus_in_answer: { 'JU04588': 1, 'JU04587': 1 },
    status: 'critical'
  },
  {
    id: '2',
    query: 'Barcode de PT05060FX',
    answer: 'El código de barras de la referencia PT05060FX es 9330...',
    intent: 'barcode_lookup',
    intent_confidence: 0.98,
    latency: 8.4,
    timestamp: '2026-04-16T09:55:12',
    skus_in_answer: { 'PT05060FX': 1 },
    status: 'warning'
  },
  {
    id: '3',
    query: '¿Precio de Royal Canin Medium?',
    answer: 'El precio para Royal Canin Medium Adult es...',
    intent: 'price_check',
    intent_confidence: 0.87,
    latency: 12.1,
    timestamp: '2026-04-16T09:52:05',
    skus_in_answer: { 'PT05060FX': 1 },
    status: 'warning'
  },
  {
    id: '4',
    query: 'Buscar transportin para gato',
    answer: 'Tenemos varios modelos de transportines...',
    intent: 'product_search',
    intent_confidence: 0.82,
    latency: 18.6,
    timestamp: '2026-04-16T09:48:30',
    skus_in_answer: {},
    status: 'critical'
  }
];

export const healthStatus = [
  { field: 'Trace ID', populated: true },
  { field: 'Intent field', populated: true },
  { field: 'User language', populated: true },
  { field: 'SKU counts', populated: true },
  { field: 'Model name', populated: false },
  { field: 'Token usage', populated: false },
  { field: 'Cost fields', populated: false },
  { field: 'Time to first token', populated: false },
  { field: 'UserID', populated: false },
  { field: 'SessionID', populated: false },
  { field: 'Scores', populated: false },
  { field: 'Corrections', populated: false },
];

export const alerts = [
  {
    level: 'critical',
    title: 'Avg latency 21s — target should be <5s',
    description: 'Investigate retrieval pipeline.'
  },
  {
    level: 'critical',
    title: 'Cost fields null — model name not passed',
    description: 'Token costs untracked, no budget visibility.'
  },
  {
    level: 'warning',
    title: '11% zero-result rate — catalog gaps',
    description: 'Review zero-result queries weekly.'
  },
  {
    level: 'info',
    title: 'Scores and corrections not logged',
    description: 'Add thumbs-up/down to chatbot UI.'
  }
];
