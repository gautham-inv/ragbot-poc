"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useEffect, useRef, useState } from "react";

type SourceChunk = {
  chunk_id?: string;
  text?: string;
  metadata?: Record<string, unknown>;
  score?: number;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceChunk[];
  rewritten_query?: string;
  enriched_query?: string;
};

function newId(prefix: string) {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const c: any = globalThis.crypto;
    if (c?.randomUUID) return `${prefix}-${c.randomUUID()}`;
  } catch {
    // ignore
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}


function MicIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={className ?? "h-4 w-4"}
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      aria-hidden="true"
    >
      <path d="M12 1a3 3 0 0 1 3 3v8a3 3 0 0 1-6 0V4a3 3 0 0 1 3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <path d="M12 19v4" />
      <path d="M8 23h8" />
    </svg>
  );
}

function Waveform() {
  return (
    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-brand-50">
      <div className="flex items-end gap-1.5">
        <span className="h-3 w-1.5 animate-pulse rounded-full bg-brand-500" />
        <span
          className="h-6 w-1.5 animate-pulse rounded-full bg-brand-500"
          style={{ animationDelay: "120ms" }}
        />
        <span
          className="h-4 w-1.5 animate-pulse rounded-full bg-brand-500"
          style={{ animationDelay: "240ms" }}
        />
        <span
          className="h-7 w-1.5 animate-pulse rounded-full bg-brand-500"
          style={{ animationDelay: "360ms" }}
        />
      </div>
    </div>
  );
}

function ThinkingDots() {
  return (
    <div className="flex items-center gap-1">
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-slate-300" />
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-slate-300" style={{ animationDelay: "150ms" }} />
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-slate-300" style={{ animationDelay: "300ms" }} />
    </div>
  );
}

type StreamEvent =
  | { type: "status"; message: string }
  | { type: "rewrite"; rewritten_query: string }
  | { type: "enrich"; enriched_query: string }
  | { type: "token"; delta: string }
  | {
      type: "done";
      answer: string;
      sources: SourceChunk[];
      rewritten_query?: string;
      enriched_query?: string;
    }
  | { type: "error"; message: string };

function parseSseEvents(buffer: string) {
  const events: string[] = [];
  let rest = buffer;
  while (true) {
    const sep = rest.indexOf("\n\n");
    if (sep === -1) break;
    events.push(rest.slice(0, sep));
    rest = rest.slice(sep + 2);
  }
  return { events, rest };
}

function getSseData(rawEvent: string) {
  const lines = rawEvent.split("\n");
  const dataLines = lines.filter((l) => l.startsWith("data:"));
  if (dataLines.length === 0) return "";
  return dataLines.map((l) => l.slice("data:".length).trimStart()).join("\n");
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [voiceMode, setVoiceMode] = useState<"idle" | "recording" | "processing">("idle");
  const [recordingSeconds, setRecordingSeconds] = useState(0);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const RECORDING_MAX_SECONDS = 240;

  const requestIdRef = useRef(0);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      try {
        recorderRef.current?.stop();
      } catch {
        // ignore
      }
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  function formatTime(seconds: number) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes.toString().padStart(2, "0")}:${remainingSeconds.toString().padStart(2, "0")}`;
  }

  function updateAssistantMessage(assistantId: string, updater: (m: Message) => Message) {
    setMessages((prev) => prev.map((m) => (m.id === assistantId ? updater(m) : m)));
  }

  async function sendPrompt(prompt: string) {
    const text = prompt.trim();
    if (!text || loading) return;

    const reqId = ++requestIdRef.current;
    const userId = newId("u");
    const assistantId = newId("a");

    const historyForBackend = messages.slice(-8).map((m) => ({ role: m.role, content: m.content }));
    setMessages((prev) => [
      ...prev,
      { id: userId, role: "user", content: text },
      { id: assistantId, role: "assistant", content: "" }
    ]);

    setInput("");
    setLoading(true);
    setLoadingStatus("Starting...");
    setStreaming(false);

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
      const res = await fetch(`${baseUrl}/api/chat_stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: text,
          history: historyForBackend
        })
      });

      if (!res.ok || !res.body) {
        // Fallback to non-streaming response if streaming is unavailable.
        const fallback = await fetch(`${baseUrl}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: text, history: historyForBackend })
        });
        const data = await fallback.json();
        const answer = typeof data?.answer === "string" ? data.answer : "No response.";
        const sources = Array.isArray(data?.sources) ? (data.sources as SourceChunk[]) : [];
        updateAssistantMessage(assistantId, (m) => ({
          ...m,
          content: answer,
          sources,
          rewritten_query: typeof data?.rewritten_query === "string" ? data.rewritten_query : undefined,
          enriched_query: typeof data?.enriched_query === "string" ? data.enriched_query : undefined
        }));
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (requestIdRef.current !== reqId) break;

        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
        const parsed = parseSseEvents(buffer);
        buffer = parsed.rest;

        for (const rawEvent of parsed.events) {
          const data = getSseData(rawEvent);
          if (!data) continue;

          let evt: StreamEvent | null = null;
          try {
            evt = JSON.parse(data) as StreamEvent;
          } catch {
            continue;
          }
          if (!evt) continue;

          if (evt.type === "status") {
            setLoadingStatus(evt.message);
            continue;
          }

          if (evt.type === "rewrite") {
            updateAssistantMessage(assistantId, (m) => ({ ...m, rewritten_query: evt.rewritten_query }));
            continue;
          }

          if (evt.type === "enrich") {
            updateAssistantMessage(assistantId, (m) => ({ ...m, enriched_query: evt.enriched_query }));
            continue;
          }

          if (evt.type === "token") {
            setStreaming(true);
            updateAssistantMessage(assistantId, (m) => ({ ...m, content: (m.content || "") + evt.delta }));
            continue;
          }

          if (evt.type === "done") {
            updateAssistantMessage(assistantId, (m) => ({
              ...m,
              content: evt.answer ?? m.content,
              sources: evt.sources ?? m.sources,
              rewritten_query: evt.rewritten_query ?? m.rewritten_query,
              enriched_query: evt.enriched_query ?? m.enriched_query
            }));
            setLoading(false);
            setLoadingStatus(null);
            return;
          }

          if (evt.type === "error") {
            updateAssistantMessage(assistantId, (m) => ({ ...m, content: evt.message || "Error." }));
            setLoading(false);
            setLoadingStatus(null);
            return;
          }
        }
      }
    } catch {
      updateAssistantMessage(assistantId, (m) => ({ ...m, content: "Error contacting server." }));
    } finally {
      if (requestIdRef.current === reqId) {
        setLoading(false);
        setLoadingStatus(null);
      }
    }
  }

  async function toggleRecording() {
    if (voiceMode === "processing") return;

    if (voiceMode === "recording") {
      setVoiceMode("processing");
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;

      try {
        recorderRef.current?.stop();
      } catch {
        setVoiceMode("idle");
      }
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mediaRecorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      };
      mediaRecorder.onstop = async () => {
        try {
          if (chunks.length === 0) return;

          const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
          const form = new FormData();
          form.append("file", blob, "audio.webm");

          const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
          const res = await fetch(`${baseUrl}/api/transcribe`, { method: "POST", body: form });
          const data = await res.json();
          const text = typeof data?.text === "string" ? data.text.trim() : "";
          if (text) setInput(text);
        } catch {
          setMessages((prev) => [
            ...prev,
            { id: newId("a"), role: "assistant", content: "Voice transcription failed." }
          ]);
        } finally {
          setVoiceMode("idle");
          setRecordingSeconds(0);
          recorderRef.current = null;

          streamRef.current?.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
        }
      };

      recorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setVoiceMode("recording");
      setRecordingSeconds(0);

      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = setInterval(() => {
        setRecordingSeconds((prev) => {
          const next = prev + 1;
          if (next >= RECORDING_MAX_SECONDS) {
            setVoiceMode("processing");
            try {
              recorderRef.current?.stop();
            } catch {
              setVoiceMode("idle");
            }
            if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
            recordingTimerRef.current = null;
            return RECORDING_MAX_SECONDS;
          }
          return next;
        });
      }, 1000);
    } catch {
      setMessages((prev) => [
        ...prev,
        { id: newId("a"), role: "assistant", content: "Microphone access denied." }
      ]);
    }
  }

  const sampleQueries = [
    { label: "Precio del TR00400" },
    { label: "Barcode de PT05060FX" },
    { label: "Marca en la pagina 300" },
    { label: "Buscar KONG Wubba Zoo" }
  ];

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900">
      <aside className="sticky top-0 hidden h-screen w-72 flex-col gap-6 overflow-y-auto border-r border-slate-200 bg-white p-6 md:flex">
        <div className="flex items-center gap-2 text-base font-semibold text-slate-800">
          <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-brand-50 text-brand-600">
            <img src="/paw.jpg" alt="Bot" className="h-full w-full object-cover" />
          </span>
          Gloria Pets Catalog Bot
        </div>
        <button onClick={() => setMessages([])} className="rounded-xl bg-brand-600 px-4 py-3 text-sm font-semibold text-white">
          New chat +
        </button>
        <div className="space-y-2 pt-2 text-xs text-slate-500">
          <div className="font-semibold uppercase tracking-wide text-slate-400">Actions</div>
          <button
            onClick={() => fetch((process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000") + "/warmup")}
            className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-left text-xs text-slate-600 hover:border-brand-300"
          >
            Warm up
          </button>
          <button
            onClick={() => setMessages([])}
            className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-left text-xs text-slate-600 hover:border-brand-300"
          >
            Clear chat
          </button>
        </div>
      </aside>

      <main className="flex h-screen flex-1 flex-col">
        <div className="flex-1 overflow-y-auto px-6 py-8 md:px-8">
          <div className="mx-auto w-full max-w-3xl">
          {messages.length === 0 && (
            <div className="mt-10 flex w-full flex-col items-center text-center">
              <div className="mb-4 flex h-16 w-16 items-center justify-center overflow-hidden rounded-full border border-slate-200 bg-white shadow-sm">
                <img src="/paw.jpg" alt="Bot" className="h-full w-full object-cover" />
              </div>
              <h1 className="text-2xl font-semibold text-slate-800 md:text-3xl">
                Chatea con tu catalogo de productos (Chat with your product catalog)
              </h1>
              <p className="mt-3 text-sm text-slate-500">
                Busca entre miles de SKUs, precios y tablas en 396 paginas de datos (Searching through thousands of SKUs, prices, and tables across 396 pages of data).
              </p>
              <div className="mt-6 grid w-full gap-3 sm:grid-cols-2">
                {sampleQueries.map((q) => (
                  <button
                    key={q.label}
                    onClick={() => sendPrompt(q.label)}
                    className="flex items-center gap-3 rounded-2xl border border-brand-100 bg-brand-50 px-4 py-3 text-left text-sm text-brand-700 shadow-sm"
                  >
                    <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-white shadow">
                      <svg viewBox="0 0 24 24" className="h-4 w-4 text-brand-600" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                        <circle cx="11" cy="11" r="7" />
                        <path d="M20 20l-3.5-3.5" />
                      </svg>
                    </span>
                    <span className="text-xs font-semibold">{q.label}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.length > 0 && (
            <div className="mt-6 w-full space-y-4 pb-8">
              {messages.map((m) => {
                const hideEmptyAssistant =
                  m.role === "assistant" && (!m.content || m.content.trim() === "") && (!m.sources || m.sources.length === 0);
                if (hideEmptyAssistant) return null;

                return (
                <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"} gap-3`}>
                  {m.role === "assistant" && (
                    <div className="mt-1 flex h-12 w-12 flex-none items-center justify-center overflow-hidden rounded-full border border-slate-200 bg-white">
                      <img src="/paw.jpg" alt="Bot" className="h-full w-full object-cover" />
                    </div>
                  )}
                  <div className="flex w-full max-w-[85%] flex-col gap-1">
                    <div
                      className={`rounded-2xl px-4 py-3 text-sm shadow-sm ${
                        m.role === "user"
                          ? "ml-auto bg-brand-600 text-white"
                          : "border border-slate-200 bg-white text-slate-700"
                      }`}
                    >
                      {m.role === "assistant" ? (
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          className="space-y-2"
                          components={{
                            p: ({ children }) => <p className="whitespace-pre-wrap">{children}</p>,
                            strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                            ul: ({ children }) => <ul className="list-disc space-y-1 pl-5">{children}</ul>,
                            ol: ({ children }) => <ol className="list-decimal space-y-1 pl-5">{children}</ol>,
                            code: ({ children }) => <code className="rounded bg-slate-50 px-1 py-0.5 text-[13px]">{children}</code>
                          }}
                        >
                          {m.content || ""}
                        </ReactMarkdown>
                      ) : (
                        <div className="whitespace-pre-wrap">{m.content}</div>
                      )}

                      {m.role === "assistant" && m.sources && m.sources.length > 0 && (
                        <details className="mt-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                          <summary className="cursor-pointer text-xs font-semibold text-slate-600">
                            Source documents ({m.sources.length})
                          </summary>
                          <div className="mt-2 space-y-2">
                            <div className="space-y-2 pt-1">
                              {m.sources.slice(0, 8).map((s, sIdx) => {
                                const meta = (s.metadata ?? {}) as Record<string, unknown>;
                                const physical = meta.physical_page_number ?? meta.page_number;
                                const sku = meta.sku;
                                const brand = meta.brand;
                                const chunkType = meta.chunk_type;
                                const parts: string[] = [];
                                if (typeof physical !== "undefined") parts.push(`Physical Page ${String(physical)}`);
                                if (typeof sku === "string" && sku) parts.push(`SKU ${sku}`);
                                if (typeof brand === "string" && brand) parts.push(String(brand));
                                if (typeof chunkType === "string" && chunkType) parts.push(String(chunkType));
                                if (typeof s.score === "number") parts.push(`Score ${s.score.toFixed(4)}`);

                                return (
                                  <div key={s.chunk_id ?? `${m.id}-${sIdx}`} className="rounded-md bg-white px-3 py-2">
                                    <div className="text-[11px] font-semibold text-slate-700">
                                      {parts.length ? parts.join(" - ") : "Source"}
                                    </div>
                                    {s.text && <div className="mt-1 text-[11px] text-slate-600">{s.text}</div>}
                                  </div>
                                );
                              })}
                              {m.sources.length > 8 && (
                                <div className="text-[11px] text-slate-500">Showing first 8 sources.</div>
                              )}
                            </div>
                          </div>
                        </details>
                      )}
                    </div>
                    
                    {m.role === "assistant" && (
                      <div className="relative ml-2 flex items-center">
                        <button
                          onClick={() => setOpenMenuId(openMenuId === m.id ? null : m.id)}
                          className="rounded-md p-1.5 text-slate-400 transition-colors hover:bg-slate-200 hover:text-slate-600"
                          title="More options"
                        >
                          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor">
                            <circle cx="5" cy="12" r="2" />
                            <circle cx="12" cy="12" r="2" />
                            <circle cx="19" cy="12" r="2" />
                          </svg>
                        </button>
                        {openMenuId === m.id && (
                          <>
                            <div className="fixed inset-0 z-10" onClick={() => setOpenMenuId(null)} />
                            <div className="absolute left-0 top-8 z-20 w-48 rounded-lg border border-slate-200 bg-white py-1 text-sm shadow-lg">
                              <button
                                onClick={() => {
                                  navigator.clipboard.writeText(m.content);
                                  setOpenMenuId(null);
                                }}
                                className="flex w-full items-center gap-2 px-4 py-2 text-left text-slate-700 hover:bg-slate-50"
                              >
                                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2">
                                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                                </svg>
                                Copy Response
                              </button>
                              <button
                                onClick={() => {
                                  setOpenMenuId(null);
                                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                                  if (typeof window !== "undefined" && (window as any).Userback) {
                                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                                    (window as any).Userback.open("feedback");
                                  } else {
                                    alert("Feedback widget is not loaded yet.");
                                  }
                                }}
                                className="flex w-full items-center gap-2 px-4 py-2 text-left text-slate-700 hover:bg-slate-50"
                              >
                                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2">
                                  <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
                                </svg>
                                Send Feedback
                              </button>
                            </div>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )})}

              {loading && !streaming && (
                <div className="flex justify-start">
                  <div className="w-full max-w-md rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm shadow-sm">
                    <div className="flex items-center gap-3">
                      <img src="/dog.gif" alt="Loading" className="h-7 w-7 flex-none" />
                      <div className="min-w-0 flex-1 truncate text-slate-600">
                        {loadingStatus ?? "Thinking..."}
                      </div>
                      <div className="ml-auto">
                        <ThinkingDots />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          )}
          </div>
        </div>

        <div className="sticky bottom-0 border-t border-slate-200 bg-slate-50 px-6 py-4 md:px-8">
          <div className="mx-auto w-full max-w-3xl">
            <div className="rounded-full border border-slate-200 bg-white px-3 py-2 shadow-sm">
              <div className="flex items-center gap-3">
              <div className="min-w-0 flex-1">
                {voiceMode === "idle" ? (
                  <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !loading) sendPrompt(input);
                    }}
                    placeholder="Pregunta sobre un producto, SKU o precio... (Ask about a product, SKU or price...)"
                    className="w-full bg-transparent text-sm text-slate-700 outline-none"
                    disabled={loading}
                  />
                ) : voiceMode === "recording" ? (
                  <div className="flex items-center gap-3 py-0.5">
                    <Waveform />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-semibold text-slate-800">Recording…</div>
                      <div className="text-xs text-slate-500">
                        {formatTime(recordingSeconds)} / {formatTime(RECORDING_MAX_SECONDS)}
                      </div>
                    </div>
                    <div className="hidden text-xs text-slate-400 sm:block">Tap mic to stop</div>
                  </div>
                ) : (
                  <div className="flex items-center gap-3 py-2 text-sm text-slate-600">
                    <ThinkingDots />
                    <div className="min-w-0 flex-1 truncate">Processing…</div>
                  </div>
                )}
              </div>

              <button
                onClick={toggleRecording}
                className={`flex h-9 w-9 items-center justify-center rounded-full border ${
                  voiceMode === "recording"
                    ? "border-brand-500 bg-brand-50 text-brand-700"
                    : voiceMode === "processing"
                      ? "cursor-not-allowed border-slate-200 bg-slate-50 text-slate-300"
                      : "border-slate-200 bg-slate-50 text-slate-600"
                }`}
                title="Voice input"
                type="button"
                disabled={voiceMode === "processing"}
              >
                <MicIcon className="h-4 w-4" />
              </button>

              <button
                onClick={() => sendPrompt(input)}
                className="rounded-full bg-brand-600 px-4 py-2 text-xs font-semibold text-white disabled:opacity-60"
                disabled={loading || voiceMode !== "idle"}
              >
                {loading ? "..." : "Send"}
              </button>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
