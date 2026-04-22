"use client";

import { signIn } from "@/lib/auth-client";
import { useRouter, useSearchParams } from "next/navigation";
import Image from "next/image";
import { useMemo, useState, Suspense } from "react";

function LoginContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const nextPath = useMemo(() => searchParams.get("next") || "/", [searchParams]);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const res = await signIn.email({
        email,
        password,
        callbackURL: nextPath,
      });

      if (res?.error) {
        setError(String(res.error?.message || res.error));
        return;
      }

      router.replace(nextPath);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="min-h-screen flex items-center justify-center p-6 bg-[var(--background)] text-[var(--foreground)]">
      <div className="w-full max-w-md rounded-xl border p-6 shadow-sm bg-[var(--surface)] border-[var(--border)]">
        <div className="flex items-center gap-2">
          <Image src="/paw.jpg" alt="Gloria Pets" width={24} height={24} priority />
          <h1 className="text-xl font-semibold">Admin login</h1>
        </div>
        <p className="mt-1 text-sm text-[var(--secondary)]">Sign in to access analytics.</p>

        <form className="mt-6 space-y-4" onSubmit={onSubmit}>
          <div>
            <label className="block text-sm font-medium text-[var(--secondary)]">Email</label>
            <input
              className="mt-1 w-full rounded-lg border px-3 py-2 text-sm outline-none bg-[var(--surface)] text-[var(--foreground)] border-[var(--border)] placeholder:text-[color:var(--secondary)] focus:border-[color:var(--secondary)]"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-[var(--secondary)]">Password</label>
            <input
              className="mt-1 w-full rounded-lg border px-3 py-2 text-sm outline-none bg-[var(--surface)] text-[var(--foreground)] border-[var(--border)] placeholder:text-[color:var(--secondary)] focus:border-[color:var(--secondary)]"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          {error ? (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">{error}</div>
          ) : null}

          <button
            className="w-full rounded-lg px-3 py-2 text-sm font-medium text-white disabled:opacity-60 bg-[var(--primary)] hover:bg-[var(--primary-hover)]"
            type="submit"
            disabled={submitting}
          >
            {submitting ? "Signing in..." : "Sign in"}
          </button>
        </form>

        <div className="mt-4 text-xs text-[var(--secondary)]">
          You'll be redirected to <span className="font-mono">{nextPath}</span> after login.
        </div>

        <div className="mt-4 text-sm">
          <a className="underline text-[var(--secondary)] hover:text-[var(--foreground)]" href={`/signup?next=${encodeURIComponent(nextPath)}`}>
            Create an admin account
          </a>
        </div>
      </div>
    </main>
  );
}

export default function AdminLoginPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center bg-[var(--background)] text-[var(--secondary)]">
        <div className="animate-pulse">Loading login...</div>
      </div>
    }>
      <LoginContent />
    </Suspense>
  );
}


