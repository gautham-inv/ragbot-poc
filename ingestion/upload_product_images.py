"""
Upload product images to Cloudinary and emit a side map keyed by SKU.

Inputs  : product_images.xlsx (sheet `product_images`) at repo root.
Outputs : data/sku_image_map.json   — { sku: {primary_image, thumbnail, images[], thumbnails[]} }
          data/upload_report.json   — run summary (counts + failures)

Environment (via .env, loaded through config.load_project_env):
  CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET

Usage:
  python ingestion/upload_product_images.py [--xlsx PATH] [--sheet NAME]
                                            [--limit N] [--sleep 0.5]
                                            [--concurrency 1]
                                            [--resume] [--dry-run]

Notes:
  * 12k+ images at 0.5s sleep → ~1-2 hours sequentially. Use --concurrency to speed up.
  * --resume skips SKUs already present in data/sku_image_map.json (by public_id match).
  * Never crashes on a single failure — failed filenames go into the report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# Ensure project root is on sys.path so we can import config.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load_project_env  # noqa: E402

load_project_env()

import cloudinary  # noqa: E402
import cloudinary.uploader  # noqa: E402


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
API_KEY = os.getenv("CLOUDINARY_API_KEY", "").strip()
API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "").strip()

CDN_BASE = f"https://res.cloudinary.com/{CLOUD_NAME}/image/upload"
UPLOAD_FOLDER = "products"
DATA_DIR = ROOT / "data"
MAP_PATH = DATA_DIR / "sku_image_map.json"
REPORT_PATH = DATA_DIR / "upload_report.json"


def _configure() -> None:
    if not (CLOUD_NAME and API_KEY and API_SECRET):
        raise SystemExit(
            "Cloudinary credentials missing. Set CLOUDINARY_CLOUD_NAME / "
            "CLOUDINARY_API_KEY / CLOUDINARY_API_SECRET in .env."
        )
    cloudinary.config(
        cloud_name=CLOUD_NAME,
        api_key=API_KEY,
        api_secret=API_SECRET,
        secure=True,
    )


# --------------------------------------------------------------------------
# Cloudinary upload (with retries + fallback)
# --------------------------------------------------------------------------
def _drive_url(file_id: str) -> str:
    """Legacy Drive URL — often returns an interstitial HTML page for non-public or
    large files, which Cloudinary then rejects with 'Invalid image file'."""
    return f"https://drive.google.com/uc?export=view&id={file_id}"


def _drive_candidate_urls(file_id: str, public_url: str) -> list[str]:
    """
    URL candidates in order of reliability for Google-Drive-hosted images.
    The `lh3` CDN serves publicly-shared images directly (no HTML wrapper),
    and is by far the most reliable source for Cloudinary remote-fetch.
    """
    urls: list[str] = []
    if file_id:
        urls.append(f"https://lh3.googleusercontent.com/d/{file_id}")
        urls.append(f"https://drive.usercontent.google.com/download?id={file_id}&export=view")
        urls.append(f"https://drive.google.com/uc?export=download&id={file_id}")
    if public_url and public_url not in urls:
        urls.append(public_url)
    # de-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _looks_like_image_bytes(content: bytes, content_type: str | None) -> bool:
    """Reject Drive-interstitial HTML / login pages before uploading to Cloudinary."""
    if content_type:
        ct = content_type.split(";", 1)[0].strip().lower()
        if ct.startswith("image/"):
            return True
        if ct.startswith("text/") or ct == "application/json":
            return False
    # Magic-byte sniff as a last resort.
    if len(content) < 16:
        return False
    if content.startswith(b"\xff\xd8\xff"):        return True  # JPEG
    if content.startswith(b"\x89PNG\r\n\x1a\n"):   return True  # PNG
    if content[:4] == b"GIF8":                     return True  # GIF
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP": return True  # WEBP
    return False


def upload_one(
    *,
    public_url: str,
    file_id: str,
    public_id: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """
    Upload a single image. Tries each candidate Drive URL via Cloudinary
    remote-fetch first; on failure, downloads bytes (with Content-Type
    validation) and uploads those. Returns the Cloudinary secure_url.
    Raises RuntimeError after all attempts are exhausted.
    """
    last_err: Exception | None = None
    candidates = _drive_candidate_urls(file_id, public_url)

    # Path 1: Cloudinary fetches each candidate directly.
    for url_candidate in candidates:
        for _ in range(1, max_retries + 1):
            try:
                res = cloudinary.uploader.upload(
                    url_candidate,
                    public_id=public_id,
                    folder=UPLOAD_FOLDER,
                    overwrite=True,
                    unique_filename=False,
                    resource_type="image",
                )
                url = res.get("secure_url") or res.get("url")
                if url:
                    return url
                last_err = RuntimeError("empty response from Cloudinary")
            except Exception as e:
                last_err = e
                # Non-retryable: URL format rejected. Skip to next candidate.
                if "Invalid image file" in str(e):
                    break
            time.sleep(retry_delay)

    # Path 2: fallback — fetch bytes ourselves, validate, then upload.
    for url_candidate in candidates:
        for _ in range(1, max_retries + 1):
            try:
                r = requests.get(
                    url_candidate,
                    timeout=30,
                    allow_redirects=True,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                r.raise_for_status()
                if not _looks_like_image_bytes(r.content, r.headers.get("content-type")):
                    last_err = RuntimeError(
                        f"non-image response from {url_candidate!r} (ct={r.headers.get('content-type')})"
                    )
                    break  # try next candidate
                res = cloudinary.uploader.upload(
                    r.content,
                    public_id=public_id,
                    folder=UPLOAD_FOLDER,
                    overwrite=True,
                    unique_filename=False,
                    resource_type="image",
                )
                url = res.get("secure_url") or res.get("url")
                if url:
                    return url
                last_err = RuntimeError("empty response from Cloudinary (bytes path)")
            except Exception as e:
                last_err = e
            time.sleep(retry_delay)

    raise RuntimeError(f"upload failed for public_id={public_id!r}: {last_err}")


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def read_xlsx(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    required = {"sku", "image_index", "file_name", "folder", "file_id", "public_url"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"xlsx missing required columns: {missing}")
    df = df.dropna(subset=["sku", "file_name"])
    df["sku"] = df["sku"].astype(str).str.strip()
    df["file_name"] = df["file_name"].astype(str).str.strip()
    df["image_index"] = pd.to_numeric(df["image_index"], errors="coerce").fillna(0).astype(int)
    df["folder"] = df["folder"].fillna("").astype(str).str.strip()
    df["file_id"] = df["file_id"].fillna("").astype(str).str.strip()
    df["public_url"] = df["public_url"].fillna("").astype(str).str.strip()
    print(f"[xlsx] shape={df.shape}")
    print(df.head(5).to_string(index=False))
    return df


def public_id_for(file_name: str) -> str:
    stem = file_name
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    # Cloudinary rejects public_ids that contain spaces or unsafe chars like `&`, `?`, `#`.
    # Replace them with `-` so filenames like "LOGO-M&C" or "... product callout.jpg" survive.
    import re
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-")
    return stem or "image"


def load_existing_map() -> dict[str, dict[str, Any]]:
    if MAP_PATH.exists():
        try:
            return json.loads(MAP_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def upload_all(
    df: pd.DataFrame,
    *,
    sleep_s: float,
    concurrency: int,
    resume: dict[str, dict[str, Any]],
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns (success_rows, failure_rows).
    success: {sku, image_index, public_id, cloudinary_url, folder}
    failure: {sku, file_name, error}
    """
    rows: list[dict[str, Any]] = df.to_dict("records")
    total = len(rows)

    # Resume: skip file_names already known.
    known_public_ids: set[str] = set()
    for entry in resume.values():
        for url in (entry or {}).get("images", []) or []:
            pid = url.rsplit("/", 1)[-1]
            known_public_ids.add(pid)

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def _do(row: dict[str, Any]) -> dict[str, Any] | None:
        pid = public_id_for(row["file_name"])
        if pid in known_public_ids:
            return {
                "sku": row["sku"],
                "image_index": int(row["image_index"]),
                "public_id": pid,
                "cloudinary_url": f"{CDN_BASE}/{UPLOAD_FOLDER}/{pid}",
                "folder": row["folder"],
                "skipped": True,
            }
        if dry_run:
            return {
                "sku": row["sku"],
                "image_index": int(row["image_index"]),
                "public_id": pid,
                "cloudinary_url": f"{CDN_BASE}/{UPLOAD_FOLDER}/{pid}",
                "folder": row["folder"],
                "dry_run": True,
            }
        try:
            url = upload_one(
                public_url=row["public_url"] or _drive_url(row["file_id"]),
                file_id=row["file_id"],
                public_id=pid,
            )
            return {
                "sku": row["sku"],
                "image_index": int(row["image_index"]),
                "public_id": pid,
                "cloudinary_url": url,
                "folder": row["folder"],
            }
        except Exception as e:
            return {"sku": row["sku"], "file_name": row["file_name"], "error": str(e), "failed": True}

    if concurrency <= 1:
        for i, row in enumerate(rows, start=1):
            out = _do(row)
            if out and out.get("failed"):
                failures.append(out)
                print(f"  [{i}/{total}] FAIL {out['file_name']}: {out['error']}")
            else:
                successes.append(out)  # type: ignore[arg-type]
                tag = "skip" if out.get("skipped") else ("dry" if out.get("dry_run") else "ok")
                print(f"  [{i}/{total}] {tag} {row['sku']} #{int(row['image_index'])} → {out['public_id']}")
            if sleep_s > 0 and not dry_run and not out.get("skipped"):
                time.sleep(sleep_s)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(_do, row): (i, row) for i, row in enumerate(rows, start=1)}
            for fut in as_completed(futures):
                i, row = futures[fut]
                out = fut.result()
                if out and out.get("failed"):
                    failures.append(out)
                    print(f"  [{i}/{total}] FAIL {out['file_name']}: {out['error']}")
                else:
                    successes.append(out)  # type: ignore[arg-type]
                    tag = "skip" if out.get("skipped") else ("dry" if out.get("dry_run") else "ok")
                    print(f"  [{i}/{total}] {tag} {row['sku']} #{int(row['image_index'])} → {out['public_id']}")

    return successes, failures


def group_by_sku(successes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for s in successes:
        groups.setdefault(s["sku"], []).append(s)
    for sku in groups:
        groups[sku].sort(key=lambda r: r["image_index"])
    return groups


def build_image_map(groups: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for sku, imgs in groups.items():
        if not imgs:
            continue
        images = [f"{CDN_BASE}/w_800,q_auto,f_auto/{UPLOAD_FOLDER}/{i['public_id']}" for i in imgs]
        thumbnails = [f"{CDN_BASE}/w_400,q_auto,f_auto/{UPLOAD_FOLDER}/{i['public_id']}" for i in imgs]
        out[sku] = {
            "sku": sku,
            "folder": imgs[0].get("folder") or "",
            "primary_image": images[0],
            "thumbnail": thumbnails[0],
            "images": images,
            "thumbnails": thumbnails,
        }
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Upload product images to Cloudinary.")
    ap.add_argument("--xlsx", default=str(ROOT / "product_images.xlsx"))
    ap.add_argument("--sheet", default="product_images")
    ap.add_argument("--limit", type=int, default=0, help="limit to first N rows (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between sequential uploads")
    ap.add_argument("--concurrency", type=int, default=1, help=">1 enables parallel uploads (ignores --sleep)")
    ap.add_argument("--resume", action="store_true", help="skip public_ids already present in data/sku_image_map.json")
    ap.add_argument("--dry-run", action="store_true", help="list what would happen, no network calls")
    args = ap.parse_args()

    _configure()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = read_xlsx(Path(args.xlsx), args.sheet)
    if args.limit > 0:
        df = df.head(args.limit)

    resume_map = load_existing_map() if args.resume else {}

    t0 = time.time()
    print(f"[upload] {len(df)} rows · concurrency={args.concurrency} · sleep={args.sleep}s"
          f"{' · resume' if args.resume else ''}{' · DRY-RUN' if args.dry_run else ''}")

    successes, failures = upload_all(
        df, sleep_s=args.sleep, concurrency=args.concurrency,
        resume=resume_map, dry_run=args.dry_run,
    )

    # Merge with any prior map on resume, so we don't lose previously-good SKUs.
    groups = group_by_sku(successes)
    image_map = build_image_map(groups)
    if args.resume:
        # previous entries stay, new ones overwrite
        merged = dict(resume_map)
        merged.update(image_map)
        image_map = merged

    if not args.dry_run:
        MAP_PATH.write_text(json.dumps(image_map, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "total_rows": int(len(df)),
        "total_successes": len([s for s in successes if not s.get("failed") and not s.get("skipped")]),
        "total_skipped": len([s for s in successes if s.get("skipped")]),
        "total_failures": len(failures),
        "total_skus": len(image_map),
        "failures": failures,
        "elapsed_sec": round(time.time() - t0, 1),
        "map_path": str(MAP_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("================ Summary ================")
    print(f"  Total rows              : {report['total_rows']}")
    print(f"  Uploaded OK             : {report['total_successes']}")
    print(f"  Skipped (resume)        : {report['total_skipped']}")
    print(f"  Failures                : {report['total_failures']}")
    print(f"  SKUs with at least 1 img: {report['total_skus']}")
    print(f"  Elapsed                 : {report['elapsed_sec']}s")
    print(f"  Map written to          : {MAP_PATH}")
    print(f"  Report written to       : {REPORT_PATH}")
    if failures:
        print("  Failed filenames:")
        for f in failures[:20]:
            print(f"    - {f['file_name']}: {f['error']}")
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more (see upload_report.json)")


if __name__ == "__main__":
    main()
