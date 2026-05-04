"""
Admin product CRUD HTTP layer.

Mounted at /admin/products in backend/app.py. Every route requires a Better Auth
session whose user.role == "admin". The session is verified by forwarding the
incoming cookie to the auth-server's /api/auth/get-session endpoint.

The actual storage logic lives in services/products.py — this module is HTTP +
auth only.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from services import products as products_service
from services.products import (
    ProductNotFoundError,
    ProductValidationError,
)

# auth-server is reachable on the internal Docker network (compose service name).
# Falls back to localhost for dev outside Docker.
AUTH_SERVER_URL = os.getenv("AUTH_SERVER_INTERNAL_URL", "http://auth-server:4000")
SESSION_PATH = "/api/auth/get-session"


router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------
async def require_admin(request: Request) -> dict[str, Any]:
    cookie = request.headers.get("cookie") or ""
    if not cookie:
        raise HTTPException(status_code=401, detail="missing session cookie")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(
                f"{AUTH_SERVER_URL}{SESSION_PATH}",
                headers={"cookie": cookie},
            )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"auth-server unreachable: {e}")

    if res.status_code != 200:
        raise HTTPException(status_code=401, detail="invalid session")

    try:
        data = res.json() or {}
    except Exception:
        raise HTTPException(status_code=401, detail="invalid session payload")

    user = (data or {}).get("user") or {}
    if not user:
        raise HTTPException(status_code=401, detail="no user in session")
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    return user


# ---------------------------------------------------------------------------
# Request/response shapes
# ---------------------------------------------------------------------------
class ProductIn(BaseModel):
    sku: str
    brand: str
    name_es: str | None = None
    name_en: str | None = None
    name_fr: str | None = None
    name_pt: str | None = None
    name_it: str | None = None
    category: str | None = None
    subcategory: str | None = None
    species: list[str] | None = None
    price_pvpr: float | None = None
    price_per_unit: float | None = None
    min_purchase_qty: int | None = None
    weight_g: float | None = None
    color: str | None = None
    scent: str | None = None
    ean: str | None = None
    # Free-form passthrough for fields the form doesn't model directly.
    extra: dict[str, Any] | None = None


class ProductPatch(BaseModel):
    brand: str | None = None
    name_es: str | None = None
    name_en: str | None = None
    name_fr: str | None = None
    name_pt: str | None = None
    name_it: str | None = None
    category: str | None = None
    subcategory: str | None = None
    species: list[str] | None = None
    price_pvpr: float | None = None
    price_per_unit: float | None = None
    min_purchase_qty: int | None = None
    weight_g: float | None = None
    color: str | None = None
    scent: str | None = None
    ean: str | None = None
    extra: dict[str, Any] | None = None


def _flatten(model: BaseModel) -> dict[str, Any]:
    raw = model.model_dump(exclude_none=True)
    extra = raw.pop("extra", None) or {}
    return {**extra, **raw}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("/products/{sku}")
async def get_product(sku: str, _: dict = Depends(require_admin)):
    try:
        return products_service.get_product(sku)
    except ProductNotFoundError:
        raise HTTPException(status_code=404, detail="not found")


@router.get("/products")
async def list_products(
    limit: int = 50,
    offset: str | None = None,
    _: dict = Depends(require_admin),
):
    rows, next_offset = products_service.list_products(limit=limit, offset=offset)
    return {"products": rows, "next_offset": next_offset}


@router.post("/products", status_code=201)
async def create_product(body: ProductIn, _: dict = Depends(require_admin)):
    try:
        payload = products_service.create_product(_flatten(body))
    except ProductValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    products_service.rebuild_bm25()
    return payload


@router.patch("/products/{sku}")
async def update_product(sku: str, body: ProductPatch, _: dict = Depends(require_admin)):
    try:
        payload = products_service.update_product(sku, _flatten(body))
    except ProductNotFoundError:
        raise HTTPException(status_code=404, detail="not found")
    except ProductValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    products_service.rebuild_bm25()
    return payload


@router.delete("/products/{sku}", status_code=204)
async def delete_product(sku: str, _: dict = Depends(require_admin)):
    try:
        products_service.delete_product(sku)
    except ProductNotFoundError:
        raise HTTPException(status_code=404, detail="not found")
    products_service.rebuild_bm25()
    return None


@router.post("/products/{sku}/images", status_code=201)
async def add_product_image(
    sku: str,
    file: UploadFile = File(...),
    _: dict = Depends(require_admin),
):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        return products_service.add_product_image(sku, file_bytes)
    except ProductNotFoundError:
        raise HTTPException(status_code=404, detail="product not found")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upload failed: {e}")


@router.delete("/products/{sku}/images/{public_id}", status_code=204)
async def remove_product_image(
    sku: str,
    public_id: str,
    _: dict = Depends(require_admin),
):
    try:
        products_service.remove_product_image(sku, public_id)
    except ProductNotFoundError:
        raise HTTPException(status_code=404, detail="product not found")
    return None
