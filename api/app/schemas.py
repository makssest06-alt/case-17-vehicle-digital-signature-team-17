"""Схемы ответов API."""
from typing import List, Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class EmbedResponse(BaseModel):
    model: str
    dim: int
    embedding: List[float]
    took_ms: float


class GalleryItem(BaseModel):
    id: int
    vehicle_id: Optional[str] = None
    image_path: str
    bbox: List[int]
    created_at: Optional[str] = None


class GalleryAddResponse(BaseModel):
    id: int
    vehicle_id: Optional[str] = None
    dim: int
    took_ms: float


class GalleryListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[GalleryItem]


class BulkImportRequest(BaseModel):
    images_dir: str = Field(..., description="Папка с картинками внутри DATA_DIR, например /data/crops/gallery")
    csv_path: str = Field(..., description="CSV с колонками image_id[,vehicle_id]")
    already_cropped: bool = Field(True, description="Картинки уже вырезаны по bbox, повторно не резать")
    limit: Optional[int] = Field(None, ge=1, description="Импортировать только первые N строк")
    truncate: bool = Field(False, description="Очистить галерею перед импортом")


class BulkImportResponse(BaseModel):
    imported: int
    skipped: int
    errors: List[str]
    took_s: float
    images_per_sec: float


class Candidate(BaseModel):
    id: int
    vehicle_id: Optional[str] = None
    image_path: str
    bbox: List[int]
    confidence: float


class SearchResponse(BaseModel):
    refusal: bool
    reason: Optional[str] = None
    message: Optional[str] = None
    threshold: float
    top_n: int
    model: str
    took_ms: float
    candidates: List[Candidate] = []


class HealthResponse(BaseModel):
    status: str
    model: str
    dim: int


class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool
    db_ok: bool
    gallery_size: Optional[int] = None
    detail: Optional[str] = None
