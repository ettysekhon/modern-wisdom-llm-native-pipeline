from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import PointStruct

from . import paths

logger = logging.getLogger(__name__)

# Alias name used to point clients to the active collection
LIVE_ALIAS = "mw_chunks_live"


@dataclass
class VectorSpec:
    size: int
    distance: str = "COSINE"


# Build a collection name per embedding version (blue/green deployment pattern)
def collection_name_for_emb_v(emb_v: str) -> str:
    # Keep lowercase and safe for collection names
    safe = emb_v.lower().replace("/", "_").replace(" ", "_").replace("-", "_")
    return f"mw_chunks_{safe}"


def client(url: str | None = None, api_key: str | None = None) -> QdrantClient:
    """Create QdrantClient with proper configuration for Fly.io or local instances."""
    actual_url = url or paths.QDRANT_URL
    actual_api_key = api_key or paths.QDRANT_API_KEY or None  # Ensure empty string becomes None

    # Debug: log the URL being used (helps diagnose connection issues)
    logger.info(f"Initializing QdrantClient with URL: {actual_url}")

    # Detect if this is a Fly.io public deployment (.fly.dev) or internal networking (.internal/.flycast)
    is_fly_io_public = actual_url and ".fly.dev" in actual_url
    is_fly_io_internal = actual_url and (".internal" in actual_url or ".flycast" in actual_url)

    if is_fly_io_internal:
        logger.info("Using Flycast - will route to nearest Qdrant machine")
    elif is_fly_io_public:
        logger.info("Using public HTTPS - routes through Fly.io proxy")

    if is_fly_io_public:
        # Fly.io public URL uses standard HTTPS port (443), not Qdrant's default port (6333)
        # Extract hostname and use explicit host/port to avoid QdrantClient defaulting to 6333
        match = re.match(r"https?://([^:/]+)", actual_url)
        if not match:
            raise ValueError(f"Invalid Fly.io URL: {actual_url}")

        host = match.group(1)
        # Always use HTTPS for Fly.io public URLs (port 80 redirects to 443)
        use_https = True
        port = 443

        logger.info(
            f"Connecting to Fly.io public Qdrant: host={host}, port={port}, "
            f"https={use_https}, http2=True, timeout=120s"
        )

        return QdrantClient(
            host=host,
            port=port,
            api_key=actual_api_key,
            timeout=120,
            https=use_https,
            prefer_grpc=False,  # Use REST API for better compatibility with Fly.io
            http2=True,  # Enable HTTP/2 - critical for Fly.io compatibility
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0,
            ),
        )
    elif is_fly_io_internal:
        # Fly.io internal networking (.internal/.flycast) - use HTTP on the specified port (typically 6333)
        match = re.match(r"https?://([^:/]+)(?::(\d+))?", actual_url)
        if not match:
            raise ValueError(f"Invalid Fly.io internal URL: {actual_url}")

        host = match.group(1)
        port_str = match.group(2)
        port = int(port_str) if port_str else 6333
        # Internal networking uses HTTP, not HTTPS
        use_https = False

        logger.info(
            f"Connecting to Fly.io internal Qdrant: host={host}, port={port}, "
            f"https={use_https}, timeout=60s"
        )

        return QdrantClient(
            host=host,
            port=port,
            api_key=actual_api_key,
            timeout=60,
            https=use_https,
            prefer_grpc=False,
            # Note: HTTP/2 is not used for internal Flycast connections
        )
    else:
        # Local Qdrant instance - use URL directly and let QdrantClient handle defaults
        # Extract port from URL if present, otherwise use default 6333
        match = re.match(r"https?://([^:/]+)(?::(\d+))?", actual_url or "")
        if match:
            host = match.group(1)
            port_str = match.group(2)
            port = int(port_str) if port_str else 6333
            use_https = actual_url.startswith("https://") if actual_url else False
        else:
            # Fallback to URL-based initialization
            logger.info(f"Connecting to local Qdrant: url={actual_url}")
            return QdrantClient(
                url=actual_url,
                api_key=actual_api_key,
                timeout=60,
            )

        logger.info(
            f"Connecting to local Qdrant: host={host}, port={port}, https={use_https}, timeout=60s"
        )

        return QdrantClient(
            host=host,
            port=port,
            api_key=actual_api_key,
            timeout=60,
            https=use_https,
            prefer_grpc=False,
        )


def ensure_collection(cli: QdrantClient, name: str, spec: VectorSpec) -> None:
    # Prefer collection_exists if available, else fall back to get_collection
    # Retry on connection errors (common with Fly.io network)
    max_retries = 3
    retry_delay = 2.0
    exists = False  # Initialize to avoid "possibly unbound" error

    logger.info(f"Checking if collection '{name}' exists (retries={max_retries})")
    for attempt in range(max_retries):
        try:
            try:
                exists = bool(cli.collection_exists(name))
            except AttributeError:
                try:
                    cli.get_collection(name)
                    exists = True
                except Exception:
                    exists = False
            logger.info(f"Collection '{name}' exists: {exists}")
            break  # Success, exit retry loop
        except (ResponseHandlingException, ConnectionError, OSError) as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed checking collection: {e}")
            if attempt < max_retries - 1:
                delay = retry_delay * (attempt + 1)
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)  # Exponential backoff
                continue
            logger.error(f"Failed to check collection after {max_retries} attempts")
            raise  # Re-raise on final attempt

    if not exists:
        logger.info(f"Creating collection '{name}' with size={spec.size}, distance={spec.distance}")
        params = models.VectorParams(
            size=spec.size,
            distance=getattr(models.Distance, spec.distance),
        )
        # Try the modern create_collection; fall back to recreate_collection if needed
        for attempt in range(max_retries):
            try:
                try:
                    cli.create_collection(collection_name=name, vectors_config=params)
                    logger.info(f"Collection '{name}' created successfully")
                except AttributeError:
                    cli.recreate_collection(collection_name=name, vectors_config=params)
                    logger.info(f"Collection '{name}' recreated successfully")
                break  # Success
            except (ResponseHandlingException, ConnectionError, OSError) as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed creating collection: {e}"
                )
                if attempt < max_retries - 1:
                    delay = retry_delay * (attempt + 1)
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                logger.error(f"Failed to create collection after {max_retries} attempts")
                raise


def upsert_points(cli: QdrantClient, collection: str, rows: Iterable[dict[str, Any]]) -> None:
    points: list[PointStruct] = []
    for r in rows:
        vec = r.get("vector")
        if vec is None:
            continue
        pid = r["chunk_id"]
        payload = dict(r)
        payload.pop("vector", None)  # Vector stored separately; payload contains metadata
        points.append(PointStruct(id=pid, vector=vec, payload=payload))

    if points:
        logger.info(f"Upserting {len(points)} points to collection '{collection}'")
        logger.info("💡 To verify which Qdrant machine (LHR) handles this request:")
        logger.info("   Check logs: fly logs --app modern-wisdom-qdrant --region lhr")
        # Retry on connection errors
        max_retries = 3
        retry_delay = 2.0
        for attempt in range(max_retries):
            try:
                cli.upsert(collection_name=collection, points=points, wait=True)
                logger.info(f"✅ Successfully upserted {len(points)} points")
                logger.info("   Check Qdrant logs to see which machine (LHR) received the request")
                break  # Success
            except (ResponseHandlingException, ConnectionError, OSError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed upserting points: {e}")
                if attempt < max_retries - 1:
                    delay = retry_delay * (attempt + 1)
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                logger.error(f"Failed to upsert points after {max_retries} attempts")
                raise
    else:
        logger.warning("No points to upsert (all vectors were None)")


def alias_set_live(cli: QdrantClient, collection: str, alias: str = LIVE_ALIAS) -> None:
    # v1.15.x API uses update_collection_aliases with Create/Delete operations
    # First delete alias (if exists), then create it for atomic flip
    logger.info(f"Setting alias '{alias}' -> '{collection}'")
    ops: list[models.AliasOperations] = [
        models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)),
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
        ),
    ]
    cli.update_collection_aliases(change_aliases_operations=ops)
    logger.info(f"Alias '{alias}' successfully set to '{collection}'")


def clear_collection(cli: QdrantClient, collection: str) -> None:
    """Delete all points from a collection."""
    from qdrant_client.models import Filter

    # Delete all points by using an empty filter (matches everything)
    cli.delete(collection_name=collection, points_selector=models.FilterSelector(filter=Filter()))


def delete_collection(cli: QdrantClient, collection: str) -> None:
    """Delete the entire collection."""
    cli.delete_collection(collection_name=collection)
