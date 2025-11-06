#!/usr/bin/env python
"""
Audit data consistency between Qdrant (vector database) and DuckDB (metadata database).

This script identifies episodes that exist in one system but not the other,
helping to diagnose data synchronization issues.
"""

from __future__ import annotations

import duckdb
from qdrant_client import QdrantClient

from modern_wisdom_rag_pipeline import paths, settings


def get_qdrant_episode_ids() -> set[str]:
    """Get all unique episode IDs from Qdrant collection."""
    client = QdrantClient(url=settings.QDRANT_URL)

    collection_name = paths.INDEX_VERSION
    print(f"📊 Scanning Qdrant collection: {collection_name}")

    try:
        # Get collection info
        collection_info = client.get_collection(collection_name)
        total_points = collection_info.points_count
        print(f"   Total points in collection: {total_points:,}")

        # Scroll through all points to get episode IDs
        episode_ids = set()
        offset = None
        batch_size = 1000
        processed = 0

        while True:
            result = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            points, next_offset = result

            if not points:
                break

            for point in points:
                if point.payload and "episode_id" in point.payload:
                    episode_ids.add(point.payload["episode_id"])

            processed += len(points)
            print(f"   Processed {processed:,}/{total_points:,} points...", end="\r")

            if next_offset is None:
                break
            offset = next_offset

        print(f"\n   Found {len(episode_ids)} unique episode IDs in Qdrant")
        return episode_ids

    except Exception as e:
        print(f"   ❌ Error accessing Qdrant: {e}")
        return set()


def get_duckdb_episode_ids() -> dict[str, dict]:
    """Get all episode IDs from DuckDB with their metadata."""
    print(f"\n📊 Scanning DuckDB: {paths.DUCKDB_PATH}")

    try:
        con = duckdb.connect(paths.DUCKDB_PATH.as_posix(), read_only=True)

        # Find the episodes table
        candidate_schemas = ["mw", "mw_staging", "main"]
        table_name = None

        for sch in candidate_schemas:
            row = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema=? AND table_name='episodes'",
                [sch],
            ).fetchone()
            if row and row[0]:
                table_name = f"{sch}.episodes"
                break

        if not table_name:
            print("   ❌ No episodes table found")
            return {}

        print(f"   Using table: {table_name}")

        # Get all episodes with key metadata
        sql = f"""
            SELECT id, title, guest, audio_url, publish_date
            FROM {table_name}
        """

        rows = con.execute(sql).fetchall()

        episodes = {}
        for row in rows:
            episodes[row[0]] = {
                "title": row[1],
                "guest": row[2],
                "audio_url": row[3],
                "publish_date": row[4],
            }

        print(f"   Found {len(episodes)} episodes in DuckDB")
        con.close()
        return episodes

    except Exception as e:
        print(f"   ❌ Error accessing DuckDB: {e}")
        return {}


def audit_consistency():
    """Run the data consistency audit."""
    print("=" * 70)
    print("🔍 DATA CONSISTENCY AUDIT: Qdrant ↔ DuckDB")
    print("=" * 70)

    # Get data from both sources
    qdrant_ids = get_qdrant_episode_ids()
    duckdb_episodes = get_duckdb_episode_ids()
    duckdb_ids = set(duckdb_episodes.keys())

    # Calculate differences
    in_qdrant_not_duckdb = qdrant_ids - duckdb_ids
    in_duckdb_not_qdrant = duckdb_ids - qdrant_ids
    in_both = qdrant_ids & duckdb_ids

    # Print summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"✓ Episodes in both systems:          {len(in_both):,}")
    print(f"⚠️  Episodes in Qdrant only:          {len(in_qdrant_not_duckdb):,}")
    print(f"⚠️  Episodes in DuckDB only:          {len(in_duckdb_not_qdrant):,}")
    print(f"📈 Total unique episodes:            {len(qdrant_ids | duckdb_ids):,}")

    if in_both:
        coverage = (len(in_both) / len(qdrant_ids | duckdb_ids)) * 100
        print(f"✨ Data consistency:                {coverage:.1f}%")

    # Detail: Episodes in Qdrant but not DuckDB
    if in_qdrant_not_duckdb:
        print("\n" + "=" * 70)
        print("⚠️  EPISODES IN QDRANT BUT NOT IN DUCKDB")
        print("=" * 70)
        print("These episodes have embeddings but no metadata.")
        print("Users will see shortened IDs instead of titles.\n")

        for i, ep_id in enumerate(sorted(in_qdrant_not_duckdb)[:10], 1):
            print(f"  {i}. {ep_id}")

        if len(in_qdrant_not_duckdb) > 10:
            print(f"  ... and {len(in_qdrant_not_duckdb) - 10} more")

    # Detail: Episodes in DuckDB but not Qdrant
    if in_duckdb_not_qdrant:
        print("\n" + "=" * 70)
        print("⚠️  EPISODES IN DUCKDB BUT NOT IN QDRANT")
        print("=" * 70)
        print("These episodes have metadata but no embeddings.")
        print("They won't appear in search results.\n")

        for i, ep_id in enumerate(sorted(in_duckdb_not_qdrant)[:10], 1):
            meta = duckdb_episodes[ep_id]
            title = meta.get("title") or "[No Title]"
            print(f"  {i}. {title[:60]}")
            print(f"      ID: {ep_id}")

        if len(in_duckdb_not_qdrant) > 10:
            print(f"  ... and {len(in_duckdb_not_qdrant) - 10} more")

    # Check for episodes with missing audio URLs
    print("\n" + "=" * 70)
    print("🎧 EPISODES WITH MISSING AUDIO URLs")
    print("=" * 70)

    missing_audio = [
        (ep_id, meta) for ep_id, meta in duckdb_episodes.items() if not meta.get("audio_url")
    ]

    if missing_audio:
        print(f"Found {len(missing_audio)} episodes without audio URLs:\n")
        for i, (ep_id, meta) in enumerate(missing_audio[:10], 1):
            title = meta.get("title") or "[No Title]"
            in_qdrant = "✓" if ep_id in qdrant_ids else "✗"
            print(f"  {i}. {title[:60]} (Qdrant: {in_qdrant})")

        if len(missing_audio) > 10:
            print(f"  ... and {len(missing_audio) - 10} more")
    else:
        print("✓ All episodes have audio URLs")

    # Recommendations
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)

    if in_qdrant_not_duckdb:
        print("\n⚠️  Episodes in Qdrant but not DuckDB:")
        print("   → Re-run metadata ingestion pipeline")
        print("   → Or clean up Qdrant index to remove orphaned embeddings")

    if in_duckdb_not_qdrant:
        print("\n⚠️  Episodes in DuckDB but not Qdrant:")
        print("   → Re-run embedding/indexing pipeline")
        print("   → Check if these episodes have transcripts available")

    if missing_audio:
        print("\n⚠️  Episodes without audio URLs:")
        print("   → Check RSS feed for updated audio URLs")
        print("   → Update episodes table with correct URLs")

    if not in_qdrant_not_duckdb and not in_duckdb_not_qdrant:
        print("\n✓ No action needed - data is consistent!")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    audit_consistency()
