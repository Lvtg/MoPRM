from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

from huggingface_hub import hf_hub_url
from huggingface_hub.file_download import get_hf_file_metadata


DEFAULT_REPO_ID = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
DEFAULT_REVISION = "98d69606595eedbdbbbf0a7d28efdcd462ba6a67"
DEFAULT_FILENAME = "pytorch_model.bin"


def human_bytes(size: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def sha256_file(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def repo_cache_dir(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / f"models--{repo_id.replace('/', '--')}"


def adopt_largest_incomplete(blob_dir: Path, etag: str, part_path: Path) -> None:
    candidates = [
        path
        for path in blob_dir.glob(f"{etag}.*.incomplete")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not candidates:
        return

    largest = max(candidates, key=lambda path: path.stat().st_size)
    largest_size = largest.stat().st_size
    current_size = part_path.stat().st_size if part_path.exists() else 0
    if largest_size <= current_size:
        return

    shutil.copy2(largest, part_path)
    print(
        f"Adopted existing partial {largest.name}: {human_bytes(largest_size)}",
        flush=True,
    )


def link_snapshot_file(blob_path: Path, snapshot_path: Path) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        if snapshot_path.stat().st_size == blob_path.stat().st_size:
            return
        snapshot_path.unlink()

    try:
        os.link(blob_path, snapshot_path)
    except OSError:
        shutil.copy2(blob_path, snapshot_path)


def maybe_finalize(
    *,
    part_path: Path,
    blob_path: Path,
    snapshot_path: Path,
    expected_size: int,
    etag: str,
    chunk_size: int,
) -> bool:
    if blob_path.exists() and blob_path.stat().st_size == expected_size:
        if re.fullmatch(r"[0-9a-f]{64}", etag):
            digest = sha256_file(blob_path, chunk_size)
            if digest != etag:
                raise RuntimeError(
                    f"Cached blob hash mismatch: expected {etag}, got {digest}"
                )
        link_snapshot_file(blob_path, snapshot_path)
        print(f"Weight already cached at {blob_path}", flush=True)
        return True

    if not part_path.exists() or part_path.stat().st_size != expected_size:
        return False

    if re.fullmatch(r"[0-9a-f]{64}", etag):
        print("Verifying SHA256...", flush=True)
        digest = sha256_file(part_path, chunk_size)
        if digest != etag:
            raise RuntimeError(f"Downloaded hash mismatch: expected {etag}, got {digest}")

    part_path.replace(blob_path)
    link_snapshot_file(blob_path, snapshot_path)
    print(f"Cached {blob_path}", flush=True)
    print(f"Snapshot file ready at {snapshot_path}", flush=True)
    return True


def download_attempt(
    *,
    url: str,
    part_path: Path,
    expected_size: int,
    timeout: float,
    chunk_size: int,
    progress_interval: float,
) -> None:
    resume_at = part_path.stat().st_size if part_path.exists() else 0
    if resume_at > expected_size:
        raise RuntimeError(
            f"Partial file is larger than expected: {resume_at} > {expected_size}"
        )

    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "identity",
        "User-Agent": "moprm-skywork-downloader/1.0",
    }
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"

    request = urllib.request.Request(url, headers=headers)
    print(
        f"Starting HTTP download at {human_bytes(resume_at)} / "
        f"{human_bytes(expected_size)}",
        flush=True,
    )

    started = time.monotonic()
    last_report = started
    start_size = resume_at
    mode = "ab" if resume_at else "wb"
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.getcode()
        if resume_at and status != 206:
            raise RuntimeError(f"Server did not honor Range request; status={status}")
        if not resume_at and status not in (200, 206):
            raise RuntimeError(f"Unexpected HTTP status={status}")

        with part_path.open(mode) as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)

                now = time.monotonic()
                if now - last_report >= progress_interval:
                    current_size = handle.tell()
                    elapsed = max(now - started, 1e-6)
                    speed = (current_size - start_size) / elapsed
                    percent = current_size / expected_size * 100
                    print(
                        f"{human_bytes(current_size)} / {human_bytes(expected_size)} "
                        f"({percent:.2f}%), this run {human_bytes(speed)}/s",
                        flush=True,
                    )
                    last_report = now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume-download the Skywork PRM weight into the local HF cache."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--cache-dir", default="models/hf_cache")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--chunk-size-mib", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--progress-interval", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    storage_dir = repo_cache_dir(cache_dir, args.repo_id)
    blob_dir = storage_dir / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=True)

    url = hf_hub_url(args.repo_id, args.filename, revision=args.revision)
    metadata = get_hf_file_metadata(url, timeout=args.timeout)
    if metadata.size is None:
        raise RuntimeError("Hub metadata did not include the expected file size.")
    if metadata.etag is None:
        raise RuntimeError("Hub metadata did not include the expected etag.")

    commit_hash = metadata.commit_hash or args.revision
    part_path = blob_dir / f"{metadata.etag}.part"
    blob_path = blob_dir / metadata.etag
    snapshot_path = storage_dir / "snapshots" / commit_hash / args.filename

    print(f"repo={args.repo_id}", flush=True)
    print(f"revision={commit_hash}", flush=True)
    print(f"etag={metadata.etag}", flush=True)
    print(f"expected_size={metadata.size} ({human_bytes(metadata.size)})", flush=True)
    print(f"part_path={part_path}", flush=True)

    chunk_size = args.chunk_size_mib * 1024 * 1024
    adopt_largest_incomplete(blob_dir, metadata.etag, part_path)

    if maybe_finalize(
        part_path=part_path,
        blob_path=blob_path,
        snapshot_path=snapshot_path,
        expected_size=metadata.size,
        etag=metadata.etag,
        chunk_size=chunk_size,
    ):
        return 0

    for attempt in range(1, args.max_attempts + 1):
        print(f"Attempt {attempt}/{args.max_attempts}", flush=True)
        try:
            fresh_metadata = get_hf_file_metadata(url, timeout=args.timeout)
            download_url = fresh_metadata.location
            download_attempt(
                url=download_url,
                part_path=part_path,
                expected_size=metadata.size,
                timeout=args.timeout,
                chunk_size=chunk_size,
                progress_interval=args.progress_interval,
            )
            if maybe_finalize(
                part_path=part_path,
                blob_path=blob_path,
                snapshot_path=snapshot_path,
                expected_size=metadata.size,
                etag=metadata.etag,
                chunk_size=chunk_size,
            ):
                return 0
        except KeyboardInterrupt:
            raise
        except (OSError, TimeoutError, urllib.error.URLError, RuntimeError) as exc:
            current_size = part_path.stat().st_size if part_path.exists() else 0
            print(
                f"Attempt failed after {human_bytes(current_size)}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < args.max_attempts:
                time.sleep(args.retry_sleep)

    current_size = part_path.stat().st_size if part_path.exists() else 0
    print(
        f"Stopped after {args.max_attempts} attempts at "
        f"{human_bytes(current_size)} / {human_bytes(metadata.size)}.",
        file=sys.stderr,
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
