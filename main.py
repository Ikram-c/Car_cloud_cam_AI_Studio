"""Main entry point and GCS storage client helpers."""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from frame_extract.cloud import (  # noqa: E402
    MockBlob,
    MockBucket,
    MockStorageClient,
    download_gpx_track,
    get_storage_client,
)

__all__ = [
    "MockBlob",
    "MockBucket",
    "MockStorageClient",
    "download_gpx_track",
    "get_storage_client",
]

if __name__ == "__main__":
    import uvicorn
    from app import app

    port = int(os.environ.get("PORT", 3000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)