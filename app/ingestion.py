"""Document ingestion utilities for the research assistant."""
from __future__ import annotations

import io
import importlib
from pathlib import Path
from typing import Dict, Iterable, List

from . import config
from .models import DocumentChunk
from .text_extraction import extract_text, chunk_text, UnsupportedDocumentTypeError


class GoogleDriveIngestionError(RuntimeError):
    """Raised when Google Drive ingestion fails."""


def _build_drive_service():
    """Instantiate a Google Drive service client using a service account.

    This function relies on ``googleapiclient`` and ``google.oauth2`` packages. Ensure
    they are installed and that ``config.GOOGLE_SERVICE_ACCOUNT_FILE`` points to a valid
    service account JSON credential file with access to the target Drive resources.
    """

    discovery = importlib.import_module("googleapiclient.discovery")
    service_account = importlib.import_module("google.oauth2.service_account")

    credentials = service_account.Credentials.from_service_account_file(
        str(config.GOOGLE_SERVICE_ACCOUNT_FILE), scopes=config.GOOGLE_DRIVE_SCOPES
    )
    return discovery.build("drive", "v3", credentials=credentials)


def _download_drive_file(service, file_info: Dict[str, str], destination: Path) -> Path:
    """Download a single file from Google Drive to ``destination``."""
    http = importlib.import_module("googleapiclient.http")

    destination.mkdir(parents=True, exist_ok=True)
    file_id = file_info["id"]
    file_name = file_info["name"]
    mime_type = file_info.get("mimeType", "")

    request = None
    if mime_type.startswith("application/vnd.google-apps"):
        # Export native Google Docs formats to PDF by default.
        export_mime_type = "application/pdf"
        request = service.files().export_media(fileId=file_id, mimeType=export_mime_type)
        file_name = f"{file_name}.pdf" if not file_name.lower().endswith(".pdf") else file_name
    else:
        request = service.files().get_media(fileId=file_id)

    buffer = io.BytesIO()
    downloader = http.MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    output_path = destination / file_name
    with output_path.open("wb") as f:
        f.write(buffer.getvalue())
    return output_path


def download_google_drive_folder(folder_id: str, destination: Path) -> List[Path]:
    """Download all files within a Google Drive folder recursively.

    Parameters
    ----------
    folder_id:
        Identifier of the Google Drive folder to download.
    destination:
        Local directory where files should be saved.
    """
    service = _build_drive_service()
    downloaded: List[Path] = []

    def _walk_folder(current_folder_id: str, current_destination: Path) -> None:
        query = f"'{current_folder_id}' in parents and trashed = false"
        page_token = None
        while True:
            response = (
                service.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token,
                )
                .execute()
            )
            for file_info in response.get("files", []):
                mime_type = file_info.get("mimeType", "")
                if mime_type == "application/vnd.google-apps.folder":
                    sub_destination = current_destination / file_info["name"]
                    _walk_folder(file_info["id"], sub_destination)
                else:
                    path = _download_drive_file(service, file_info, current_destination)
                    downloaded.append(path)
            page_token = response.get("nextPageToken")
            if not page_token:
                break

    try:
        _walk_folder(folder_id, destination)
    except Exception as exc:  # noqa: BLE001 - propagate as custom error
        raise GoogleDriveIngestionError("Failed to download Google Drive folder") from exc

    return downloaded


def build_document_chunks(files: Iterable[Path]) -> List[DocumentChunk]:
    """Extract and chunk text content from the provided files."""
    chunks: List[DocumentChunk] = []
    for path in files:
        try:
            text = extract_text(path)
        except UnsupportedDocumentTypeError:
            continue
        except FileNotFoundError:
            continue
        document_chunks = chunk_text(text)
        for index, content in enumerate(document_chunks):
            metadata = {
                "source_path": str(path),
                "document_name": path.name,
                "chunk_index": index,
            }
            chunks.append(DocumentChunk(content=content, metadata=metadata))
    return chunks


def ingest_from_google_drive(folder_id: str) -> List[DocumentChunk]:
    """Download a Google Drive folder and return document chunks."""
    destination = config.RAW_DATA_DIR / folder_id
    files = download_google_drive_folder(folder_id, destination)
    return build_document_chunks(files)


def ingest_from_notion_placeholder(*_: str) -> List[DocumentChunk]:
    """Placeholder for future Notion ingestion support."""
    return []


def ingest_from_canva_placeholder(*_: str) -> List[DocumentChunk]:
    """Placeholder for future Canva ingestion support."""
    return []
