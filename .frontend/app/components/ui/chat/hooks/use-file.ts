"use client";

import type { DocumentFile, DocumentFileType } from "@llamaindex/chat-ui";
import { useState } from "react";
import { apiFetch } from "./api-key";

// @llamaindex/chat-ui's useFile (0.0.12) with its upload sent through apiFetch, so it carries the API key: chat-ui's
// own hook calls fetch with fixed headers. Same request body and the same return value, so CustomChatInput is
// otherwise unchanged.

const DOCUMENT_TYPES: Record<string, DocumentFileType> = {
  "text/csv": "csv",
  "application/pdf": "pdf",
  "text/plain": "txt",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
    "docx",
};

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

export function useFile({ uploadAPI }: { uploadAPI: string }) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [files, setFiles] = useState<DocumentFile[]>([]);

  const removeDoc = (file: DocumentFile) =>
    setFiles((current) => current.filter((f) => f.id !== file.id));

  const reset = () => {
    setImageUrl(null);
    setFiles([]);
  };

  // An image stays in the page (sent with the next message); a document is uploaded to the API.
  const uploadFile = async (
    file: File,
    requestParams: Record<string, unknown> = {},
  ) => {
    if (file.type.startsWith("image/")) {
      setImageUrl(await readAsDataUrl(file));
      return;
    }
    if (!DOCUMENT_TYPES[file.type])
      throw new Error("Unsupported document type.");
    const response = await apiFetch(uploadAPI, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...requestParams,
        base64: await readAsDataUrl(file),
        name: file.name,
      }),
    });
    if (!response.ok) throw new Error("Failed to upload document.");
    const uploaded = (await response.json()) as DocumentFile;
    setFiles((current) =>
      current.some((f) => f.id === uploaded.id)
        ? current
        : [...current, uploaded],
    );
  };

  const getAnnotations = () => [
    ...(imageUrl ? [{ type: "image", data: { url: imageUrl } }] : []),
    ...(files.length ? [{ type: "document_file", data: { files } }] : []),
  ];

  return {
    imageUrl,
    setImageUrl,
    files,
    removeDoc,
    reset,
    getAnnotations,
    uploadFile,
  };
}
