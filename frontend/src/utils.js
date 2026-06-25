import React, { useEffect, useMemo, useRef, useState } from "react";
import htm from "htm";

export { useEffect, useMemo, useRef, useState };
export const html = htm.bind(React.createElement);

export function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

export function formatDate(isoText) {
    const date = new Date(isoText);
    if (Number.isNaN(date.getTime())) {
        return "-";
    }

    return date.toLocaleString("ko-KR", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
    });
}

export function formatBytes(bytes) {
    if (!bytes) return "0 MB";
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export async function copyText(text) {
    await navigator.clipboard.writeText(text ?? "");
}

export function downloadTextFile(filename, content) {
    const blob = new Blob([content ?? ""], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
}

export function getFilesFromInput(event) {
    return Array.from(event?.target?.files || []);
}
