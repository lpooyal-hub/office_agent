function getAccessCode() {
    const accessCode = document.getElementById("accessCode").value.trim();
    if (!accessCode) {
        alert("접속 코드를 입력해주세요.");
        return "";
    }
    return accessCode;
}

function setLoading(id, visible) {
    document.getElementById(id).style.display = visible ? "block" : "none";
}

function setAlert(message = "") {
    const alertBox = document.getElementById("globalAlert");
    if (!message) {
        alertBox.style.display = "none";
        alertBox.innerText = "";
        return;
    }

    alertBox.innerText = message;
    alertBox.style.display = "block";
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function formatDate(isoText) {
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

function formatBytes(bytes) {
    if (!bytes) return "0 MB";
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function copyText(text) {
    navigator.clipboard.writeText(text).then(() => {
        setAlert("클립보드에 복사했습니다.");
        window.setTimeout(() => setAlert(""), 1400);
    }).catch(() => {
        setAlert("복사에 실패했습니다. 브라우저 권한을 확인해주세요.");
    });
}

function copyEncodedText(encodedText) {
    copyText(decodeURIComponent(encodedText));
}

function downloadTextFile(filename, content) {
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
}

function downloadEncodedTextFile(filename, encodedText) {
    downloadTextFile(filename, decodeURIComponent(encodedText));
}

function getIndexStatusLabel(status) {
    const labels = {
        pending: "색인 대기",
        indexed: "색인 완료",
        failed: "색인 실패",
    };
    return labels[status] || "상태 미확인";
}

function renderDocumentLibrary(documents) {
    const list = document.getElementById("docLibraryList");
    const empty = document.getElementById("docLibraryEmpty");
    document.getElementById("docCount").innerText = `${documents.length}개`;

    if (!documents.length) {
        list.innerHTML = "";
        empty.style.display = "block";
        return;
    }

    empty.style.display = "none";
    list.innerHTML = documents.map((doc) => {
        const tags = (doc.tags || []).map((tag) => `<span class="doc-tag">${escapeHtml(tag)}</span>`).join("");
        const owner = doc.owner ? ` · 소유자: ${escapeHtml(doc.owner)}` : "";
        const indexedAt = doc.indexed_at ? ` · 색인: ${formatDate(doc.indexed_at)}` : "";
        return `
        <div class="doc-item">
            <div class="doc-header">
                <div>
                    <div class="doc-title-row">
                        <div class="doc-name">${escapeHtml(doc.display_name)}</div>
                        <span class="doc-status ${escapeHtml(doc.index_status || "unknown")}">${getIndexStatusLabel(doc.index_status)}</span>
                    </div>
                    <div class="doc-sub">${formatBytes(doc.size_bytes)} · ${escapeHtml(doc.content_type || "-")} · 업로드: ${formatDate(doc.uploaded_at)}${indexedAt}${owner}</div>
                </div>
                <button class="btn secondary" onclick="deleteDocument('${encodeURIComponent(doc.stored_name)}')">삭제</button>
            </div>
            ${doc.description ? `<div class="doc-description">${escapeHtml(doc.description)}</div>` : ""}
            ${tags ? `<div class="doc-tags">${tags}</div>` : ""}
        </div>
    `;
    }).join("");
}

async function loadDocumentLibrary() {
    try {
        const response = await fetch("/documents");
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "문서 목록을 불러오지 못했습니다.");
        renderDocumentLibrary(data.documents || []);
    } catch (error) {
        console.error(error);
        setAlert(`문서 목록 조회 실패\n${error.message}`);
    }
}

async function uploadDocuments() {
    const fileInput = document.getElementById("docInput");
    const btn = document.getElementById("docBtn");
    const resultBox = document.getElementById("docResult");

    if (!fileInput.files.length) return alert("업로드할 문서를 선택해주세요!");
    const accessCode = getAccessCode();
    if (!accessCode) return;

    const formData = new FormData();
    formData.append("access_code", accessCode);
    formData.append("tags", document.getElementById("docTags").value.trim());
    formData.append("owner", document.getElementById("docOwner").value.trim());
    formData.append("description", document.getElementById("docDescription").value.trim());
    Array.from(fileInput.files).forEach((file) => formData.append("files", file));

    setAlert("");
    setLoading("docLoading", true);
    resultBox.style.display = "none";
    btn.disabled = true;

    try {
        const response = await fetch("/documents", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "문서 업로드 서버 응답 오류");

        const names = (data.files || []).map((file) => file.display_name).join(", ");
        resultBox.innerText = `업로드 완료\n${names}\n현재 문서 수: ${data.count}개`;
        resultBox.style.display = "block";
        fileInput.value = "";
        document.getElementById("docTags").value = "";
        document.getElementById("docOwner").value = "";
        document.getElementById("docDescription").value = "";
        renderDocumentLibrary(data.documents || []);
    } catch (error) {
        console.error(error);
        setAlert(`문서 업로드 중 오류가 발생했습니다.\n${error.message}`);
    } finally {
        setLoading("docLoading", false);
        btn.disabled = false;
    }
}

async function deleteDocument(encodedStoredName) {
    const accessCode = getAccessCode();
    if (!accessCode) return;
    if (!window.confirm("이 문서를 문서 보관함에서 삭제할까요?")) return;

    try {
        const response = await fetch(`/documents/${encodedStoredName}?access_code=${encodeURIComponent(accessCode)}`, {
            method: "DELETE",
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "문서 삭제 실패");
        renderDocumentLibrary(data.documents || []);
        setAlert(data.message || "문서를 삭제했습니다.");
    } catch (error) {
        console.error(error);
        setAlert(`문서 삭제 중 오류가 발생했습니다.\n${error.message}`);
    }
}

async function clearDocuments() {
    const accessCode = getAccessCode();
    if (!accessCode) return;
    if (!window.confirm("문서 보관함을 모두 비울까요?")) return;

    try {
        const response = await fetch(`/documents?access_code=${encodeURIComponent(accessCode)}`, {
            method: "DELETE",
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "문서 초기화 실패");
        renderDocumentLibrary([]);
        setAlert(data.message || "문서 보관함을 비웠습니다.");
    } catch (error) {
        console.error(error);
        setAlert(`문서 초기화 중 오류가 발생했습니다.\n${error.message}`);
    }
}

function renderSummaryResult(data) {
    const resultArea = document.getElementById("summaryCards");
    const wrapper = document.getElementById("summaryResultBlock");
    const combined = data.combined_summary || "";

    resultArea.innerHTML = (data.summaries || []).map((item, index) => {
        const encodedSummary = encodeURIComponent(item.summary || "");
        const safeFilename = (item.filename || "summary").replace(/[\\/:*?"<>|]+/g, "_");
        return `
        <div class="summary-card">
            <div class="summary-header">
                <div class="summary-title">${index + 1}. ${escapeHtml(item.filename)}</div>
                <div class="inline-actions">
                    <button class="btn secondary" onclick="copyEncodedText('${encodedSummary}')">복사</button>
                    <button class="btn secondary" onclick="downloadEncodedTextFile('${escapeHtml(`${safeFilename}.summary.txt`)}', '${encodedSummary}')">저장</button>
                </div>
            </div>
            <div class="summary-content">${escapeHtml(item.summary)}</div>
        </div>
    `;
    }).join("");

    document.getElementById("summaryCombined").innerText = combined;
    wrapper.style.display = "block";
}

async function summarizeDocuments() {
    const fileInput = document.getElementById("summaryInput");
    const btn = document.getElementById("summaryBtn");

    if (!fileInput.files.length) return alert("요약할 문서를 선택해주세요!");
    const accessCode = getAccessCode();
    if (!accessCode) return;

    const formData = new FormData();
    formData.append("access_code", accessCode);
    formData.append("tags", document.getElementById("docTags").value.trim());
    formData.append("owner", document.getElementById("docOwner").value.trim());
    formData.append("description", document.getElementById("docDescription").value.trim());
    Array.from(fileInput.files).forEach((file) => formData.append("files", file));

    setAlert("");
    setLoading("summaryLoading", true);
    document.getElementById("summaryResultBlock").style.display = "none";
    btn.disabled = true;

    try {
        const response = await fetch("/summarize", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "문서 요약 서버 응답 오류");
        renderSummaryResult(data);
        fileInput.value = "";
    } catch (error) {
        console.error(error);
        setAlert(`문서 요약 중 오류가 발생했습니다.\n${error.message}`);
    } finally {
        setLoading("summaryLoading", false);
        btn.disabled = false;
    }
}

function renderSources(elementId, sources) {
    const container = document.getElementById(elementId);
    if (!sources || !sources.length) {
        container.innerHTML = '<div class="empty" style="display:block;">참고 문서를 찾지 못했습니다.</div>';
        return;
    }

    container.innerHTML = sources.map((source) => `
        <div class="source-card">
            <div class="source-title">${escapeHtml(source.source)}</div>
            <div class="source-meta">유사도 ${source.score.toFixed(2)}</div>
            <div class="source-excerpt">${escapeHtml(source.excerpt)}</div>
        </div>
    `).join("");
}

async function askPolicyBot() {
    const questionInput = document.getElementById("chatQuestion");
    const btn = document.getElementById("chatBtn");
    const resultArea = document.getElementById("chatResult");

    if (!questionInput.value.trim()) return alert("질문을 입력해주세요!");
    const accessCode = getAccessCode();
    if (!accessCode) return;

    const formData = new FormData();
    formData.append("question", questionInput.value.trim());
    formData.append("access_code", accessCode);

    setAlert("");
    setLoading("chatLoading", true);
    resultArea.style.display = "none";
    btn.disabled = true;

    try {
        const response = await fetch("/chat", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "챗봇 서버 응답 오류");

        document.getElementById("chatAnswer").innerText = data.answer || "답변을 생성하지 못했습니다.";
        document.getElementById("chatContext").innerText = data.retrieved_rule || "참고 문서를 찾지 못했습니다.";
        renderSources("chatSources", data.sources || []);
        resultArea.style.display = "block";
    } catch (error) {
        console.error(error);
        setAlert(`질문 처리 중 오류가 발생했습니다.\n${error.message}`);
    } finally {
        setLoading("chatLoading", false);
        btn.disabled = false;
    }
}

async function startProcess() {
    const fileInput = document.getElementById("audioInput");
    const btn = document.getElementById("submitBtn");
    const resultArea = document.getElementById("resultArea");
    const sttMode = document.querySelector('input[name="sttMode"]:checked').value;

    if (!fileInput.files[0]) return alert("파일을 선택해주세요!");
    const accessCode = getAccessCode();
    if (!accessCode) return;

    const formData = new FormData();
    formData.append("audio_file", fileInput.files[0]);
    formData.append("stt_mode", sttMode);
    formData.append("access_code", accessCode);

    setAlert("");
    setLoading("loading", true);
    resultArea.style.display = "none";
    btn.disabled = true;

    try {
        const response = await fetch("/process", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "서버 응답 오류");

        document.getElementById("retrievedRule").innerText = data.retrieved_rule || "관련 내규를 찾지 못했습니다.";
        document.getElementById("summary").innerText = data.summary || "요약 결과가 없습니다.";
        document.getElementById("script").innerText = data.script || "인식된 스크립트가 없습니다.";
        renderSources("audioSources", data.sources || []);
        resultArea.style.display = "block";
    } catch (error) {
        console.error(error);
        setAlert(`처리 중 오류가 발생했습니다.\n${error.message}`);
    } finally {
        setLoading("loading", false);
        btn.disabled = false;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    loadDocumentLibrary();
});
