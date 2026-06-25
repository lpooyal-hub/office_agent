async function parseJson(response) {
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.detail || "서버 요청 처리 중 오류가 발생했습니다.");
    }
    return data;
}

function appendFiles(formData, files) {
    files.forEach((file) => formData.append("files", file));
}

export async function listDocuments() {
    const response = await fetch("/documents");
    return parseJson(response);
}

export async function uploadDocuments(accessCode, files) {
    const formData = new FormData();
    formData.append("access_code", accessCode);
    appendFiles(formData, files);

    const response = await fetch("/documents", {
        method: "POST",
        body: formData,
    });
    return parseJson(response);
}

export async function deleteDocument(storedName, accessCode) {
    const response = await fetch(
        `/documents/${encodeURIComponent(storedName)}?access_code=${encodeURIComponent(accessCode)}`,
        { method: "DELETE" },
    );
    return parseJson(response);
}

export async function clearDocuments(accessCode) {
    const response = await fetch(`/documents?access_code=${encodeURIComponent(accessCode)}`, {
        method: "DELETE",
    });
    return parseJson(response);
}

export async function summarizeDocuments(accessCode, files) {
    const formData = new FormData();
    formData.append("access_code", accessCode);
    appendFiles(formData, files);

    const response = await fetch("/summarize", {
        method: "POST",
        body: formData,
    });
    return parseJson(response);
}

export async function askPolicyBot(accessCode, question) {
    const formData = new FormData();
    formData.append("access_code", accessCode);
    formData.append("question", question);

    const response = await fetch("/chat", {
        method: "POST",
        body: formData,
    });
    return parseJson(response);
}

export async function processAudio(accessCode, file, sttMode) {
    const formData = new FormData();
    formData.append("access_code", accessCode);
    formData.append("audio_file", file);
    formData.append("stt_mode", sttMode);

    const response = await fetch("/process", {
        method: "POST",
        body: formData,
    });
    return parseJson(response);
}
