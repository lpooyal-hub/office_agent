import ReactDOM from "react-dom/client";

import {
    askPolicyBot,
    clearDocuments,
    deleteDocument,
    listDocuments,
    processAudio,
    summarizeDocuments,
    uploadDocuments,
} from "./api.js";
import {
    AlertBanner,
    AudioSection,
    ChatSection,
    DocumentLibrarySection,
    HeroSection,
    SummarySection,
} from "./components.js";
import { getFilesFromInput, html, useEffect, useMemo, useRef, useState } from "./utils.js";
import "./styles.css";

function App() {
    const initialPage = window.location.hash === "#overview" ? "overview" : "workspace";
    const [accessCode, setAccessCode] = useState("");
    const [alert, setAlert] = useState("");
    const [activePage, setActivePage] = useState(initialPage);
    const [activeWorkspace, setActiveWorkspace] = useState("documents");
    const [documents, setDocuments] = useState([]);
    const [docFiles, setDocFiles] = useState([]);
    const [summaryFiles, setSummaryFiles] = useState([]);
    const [audioFiles, setAudioFiles] = useState([]);
    const [chatQuestion, setChatQuestion] = useState("");
    const [sttMode, setSttMode] = useState("local");
    const [summaryResult, setSummaryResult] = useState(null);
    const [audioResult, setAudioResult] = useState(null);
    const [chatResult, setChatResult] = useState(null);
    const [docInputKey, setDocInputKey] = useState(0);
    const [summaryInputKey, setSummaryInputKey] = useState(0);
    const [audioInputKey, setAudioInputKey] = useState(0);
    const [docLoading, setDocLoading] = useState(false);
    const [summaryLoading, setSummaryLoading] = useState(false);
    const [audioLoading, setAudioLoading] = useState(false);
    const [chatLoading, setChatLoading] = useState(false);

    const accessCodeTimerRef = useRef(null);

    const documentCount = useMemo(() => documents.length, [documents]);
    const workspaceItems = useMemo(() => ([
        {
            key: "documents",
            label: "문서 보관함",
            description: "업로드한 문서를 지식 자산으로 관리합니다.",
        },
        {
            key: "summary",
            label: "문서 요약",
            description: "여러 문서를 읽고 핵심 내용을 빠르게 정리합니다.",
        },
        {
            key: "meeting",
            label: "회의 정리",
            description: "음성 파일을 요약하고 후속 작업까지 정리합니다.",
        },
        {
            key: "chat",
            label: "사내 규정 챗봇",
            description: "문서 기반으로 답변과 근거를 함께 확인합니다.",
        },
    ]), []);

    const activeWorkspaceMeta = useMemo(
        () => workspaceItems.find((item) => item.key === activeWorkspace) || workspaceItems[0],
        [activeWorkspace, workspaceItems],
    );

    function showAlert(message, timeoutMs = 0) {
        setAlert(message);
        if (accessCodeTimerRef.current) {
            window.clearTimeout(accessCodeTimerRef.current);
        }
        if (timeoutMs > 0) {
            accessCodeTimerRef.current = window.setTimeout(() => setAlert(""), timeoutMs);
        }
    }

    function requireAccessCode() {
        const trimmed = accessCode.trim();
        if (!trimmed) {
            throw new Error("접속 코드를 입력해주세요.");
        }
        return trimmed;
    }

    async function refreshDocuments() {
        const data = await listDocuments();
        setDocuments(data.documents || []);
    }

    useEffect(() => {
        refreshDocuments().catch((error) => {
            console.error(error);
            showAlert(`문서 목록 조회 실패\n${error.message}`);
        });
    }, []);

    useEffect(() => {
        const nextHash = activePage === "overview" ? "#overview" : "#workspace";
        if (window.location.hash !== nextHash) {
            window.history.replaceState(null, "", nextHash);
        }
    }, [activePage]);

    async function handleDocumentUpload() {
        if (!docFiles.length) {
            showAlert("업로드할 문서를 선택해주세요.", 1800);
            return;
        }

        try {
            const code = requireAccessCode();
            setDocLoading(true);
            setAlert("");
            const data = await uploadDocuments(code, docFiles);
            setDocuments(data.documents || []);
            setDocFiles([]);
            setDocInputKey((value) => value + 1);
            showAlert(data.message || "문서 업로드가 완료되었습니다.", 1800);
        } catch (error) {
            console.error(error);
            showAlert(`문서 업로드 중 오류가 발생했습니다.\n${error.message}`);
        } finally {
            setDocLoading(false);
        }
    }

    async function handleDeleteDocument(storedName) {
        try {
            const code = requireAccessCode();
            if (!window.confirm("이 문서를 문서 보관함에서 삭제할까요?")) return;
            const data = await deleteDocument(storedName, code);
            setDocuments(data.documents || []);
            showAlert(data.message || "문서를 삭제했습니다.", 1800);
        } catch (error) {
            console.error(error);
            showAlert(`문서 삭제 중 오류가 발생했습니다.\n${error.message}`);
        }
    }

    async function handleClearDocuments() {
        try {
            const code = requireAccessCode();
            if (!window.confirm("문서 보관함을 모두 비울까요?")) return;
            const data = await clearDocuments(code);
            setDocuments(data.documents || []);
            showAlert(data.message || "문서 보관함을 비웠습니다.", 1800);
        } catch (error) {
            console.error(error);
            showAlert(`문서 초기화 중 오류가 발생했습니다.\n${error.message}`);
        }
    }

    async function handleSummarizeDocuments() {
        if (!summaryFiles.length) {
            showAlert("요약할 문서를 선택해주세요.", 1800);
            return;
        }

        try {
            const code = requireAccessCode();
            setSummaryLoading(true);
            setAlert("");
            const data = await summarizeDocuments(code, summaryFiles);
            setSummaryResult(data);
            setSummaryFiles([]);
            setSummaryInputKey((value) => value + 1);
            showAlert(data.message || "문서 요약이 완료되었습니다.", 1800);
        } catch (error) {
            console.error(error);
            showAlert(`문서 요약 중 오류가 발생했습니다.\n${error.message}`);
        } finally {
            setSummaryLoading(false);
        }
    }

    async function handleProcessAudio() {
        if (!audioFiles.length) {
            showAlert("파일을 선택해주세요.", 1800);
            return;
        }

        try {
            const code = requireAccessCode();
            setAudioLoading(true);
            setAlert("");
            const data = await processAudio(code, audioFiles[0], sttMode);
            setAudioResult(data);
            setAudioFiles([]);
            setAudioInputKey((value) => value + 1);
            showAlert("회의 정리가 완료되었습니다.", 1800);
        } catch (error) {
            console.error(error);
            showAlert(`처리 중 오류가 발생했습니다.\n${error.message}`);
        } finally {
            setAudioLoading(false);
        }
    }

    async function handleChatSubmit() {
        if (!chatQuestion.trim()) {
            showAlert("질문을 입력해주세요.", 1800);
            return;
        }

        try {
            const code = requireAccessCode();
            setChatLoading(true);
            setAlert("");
            const data = await askPolicyBot(code, chatQuestion.trim());
            setChatResult(data);
            showAlert("답변 생성이 완료되었습니다.", 1600);
        } catch (error) {
            console.error(error);
            showAlert(`질문 처리 중 오류가 발생했습니다.\n${error.message}`);
        } finally {
            setChatLoading(false);
        }
    }

    return html`
        <div className="page">
            <div className="topbar">
                <div className="brand-block">
                    <div className="brand">
                        <span className="brand-mark">Office-Agent</span>
                        <p className="brand-title">회의와 문서를 연결하는 사내 업무 지원 AI</p>
                    </div>
                    <div className="topbar-tags">
                        <button className=${`topbar-nav-chip${activePage === "overview" ? " active" : ""}`} onClick=${() => setActivePage("overview")}>서비스 소개</button>
                        <button className=${`topbar-nav-chip${activePage === "workspace" ? " active" : ""}`} onClick=${() => setActivePage("workspace")}>작업 화면</button>
                    </div>
                </div>
                <div className="topbar-note">
                    <span className="topbar-note-label">서비스 상태</span>
                    <p>문서 저장, 검색, 요약, 답변 흐름을 실제 작업 화면에서 바로 확인할 수 있습니다.</p>
                </div>
            </div>

            ${activePage === "overview" ? html`
                <${HeroSection}
                    documentCount=${documentCount}
                    onOpenWorkspace=${() => setActivePage("workspace")}
                    onOpenDocuments=${() => {
                        setActivePage("workspace");
                        setActiveWorkspace("documents");
                    }}
                />
            ` : null}
            <${AlertBanner} message=${alert} />

            <section id="workspace" className=${`workspace-shell studio${activePage !== "workspace" ? " hidden-section" : ""}`}>
                <div className="workspace-topbar">
                    <div>
                        <p className="section-kicker">작업 공간</p>
                        <h2 className="workspace-title">${activeWorkspaceMeta.label}</h2>
                        <p className="workspace-copy">${activeWorkspaceMeta.description}</p>
                    </div>
                    <div className="workspace-access">
                        <span className="workspace-access-label">접속 코드</span>
                        <input
                            className="workspace-access-input"
                            type="password"
                            autoComplete="off"
                            placeholder="공통 접속 코드"
                            value=${accessCode}
                            onChange=${(event) => setAccessCode(event.target.value)}
                        />
                    </div>
                </div>

                <div className="workspace-nav">
                    ${workspaceItems.map((item) => html`
                        <button
                            key=${item.key}
                            className=${`workspace-tab${activeWorkspace === item.key ? " active" : ""}`}
                            onClick=${() => setActiveWorkspace(item.key)}
                        >
                            <span className="workspace-tab-title">${item.label}</span>
                            <span className="workspace-tab-copy">${item.description}</span>
                        </button>
                    `)}
                </div>

                <div className="workspace-stage">
                    ${activeWorkspace === "documents" ? html`
                        <${DocumentLibrarySection}
                            documents=${documents}
                            files=${docFiles}
                            inputKey=${docInputKey}
                            loading=${docLoading}
                            onFilesChange=${(event) => setDocFiles(getFilesFromInput(event))}
                            onUpload=${handleDocumentUpload}
                            onDelete=${handleDeleteDocument}
                            onClear=${handleClearDocuments}
                        />
                    ` : null}

                    ${activeWorkspace === "summary" ? html`
                        <${SummarySection}
                            files=${summaryFiles}
                            inputKey=${summaryInputKey}
                            loading=${summaryLoading}
                            result=${summaryResult}
                            onFilesChange=${(event) => setSummaryFiles(getFilesFromInput(event))}
                            onSummarize=${handleSummarizeDocuments}
                        />
                    ` : null}

                    ${activeWorkspace === "meeting" ? html`
                        <${AudioSection}
                            files=${audioFiles}
                            inputKey=${audioInputKey}
                            sttMode=${sttMode}
                            loading=${audioLoading}
                            result=${audioResult}
                            onFilesChange=${(event) => setAudioFiles(getFilesFromInput(event))}
                            onModeChange=${(event) => setSttMode(event.target.value)}
                            onSubmit=${handleProcessAudio}
                        />
                    ` : null}

                    ${activeWorkspace === "chat" ? html`
                        <${ChatSection}
                            question=${chatQuestion}
                            loading=${chatLoading}
                            result=${chatResult}
                            onQuestionChange=${(event) => setChatQuestion(event.target.value)}
                            onSubmit=${handleChatSubmit}
                        />
                    ` : null}
                </div>
            </section>
        </div>
    `;
}

const rootElement = document.getElementById("app");
ReactDOM.createRoot(rootElement).render(html`<${App} />`);
