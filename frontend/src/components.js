import { copyText, downloadTextFile, formatBytes, formatDate, html } from "./utils.js";

function getSafeFilename(name, suffix = ".txt") {
    const base = String(name || "download").replace(/[\\/:*?"<>|]+/g, "_");
    return base.endsWith(suffix) ? base : `${base}${suffix}`;
}

function SectionHeading({ kicker, title, description, pill }) {
    return html`
        <div>
            ${kicker ? html`<p className="section-kicker">${kicker}</p>` : null}
            <div className="section-head">
                <div>
                    <h2>${title}</h2>
                    ${description ? html`<p>${description}</p>` : null}
                </div>
                ${pill ? html`<span className="pill">${pill}</span>` : null}
            </div>
        </div>
    `;
}

export function HeroSection({ documentCount, onOpenWorkspace, onOpenDocuments }) {
    return html`
        <section className="hero-simple">
            <div className="hero-simple-main">
                <div className="hero-badges">
                    <span className="hero-badge">회의 요약</span>
                    <span className="hero-badge subtle">문서 검색</span>
                    <span className="hero-badge subtle">근거 기반 답변</span>
                </div>
                <p className="eyebrow hero-showcase-kicker">업무 지원 AI</p>
                <h1 className="hero-simple-title">회의 요약과 사내 문서 검색을 한 화면에서 이어 쓰는 업무 지원 AI</h1>
                <p className="hero-simple-copy">
                    회의 음성을 정리하고, 업로드한 회사 문서를 검색해 근거와 함께 답변합니다.
                    기능 소개를 나열하기보다 실제로 바로 써볼 수 있는 작업 흐름 중심으로 구성했습니다.
                </p>
                <div className="hero-actions">
                    <button className="btn" onClick=${onOpenWorkspace}>작업 화면 열기</button>
                    <button className="btn secondary ghost" onClick=${onOpenDocuments}>문서 보관함 보기</button>
                </div>
                <div className="hero-proof-grid">
                    <div className="hero-proof-card dark">
                        <span className="label">업무 문제</span>
                        <strong>회의 정리와 문서 질의응답이 분리돼 반복 작업이 많았습니다.</strong>
                    </div>
                    <div className="hero-proof-card dark">
                        <span className="label">구현 방식</span>
                        <strong>STT, RAG, ChromaDB, 결과 저장 흐름을 하나의 서비스로 연결했습니다.</strong>
                    </div>
                    <div className="hero-proof-card dark">
                        <span className="label">운영 관점</span>
                        <strong>문서 관리, 근거 노출, 접근 코드까지 공개 운영 기준으로 다듬었습니다.</strong>
                    </div>
                </div>
            </div>

            <div className="hero-simple-side">
                <div className="hero-side-card hero-side-card-primary">
                    <p className="hero-side-kicker">서비스 한눈에 보기</p>
                    <h2>문서 업로드부터 검색, 요약, 답변까지 바로 확인할 수 있습니다.</h2>
                    <ul className="hero-side-list">
                        <li>문서 보관함과 벡터 저장소를 같은 흐름으로 관리</li>
                        <li>답변 뒤에 근거 문서 하이라이트까지 함께 제공</li>
                        <li>공개 데모 운영을 고려한 접속 코드와 요청 제한 적용</li>
                    </ul>
                </div>
                <div className="hero-summary-strip">
                    <div className="hero-mini-card">
                        <span className="label">활성 문서</span>
                        <strong>${documentCount}개</strong>
                    </div>
                    <div className="hero-mini-card">
                        <span className="label">검색 저장소</span>
                        <strong>ChromaDB</strong>
                    </div>
                    <div className="hero-mini-card">
                        <span className="label">접근 제어</span>
                        <strong>코드 보호</strong>
                    </div>
                </div>
                <div className="hero-side-card">
                    <p className="hero-side-kicker">추천 진입 순서</p>
                    <div className="hero-flow-list">
                        <span>1. 문서 보관함</span>
                        <span>2. 회의 정리</span>
                        <span>3. 사내 규정 챗봇</span>
                    </div>
                </div>
            </div>
        </section>
    `;
}

export function AlertBanner({ message }) {
    if (!message) return null;
    return html`<div className="alert" style=${{ display: "block" }}>${message}</div>`;
}

export function DocumentLibrarySection({
    documents,
    files,
    inputKey,
    loading,
    onFilesChange,
    onUpload,
    onDelete,
    onClear,
}) {
    return html`
        <section className="panel">
            <div className="panel-inner">
                <${SectionHeading}
                    kicker="지식 보관함"
                    title="문서 지식 보관함"
                    description="업로드한 문서는 챗봇과 회의 요약의 공통 지식으로 사용됩니다."
                />
                <div className="actions" style=${{ marginTop: "-6px", marginBottom: "18px" }}>
                    <button className="btn secondary" onClick=${onClear}>전체 비우기</button>
                </div>
                <div className="field">
                    <label htmlFor="docInput">문서 업로드</label>
                    <input key=${inputKey} id="docInput" type="file" accept=".pdf,.docx,.txt,.md" multiple onChange=${onFilesChange} />
                </div>
                <div className="actions" style=${{ marginTop: "12px" }}>
                    <button className="btn" onClick=${onUpload} disabled=${loading || !files.length}>
                        ${loading ? "업로드 중..." : "지식 보관함에 추가"}
                    </button>
                </div>
                ${loading ? html`<div className="loading" style=${{ display: "block", marginTop: "14px" }}>문서를 읽고 지식 저장소와 검색 인덱스를 갱신하는 중입니다...</div>` : null}
                ${!documents.length
                    ? html`<div className="empty" style=${{ display: "block", marginTop: "14px" }}>아직 업로드된 문서가 없습니다.</div>`
                    : html`
                        <div className="doc-list" style=${{ marginTop: "14px" }}>
                            ${documents.map((doc) => html`
                                <div className="doc-item" key=${doc.stored_name}>
                                    <div className="doc-header">
                                        <div>
                                            <div className="doc-name">${doc.display_name}</div>
                                            <div className="doc-sub">${formatBytes(doc.size_bytes)} · ${formatDate(doc.uploaded_at)}</div>
                                        </div>
                                        <button className="btn secondary" onClick=${() => onDelete(doc.stored_name)}>삭제</button>
                                    </div>
                                </div>
                            `)}
                        </div>
                    `}
            </div>
        </section>
    `;
}

export function SummarySection({ files, inputKey, loading, result, onFilesChange, onSummarize }) {
    return html`
        <section className="panel">
            <div className="panel-inner">
                <${SectionHeading}
                    kicker="문서 요약"
                    title="AI 문서 요약"
                    description="문서별 핵심 요약과 후속 작업을 만들고 바로 저장할 수 있습니다."
                    pill="최대 5개 문서"
                />
                <div className="field">
                    <label htmlFor="summaryInput">요약할 문서 선택</label>
                    <input key=${inputKey} id="summaryInput" type="file" accept=".pdf,.docx,.txt,.md" multiple onChange=${onFilesChange} />
                </div>
                <div className="actions" style=${{ marginTop: "12px" }}>
                    <button className="btn" onClick=${onSummarize} disabled=${loading || !files.length}>
                        ${loading ? "요약 생성 중..." : "문서 요약 생성"}
                    </button>
                </div>
                ${loading ? html`<div className="loading" style=${{ display: "block", marginTop: "14px" }}>문서를 읽고 요약을 생성하는 중입니다...</div>` : null}
                ${result ? html`
                    <div className="result-block" style=${{ display: "block" }}>
                        <div className="summary-list">
                            ${(result.summaries || []).map((item, index) => html`
                                <div className="summary-card" key=${`${item.filename}-${index}`}>
                                    <div className="summary-header">
                                        <div className="summary-title">${index + 1}. ${item.filename}</div>
                                        <div className="inline-actions">
                                            <button className="btn secondary" onClick=${() => copyText(item.summary || "")}>복사</button>
                                            <button className="btn secondary" onClick=${() => downloadTextFile(getSafeFilename(`${item.filename}.summary`, ".txt"), item.summary || "")}>저장</button>
                                        </div>
                                    </div>
                                    <div className="summary-content">${item.summary}</div>
                                </div>
                            `)}
                        </div>
                        <div className="result-section">
                            <div className="section-head">
                                <div>
                                    <h3>통합 결과</h3>
                                    <p>팀 공유용으로 전체 결과를 한 번에 복사하거나 저장할 수 있습니다.</p>
                                </div>
                                <div className="inline-actions">
                                    <button className="btn secondary" onClick=${() => copyText(result.combined_summary || "")}>전체 복사</button>
                                    <button className="btn secondary" onClick=${() => downloadTextFile("document-summary.txt", result.combined_summary || "")}>전체 저장</button>
                                </div>
                            </div>
                            <div className="result-box" style=${{ display: "block" }}>${result.combined_summary}</div>
                        </div>
                    </div>
                ` : null}
            </div>
        </section>
    `;
}

function SourceCards({ sources, emptyMessage = "참고 문서를 찾지 못했습니다." }) {
    if (!sources?.length) {
        return html`<div className="empty" style=${{ display: "block" }}>${emptyMessage}</div>`;
    }

    return html`
        <div className="source-list">
            ${sources.map((source, index) => html`
                <div className="source-card" key=${`${source.source}-${index}`}>
                    <div className="source-title">${source.source}</div>
                    <div className="source-meta">유사도 ${(source.score ?? 0).toFixed(2)}</div>
                    <div className="source-excerpt">${source.excerpt}</div>
                </div>
            `)}
        </div>
    `;
}

export function AudioSection({ files, inputKey, sttMode, loading, result, onFilesChange, onModeChange, onSubmit }) {
    return html`
        <section className="panel">
            <div className="panel-inner">
                <${SectionHeading}
                    kicker="회의 정리"
                    title="음성 요약 및 업무 정리"
                    description="회의 음성을 전사하고, 관련 회사 문서를 붙여 요약과 후속 작업을 정리합니다."
                />
                <div className="mode-options">
                    <label className="mode-option">
                        <input type="radio" name="sttMode" value="local" checked=${sttMode === "local"} onChange=${onModeChange} />
                        <strong>일반 모드</strong>
                        <span>서버에서 음성을 변환합니다. 비용은 없지만 긴 음성은 느릴 수 있습니다.</span>
                    </label>
                    <label className="mode-option">
                        <input type="radio" name="sttMode" value="openai" checked=${sttMode === "openai"} onChange=${onModeChange} />
                        <strong>고성능 모드</strong>
                        <span>API 기반 음성 변환을 사용합니다. 정확도와 속도를 우선하며 비용이 발생합니다.</span>
                    </label>
                </div>
                <div className="field">
                    <label htmlFor="audioInput">음성 파일 선택</label>
                    <input key=${inputKey} id="audioInput" type="file" accept="audio/*" onChange=${onFilesChange} />
                </div>
                <div className="actions" style=${{ marginTop: "12px" }}>
                    <button className="btn" onClick=${onSubmit} disabled=${loading || !files.length}>
                        ${loading ? "회의 정리 중..." : "회의 정리 시작"}
                    </button>
                </div>
                ${loading ? html`<div className="loading" style=${{ display: "block", marginTop: "14px" }}>음성을 전사하고 관련 문서를 검색해 업무 요약을 작성하는 중입니다...</div>` : null}
                ${result ? html`
                    <div className="result-block" style=${{ display: "block" }}>
                        <div className="result-section">
                            <div className="section-head">
                                <div>
                                    <h3>요약 및 업무 정리</h3>
                                    <p>회의 요약은 복사하거나 텍스트 파일로 저장할 수 있습니다.</p>
                                </div>
                                <div className="inline-actions">
                                    <button className="btn secondary" onClick=${() => copyText(result.summary || "")}>복사</button>
                                    <button className="btn secondary" onClick=${() => downloadTextFile("meeting-summary.txt", result.summary || "")}>저장</button>
                                </div>
                            </div>
                            <div className="result-box" style=${{ display: "block" }}>${result.summary || "요약 결과가 없습니다."}</div>
                        </div>
                        <div className="result-section">
                            <div className="section-head">
                                <div>
                                    <h3>검색된 관련 회사 문서</h3>
                                    <p>요약 생성에 사용된 전체 검색 문맥입니다.</p>
                                </div>
                            </div>
                            <div className="result-box" style=${{ display: "block" }}>${result.retrieved_rule || "관련 내규를 찾지 못했습니다."}</div>
                        </div>
                        <div className="result-section">
                            <div className="section-head">
                                <div>
                                    <h3>근거 문서 하이라이트</h3>
                                    <p>실제로 참고한 문서와 발췌 내용을 카드 형태로 보여줍니다.</p>
                                </div>
                            </div>
                            <${SourceCards} sources=${result.sources || []} />
                        </div>
                        <div className="result-section">
                            <div className="section-head">
                                <div>
                                    <h3>전체 스크립트</h3>
                                    <p>전사 결과 원문입니다.</p>
                                </div>
                                <div className="inline-actions">
                                    <button className="btn secondary" onClick=${() => copyText(result.script || "")}>복사</button>
                                    <button className="btn secondary" onClick=${() => downloadTextFile("meeting-script.txt", result.script || "")}>저장</button>
                                </div>
                            </div>
                            <div className="result-box" style=${{ display: "block" }}>${result.script || "인식된 스크립트가 없습니다."}</div>
                        </div>
                    </div>
                ` : null}
            </div>
        </section>
    `;
}

export function ChatSection({ question, loading, result, onQuestionChange, onSubmit }) {
    return html`
        <section className="panel sidebar-card">
            <div className="panel-inner">
                <${SectionHeading}
                    kicker="규정 챗봇"
                    title="사내 규정 챗봇"
                    description="업로드된 회사 문서를 근거로 빠르고 간결하게 답합니다."
                    pill="근거 기반 답변"
                />
                <div className="field">
                    <label htmlFor="chatQuestion">질문 입력</label>
                    <textarea
                        id="chatQuestion"
                        placeholder="예: 신입사원 연차는 어떻게 발생하나요?"
                        value=${question}
                        onChange=${onQuestionChange}
                    />
                </div>
                <div className="actions" style=${{ marginTop: "12px" }}>
                    <button className="btn" onClick=${onSubmit} disabled=${loading || !question.trim()}>
                        ${loading ? "답변 생성 중..." : "답변 생성"}
                    </button>
                </div>
                ${loading ? html`<div className="loading" style=${{ display: "block", marginTop: "14px" }}>회사 문서를 검색하고 답변을 생성하는 중입니다...</div>` : null}
                ${result ? html`
                    <div className="result-block" style=${{ display: "block" }}>
                        <div className="result-section">
                            <div className="section-head">
                                <div>
                                    <h3>답변</h3>
                                    <p>응답을 바로 복사하거나 저장할 수 있습니다.</p>
                                </div>
                                <div className="inline-actions">
                                    <button className="btn secondary" onClick=${() => copyText(result.answer || "")}>복사</button>
                                    <button className="btn secondary" onClick=${() => downloadTextFile("policy-answer.txt", result.answer || "")}>저장</button>
                                </div>
                            </div>
                            <div className="result-box" style=${{ display: "block" }}>${result.answer || "답변을 생성하지 못했습니다."}</div>
                        </div>
                        <div className="result-section">
                            <div className="section-head">
                                <div>
                                    <h3>참고한 전체 문맥</h3>
                                    <p>챗봇이 참고한 전체 검색 문맥입니다.</p>
                                </div>
                            </div>
                            <div className="result-box" style=${{ display: "block" }}>${result.retrieved_rule || "참고 문서를 찾지 못했습니다."}</div>
                        </div>
                        <div className="result-section">
                            <div className="section-head">
                                <div>
                                    <h3>근거 문서 하이라이트</h3>
                                    <p>문서명과 발췌문으로 답변의 근거를 빠르게 확인할 수 있습니다.</p>
                                </div>
                            </div>
                            <${SourceCards} sources=${result.sources || []} />
                        </div>
                    </div>
                ` : null}
            </div>
        </section>
    `;
}
