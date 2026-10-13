/* ─────────────────────────────────────────────────────────────
   AI TUTOR PLATFORM — FRONTEND JAVASCRIPT CONTROLLER
───────────────────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
    const CANONICAL_ENGINES = ['standard', 'chandra', 'qwen2.5vl', 'surya'];

    function getStoredOcrEngine() {
        try {
            const stored = localStorage.getItem('selectedOcrEngine');
            if (stored && CANONICAL_ENGINES.includes(stored.toLowerCase().trim())) {
                return stored.toLowerCase().trim();
            }
        } catch (e) {}
        return 'standard';
    }

    // State management
    let state = {
        activeTab: 'student-portal',
        selectedOcrEngine: getStoredOcrEngine(),
        tests: [],
        selectedTestId: null,
        studentFile: null,
        labFile: null,
        submissions: []
    };

    // DOM Elements - Navigation
    const roleTabs = document.querySelectorAll('.role-tab');
    const portalViews = document.querySelectorAll('.portal-view');
    const globalOcrEngine = document.getElementById('global-ocr-engine');

    // DOM Elements - Student Portal
    const studentTestList = document.getElementById('student-test-list');
    const activeTestCount = document.getElementById('active-test-count');
    const submitModal = document.getElementById('submit-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');
    const modalTestTitle = document.getElementById('modal-test-title');
    const modalTestSubject = document.getElementById('modal-test-subject');
    const modalTestQuestion = document.getElementById('modal-test-question');
    const modalTestMarks = document.getElementById('modal-test-marks');
    const studentSubmitForm = document.getElementById('student-submit-form');
    const stuDropZone = document.getElementById('stu-drop-zone');
    const stuFileInput = document.getElementById('stu-file-input');
    const stuFileChip = document.getElementById('stu-file-chip');
    const stuFileName = document.getElementById('stu-file-name');
    const stuFileSize = document.getElementById('stu-file-size');
    const stuFileRemove = document.getElementById('stu-file-remove');
    const stuSubmitBtn = document.getElementById('stu-submit-btn');
    const stuLoadSampleImg = document.getElementById('stu-load-sample-img');
    const stuLoadSamplePdf = document.getElementById('stu-load-sample-pdf');
    const optionsToggle = document.getElementById('options-toggle');
    const optionsBody = document.getElementById('options-body');
    const evalLoading = document.getElementById('eval-loading');
    const evalResult = document.getElementById('eval-result');

    // Results DOM
    const resScore = document.getElementById('res-score');
    const resMaxScore = document.getElementById('res-max-score');
    const resStatusBadge = document.getElementById('res-status-badge');
    const resPercentage = document.getElementById('res-percentage');
    const resReasoning = document.getElementById('res-reasoning');
    const resCoveredList = document.getElementById('res-covered-list');
    const resMissingList = document.getElementById('res-missing-list');
    const resFeedback = document.getElementById('res-feedback');
    const resExtractedText = document.getElementById('res-extracted-text');
    const resPreprocessedImg = document.getElementById('res-preprocessed-img');
    const resImgContainer = document.getElementById('res-img-container');

    // DOM Elements - Teacher Portal
    const openCreateTestBtn = document.getElementById('open-create-test-btn');
    const createTestModal = document.getElementById('create-test-modal');
    const closeCreateModalBtn = document.getElementById('close-create-modal-btn');
    const createTestForm = document.getElementById('create-test-form');
    const teacherTestList = document.getElementById('teacher-test-list');
    const currentTestName = document.getElementById('current-test-name');
    const currentTestMeta = document.getElementById('current-test-meta');
    const submissionsTableBody = document.getElementById('submissions-table-body');
    const autoExtractQpBtn = document.getElementById('auto-extract-qp-btn');
    const qpFileInput = document.getElementById('qp-file-input');
    const qpDropZone = document.getElementById('qp-drop-zone');
    const newTestQuestions = document.getElementById('new-test-questions');
    const autoExtractModelBtn = document.getElementById('auto-extract-model-btn');
    const modelPdfInput = document.getElementById('model-pdf-input');
    const modelDropZone = document.getElementById('model-drop-zone');
    const newTestModelAns = document.getElementById('new-test-model-ans');
    const btnLoadTestSample = document.getElementById('btn-load-test-sample');

    // Teacher Override Modal
    const overrideModal = document.getElementById('override-modal');
    const closeOverrideBtn = document.getElementById('close-override-btn');
    const overrideForm = document.getElementById('override-form');
    const overrideStuName = document.getElementById('override-stu-name');
    const overrideAiScore = document.getElementById('override-ai-score');
    const overrideNewMarks = document.getElementById('override-new-marks');
    const overrideRemarks = document.getElementById('override-remarks');
    let currentOverrideSubId = null;

    // DOM Elements - OCR Lab
    const labDropZone = document.getElementById('lab-drop-zone');
    const labFileInput = document.getElementById('lab-file-input');
    const labFileChip = document.getElementById('lab-file-chip');
    const labFileName = document.getElementById('lab-file-name');
    const labSampleImg = document.getElementById('lab-sample-img');
    const labSamplePdf = document.getElementById('lab-sample-pdf');
    const labProcessBtn = document.getElementById('lab-process-btn');
    const labOcrEngine = document.getElementById('lab-ocr-engine');
    const labStatTime = document.getElementById('lab-stat-time');
    const labStatConf = document.getElementById('lab-stat-conf');
    const labStatSkew = document.getElementById('lab-stat-skew');
    const labStatEngine = document.getElementById('lab-stat-engine');

    const labOrigImg = document.getElementById('lab-orig-img');
    const labProcImg = document.getElementById('lab-proc-img');
    const labTextOutput = document.getElementById('lab-text-output');
    const labCopyBtn = document.getElementById('lab-copy-btn');

    // ─────────────────────────────────────────────────────────────
    // TOAST NOTIFICATIONS
    // ─────────────────────────────────────────────────────────────
    function showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = 'toast';
        const icon = type === 'success' ? '✅' : (type === 'error' ? '❌' : 'ℹ️');
        toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            setTimeout(() => toast.remove(), 200);
        }, 3500);
    }

    // ─────────────────────────────────────────────────────────────
    // TAB SWITCHING
    // ─────────────────────────────────────────────────────────────
    roleTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            state.activeTab = target;

            roleTabs.forEach(t => t.classList.remove('active'));
            portalViews.forEach(v => v.classList.remove('active'));

            tab.classList.add('active');
            const targetView = document.getElementById(`view-${target}`);
            if (targetView) targetView.classList.add('active');

            if (target === 'student-portal' || target === 'teacher-portal') {
                loadTests();
            }
        });
    });

    // ─────────────────────────────────────────────────────────────
    // API HELPERS: LOAD TESTS & SUBMISSIONS
    // ─────────────────────────────────────────────────────────────
    function isTestActive(test) {
        if (!test) return false;
        if (test.is_published === false) return false;
        if (test.status && test.status !== 'active' && test.status !== 'published') return false;
        return true;
    }

    async function loadTests() {
        if (studentTestList && (!state.tests || state.tests.length === 0)) {
            studentTestList.innerHTML = '<div class="table-empty">Loading active tests...</div>';
        }
        try {
            const res = await fetch('/api/tests');
            if (!res.ok) {
                throw new Error(`Server returned HTTP ${res.status}`);
            }
            const data = await res.json();
            if (data.success && Array.isArray(data.tests)) {
                state.tests = data.tests;
                renderStudentTests();
                renderTeacherTests();
            } else {
                throw new Error(data.detail || 'Invalid response from server');
            }
        } catch (e) {
            console.error('Failed to load tests:', e);
            if (studentTestList) {
                studentTestList.innerHTML = '<div class="table-empty" style="color: #dc2626;">Failed to load active tests. Please try refreshing.</div>';
            }
            if (activeTestCount) {
                activeTestCount.innerText = '0';
            }
        }
    }

    function renderStudentTests() {
        if (!studentTestList) return;
        studentTestList.innerHTML = '';

        const activeTests = (state.tests || []).filter(isTestActive);
        if (activeTestCount) {
            activeTestCount.innerText = activeTests.length;
        }

        if (activeTests.length === 0) {
            studentTestList.innerHTML = '<div class="table-empty">No active tests posted yet.</div>';
            return;
        }

        activeTests.forEach(test => {
            const card = document.createElement('div');
            card.className = 'test-card';
            card.innerHTML = `
                <div>
                    <span class="test-card-subject">${escapeHtml(test.subject || 'General')}</span>
                    <h3 class="test-card-title">${escapeHtml(test.title || 'Untitled Test')}</h3>
                    <div class="test-card-meta">
                        <span>📅 Due: ${escapeHtml(test.due_date || 'No due date')}</span>
                        <span>📝 ${escapeHtml(test.id)}</span>
                    </div>
                </div>
                <div class="test-card-footer">
                    <span class="marks-tag">Max: ${escapeHtml(test.max_marks !== undefined ? test.max_marks : 10)} Marks</span>
                    <button type="button" class="btn-primary open-submit-btn" data-id="${escapeHtml(test.id)}">
                        <span>Submit Solution</span>
                    </button>
                </div>
            `;
            studentTestList.appendChild(card);
        });

        document.querySelectorAll('.open-submit-btn').forEach(btn => {
            btn.addEventListener('click', () => openSubmitModal(btn.dataset.id));
        });
    }

    function renderTeacherTests() {
        if (!teacherTestList) return;
        teacherTestList.innerHTML = '';

        if (!state.tests || state.tests.length === 0) {
            teacherTestList.innerHTML = '<div class="sidebar-item-meta" style="padding: 12px;">No tests created yet.</div>';
            return;
        }

        state.tests.forEach((test, idx) => {
            const item = document.createElement('div');
            item.className = `test-sidebar-item ${test.id === state.selectedTestId ? 'active' : ''}`;
            item.dataset.id = test.id;
            const kbStatus = test.knowledge_status || 'ready';
            const kbBadge = kbStatus === 'ready' ? '<span style="color:#16a34a; font-size:0.75rem; font-weight:600;">KB Ready ✓</span>' :
                            kbStatus === 'processing' ? '<span style="color:#d97706; font-size:0.75rem; font-weight:600;">KB Processing...</span>' :
                            kbStatus === 'failed' ? '<span style="color:#dc2626; font-size:0.75rem; font-weight:600;">KB Failed ⚠️</span>' : '';
            item.innerHTML = `
                <div class="sidebar-item-title">${escapeHtml(test.title)}</div>
                <div class="sidebar-item-meta" style="display: flex; justify-content: space-between; align-items: center;">
                    <span>${test.submissions_count || 0} Subs | Max: ${test.max_marks} M</span>
                    ${kbBadge}
                </div>
            `;
            item.addEventListener('click', () => selectTeacherTest(test.id));
            teacherTestList.appendChild(item);

            if (idx === 0 && !state.selectedTestId) {
                selectTeacherTest(test.id);
            }
        });
    }

    async function selectTeacherTest(testId) {
        state.selectedTestId = testId;
        const test = state.tests.find(t => t.id === testId);
        if (!test) return;

        // Highlight active sidebar item
        document.querySelectorAll('.test-sidebar-item').forEach(el => {
            el.classList.toggle('active', el.dataset.id === testId);
        });

        currentTestName.innerText = test.title;
        currentTestMeta.innerText = `Course: ${test.subject} | Max Marks: ${test.max_marks} | Due: ${test.due_date}`;

        // Update Knowledge Base Status Card
        const kbCard = document.getElementById('teacher-kb-status-card');
        const kbBadge = document.getElementById('teacher-kb-status-badge');
        const reingestBtn = document.getElementById('teacher-reingest-btn');
        if (kbCard && kbBadge) {
            kbCard.style.display = 'inline-flex';
            const status = test.knowledge_status || 'ready';
            if (status === 'ready') {
                const chunksCount = test.knowledge_stats?.chunks_created || (test.questions?.length || 0);
                kbBadge.style.background = '#dcfce7';
                kbBadge.style.color = '#166534';
                kbBadge.innerText = `Ready ✓ (${chunksCount} Chroma chunks)`;
                kbBadge.onclick = null;
                kbBadge.style.cursor = 'default';
                kbBadge.title = `Knowledge base active: ${chunksCount} semantic chunks indexed in persistent ChromaDB.`;
            } else if (status === 'processing') {
                kbBadge.style.background = '#fef9c3';
                kbBadge.style.color = '#854d0e';
                kbBadge.innerText = 'Processing... ⏳';
                kbBadge.onclick = null;
                kbBadge.style.cursor = 'default';
                kbBadge.title = 'Semantic chunking & Ollama embedding currently in progress.';
            } else if (status === 'failed') {
                kbBadge.style.background = '#fee2e2';
                kbBadge.style.color = '#991b1b';
                kbBadge.innerText = 'Failed — View Details ⚠️';
                kbBadge.style.cursor = 'pointer';
                kbBadge.title = 'Click to inspect error details';
                kbBadge.onclick = () => alert(`Knowledge Ingestion Error for ${test.id}:\n\n${test.knowledge_error || 'Unknown error'}`);
            } else {
                kbBadge.style.background = '#f1f5f9';
                kbBadge.style.color = '#475569';
                kbBadge.innerText = status;
                kbBadge.onclick = null;
                kbBadge.style.cursor = 'default';
            }
        }

        if (reingestBtn) {
            reingestBtn.onclick = async () => {
                showToast(`Re-indexing knowledge base for '${testId}'...`);
                try {
                    const res = await fetch(`/api/tests/${testId}/ingest`, { method: 'POST' });
                    const d = await res.json();
                    if (d.success) {
                        showToast(`Knowledge base indexed successfully (${d.chunks_created} chunks in ChromaDB)!`, 'success');
                        await loadTests();
                        selectTeacherTest(testId);
                    } else {
                        showToast(d.detail || 'Ingestion failed.', 'error');
                    }
                } catch (err) {
                    showToast(`Error re-indexing: ${err.message}`, 'error');
                }
            };
        }

        const deleteTestBtn = document.getElementById('teacher-delete-test-btn');
        if (deleteTestBtn) {
            deleteTestBtn.onclick = async () => {
                if (!confirm(`Are you sure you want to completely delete the test '${test.title}' and all its submissions? This cannot be undone.`)) {
                    return;
                }
                showToast(`Deleting test '${testId}'...`);
                try {
                    const res = await fetch(`/api/tests/${testId}`, { method: 'DELETE' });
                    const d = await res.json();
                    if (d.success) {
                        showToast(`Test deleted successfully.`, 'success');
                        state.selectedTestId = null;
                        document.getElementById('current-test-name').innerText = 'Select a test to view submissions';
                        document.getElementById('current-test-meta').innerText = 'Real-time student evaluation records.';
                        kbCard.style.display = 'none';
                        if (submissionsTableBody) {
                            submissionsTableBody.innerHTML = '<tr><td colspan="7" class="table-empty">Select a test to view its submissions.</td></tr>';
                        }
                        await loadTests();
                    } else {
                        showToast(d.detail || 'Deletion failed.', 'error');
                    }
                } catch (err) {
                    showToast(`Error deleting test: ${err.message}`, 'error');
                }
            };
        }

        try {
            const res = await fetch(`/api/tests/${testId}/submissions`);
            const data = await res.json();
            if (data.success) {
                state.submissions = data.submissions;
                renderSubmissionsTable(data.submissions, test.max_marks);
            }
        } catch (e) {
            console.error('Failed to load submissions:', e);
        }
    }

    function renderSubmissionsTable(subs, maxMarks) {
        if (!submissionsTableBody) return;
        submissionsTableBody.innerHTML = '';

        if (!subs || subs.length === 0) {
            submissionsTableBody.innerHTML = '<tr><td colspan="7" class="table-empty">No student submissions recorded yet for this test.</td></tr>';
            return;
        }

        subs.forEach(sub => {
            const evalData = sub.evaluation || {};
            const aiScore = evalData.marks_obtained !== undefined ? evalData.marks_obtained : '-';
            const finalMarks = sub.teacher_override !== null && sub.teacher_override !== undefined ? sub.teacher_override : aiScore;
            const status = evalData.status || (sub.evaluation ? 'Evaluated' : 'Pending Evaluation');

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${escapeHtml(sub.student_name)}</strong></td>
                <td><code>${escapeHtml(sub.student_roll)}</code></td>
                <td>${sub.submitted_at ? sub.submitted_at.split(' ')[1] : '-'}</td>
                <td><span class="score-pill">${aiScore} / ${maxMarks}</span></td>
                <td><strong>${finalMarks} / ${maxMarks}</strong> ${sub.teacher_override !== null ? '<span style="font-size:0.75rem; color:#8b5cf6;">(Override)</span>' : ''}</td>
                <td><span class="score-pct-badge">${status}</span></td>
                <td>
                    <div style="display: flex; gap: 0.4rem;">
                        <button type="button" class="btn-pill inspect-sub-btn" data-id="${sub.id}" style="background: #ede9fe; color: #6d28d9; border-color: #ddd6fe;">👁️ Inspect Sheet</button>
                        <button type="button" class="btn-pill override-btn" data-id="${sub.id}">Override</button>
                    </div>
                </td>
            `;
            submissionsTableBody.appendChild(tr);
        });

        document.querySelectorAll('.inspect-sub-btn').forEach(btn => {
            btn.addEventListener('click', () => openInspectModal(btn.dataset.id));
        });

        document.querySelectorAll('.override-btn').forEach(btn => {
            btn.addEventListener('click', () => openOverrideModal(btn.dataset.id));
        });
    }

    // ─────────────────────────────────────────────────────────────
    // TEACHER INSPECTION MODAL (VIEW SHEET & EXTRACTION)
    // ─────────────────────────────────────────────────────────────
    const inspectModal = document.getElementById('inspect-modal');
    const closeInspectModalBtn = document.getElementById('close-inspect-modal-btn');
    const inspectStuName = document.getElementById('inspect-stu-name');
    const inspectStuRoll = document.getElementById('inspect-stu-roll');
    const inspectStuTime = document.getElementById('inspect-stu-time');
    const inspectAiScore = document.getElementById('inspect-ai-score');
    const inspectTestTag = document.getElementById('inspect-test-tag');
    const inspectOrigImg = document.getElementById('inspect-orig-img');
    const inspectProcImg = document.getElementById('inspect-proc-img');
    const inspectDocPlaceholder = document.getElementById('inspect-doc-placeholder');
    const inspectFilename = document.getElementById('inspect-filename');
    const inspectTabOrig = document.getElementById('inspect-tab-orig');
    const inspectTabProc = document.getElementById('inspect-tab-proc');
    const inspectTabRaw = document.getElementById('inspect-tab-raw');
    const inspectTabSegmented = document.getElementById('inspect-tab-segmented');
    const inspectExtractedText = document.getElementById('inspect-extracted-text');
    const inspectSegmentedAnswers = document.getElementById('inspect-segmented-answers');
    const inspectCopyTextBtn = document.getElementById('inspect-copy-text-btn');
    const inspectReasoning = document.getElementById('inspect-reasoning');
    const inspectCoveredList = document.getElementById('inspect-covered-list');
    const inspectMissingList = document.getElementById('inspect-missing-list');
    const inspectFeedback = document.getElementById('inspect-feedback');
    const inspectOverrideForm = document.getElementById('inspect-override-form');
    const inspectNewMarks = document.getElementById('inspect-new-marks');
    const inspectRemarks = document.getElementById('inspect-remarks');
    let currentInspectSubId = null;

    function openInspectModal(subId) {
        const sub = state.submissions.find(s => s.id === subId);
        if (!sub) return;

        currentInspectSubId = subId;
        const test = state.tests.find(t => t.id === sub.test_id) || {};
        const evalData = sub.evaluation || {};

        inspectTestTag.innerText = test.title || 'Assignment';
        inspectStuName.innerText = sub.student_name;
        inspectStuRoll.innerText = sub.student_roll;
        inspectStuTime.innerText = sub.submitted_at || '-';
        inspectAiScore.innerText = `${evalData.marks_obtained || 0} / ${evalData.max_marks || test.max_marks || 10} Marks`;

        // Configure sheet images
        const hasOrigImg = !!sub.original_image;
        const hasProcImg = !!sub.preprocessed_image;

        if (hasOrigImg || hasProcImg) {
            inspectDocPlaceholder.style.display = 'none';
            if (hasOrigImg) {
                inspectOrigImg.src = sub.original_image;
                inspectOrigImg.style.display = 'block';
                inspectProcImg.style.display = 'none';
                inspectTabOrig.classList.add('active');
                inspectTabProc.classList.remove('active');
            } else if (hasProcImg) {
                inspectProcImg.src = sub.preprocessed_image;
                inspectProcImg.style.display = 'block';
                inspectOrigImg.style.display = 'none';
                inspectTabProc.classList.add('active');
                inspectTabOrig.classList.remove('active');
            }
        } else {
            inspectOrigImg.style.display = 'none';
            inspectProcImg.style.display = 'none';
            inspectDocPlaceholder.style.display = 'flex';
            inspectFilename.innerText = sub.filename || 'solution.pdf';
        }

        // Extracted Text & Segmented Answers
        inspectExtractedText.innerText = sub.extracted_text || 'No text extracted from document.';
        inspectExtractedText.style.display = 'block';
        inspectSegmentedAnswers.style.display = 'none';
        if (inspectTabRaw) inspectTabRaw.classList.add('active');
        if (inspectTabSegmented) inspectTabSegmented.classList.remove('active');

        // Populate Segmented Answers
        inspectSegmentedAnswers.innerHTML = '';
        if (sub.segmented_answers && sub.segmented_answers.length > 0) {
            sub.segmented_answers.forEach(seg => {
                const card = document.createElement('div');
                card.style.border = '1px solid var(--border-color)';
                card.style.borderRadius = '6px';
                card.style.padding = '10px';
                card.style.marginBottom = '10px';
                card.style.background = '#f9fafb';
                
                const qNum = seg.detected_question_number ? `Q${seg.detected_question_number}` : 'Unknown Q';
                const subQ = seg.sub_question_identifier ? `(${seg.sub_question_identifier})` : '';
                const mapStatus = seg.mapping_status === 'mapped' ? 
                    `<span style="color: green; font-weight: 600;">✓ Mapped</span>` : 
                    `<span style="color: #dc2626; font-weight: 600;">⚠️ ${seg.mapping_status}</span>`;
                
                const eres = seg.evaluation_result || {};
                let scoreBadge = '';
                if (eres.needs_teacher_review) {
                    scoreBadge = `<span style="background: #fef08a; padding: 2px 6px; border-radius: 12px; font-size: 0.8rem; color: #854d0e;">Review Required</span>`;
                } else if (eres.evaluation_status === 'completed') {
                    scoreBadge = `<span style="background: #dcfce7; padding: 2px 6px; border-radius: 12px; font-size: 0.8rem; color: #166534;">Score: ${eres.marks_awarded}/${eres.max_marks}</span>`;
                }
                
                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px; border-bottom: 1px solid #e5e7eb; padding-bottom: 5px;">
                        <strong>${qNum}${subQ}</strong>
                        <div>${mapStatus} | Pg ${seg.page_reference || '?'} ${scoreBadge}</div>
                    </div>
                    <pre style="white-space: pre-wrap; font-family: inherit; margin: 0; font-size: 0.9rem;">${seg.extracted_answer || '(Empty)'}</pre>
                    ${eres.evaluation_status === 'completed' ? `
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px dashed #cbd5e1; font-size: 0.85rem;">
                        <strong>Reasoning:</strong> ${eres.reasoning || 'N/A'}<br/>
                        <strong>Missing:</strong> ${(eres.missing_concepts || []).join(', ') || 'None'}<br/>
                        <strong>Feedback:</strong> <span style="color: #2563eb;">${eres.feedback || 'N/A'}</span>
                    </div>` : ''}
                `;
                inspectSegmentedAnswers.appendChild(card);
            });
        } else {
            inspectSegmentedAnswers.innerHTML = '<p style="color: #6b7280; text-align: center; margin-top: 1rem;">No segmented answers available.</p>';
        }

        // AI Reasoning & breakdown summary
        let statusBadge = evalData.status || 'Evaluated';
        if (evalData.review_count > 0) {
            statusBadge = `Needs Review (${evalData.review_count})`;
        }
        inspectReasoning.innerText = `Evaluated ${evalData.evaluated_count || 0} questions. ${evalData.review_count || 0} answers require teacher review. Status: ${statusBadge}`;
        inspectFeedback.innerText = 'See segmented answers tab for detailed per-question feedback and scoring.';

        // Clear outdated lists
        inspectCoveredList.innerHTML = '';
        inspectMissingList.innerHTML = '';

        // Pre-fill override form
        inspectNewMarks.value = sub.teacher_override !== null ? sub.teacher_override : (evalData.marks_obtained || 0);
        inspectRemarks.value = sub.teacher_remarks || '';

        inspectModal.style.display = 'flex';
    }

    if (closeInspectModalBtn) {
        closeInspectModalBtn.addEventListener('click', () => {
            inspectModal.style.display = 'none';
        });
    }

    if (inspectTabOrig) {
        inspectTabOrig.addEventListener('click', () => {
            const sub = state.submissions.find(s => s.id === currentInspectSubId);
            if (!sub || !sub.original_image) return;
            inspectTabOrig.classList.add('active');
            inspectTabProc.classList.remove('active');
            inspectOrigImg.style.display = 'block';
            inspectProcImg.style.display = 'none';
        });
    }

    if (inspectTabProc) {
        inspectTabProc.addEventListener('click', () => {
            const sub = state.submissions.find(s => s.id === currentInspectSubId);
            if (!sub || !sub.preprocessed_image) return;
            inspectTabProc.classList.add('active');
            inspectTabOrig.classList.remove('active');
            inspectProcImg.style.display = 'block';
            inspectOrigImg.style.display = 'none';
        });
    }

    if (inspectTabRaw) {
        inspectTabRaw.addEventListener('click', () => {
            inspectTabRaw.classList.add('active');
            inspectTabSegmented.classList.remove('active');
            inspectExtractedText.style.display = 'block';
            inspectSegmentedAnswers.style.display = 'none';
        });
    }

    if (inspectTabSegmented) {
        inspectTabSegmented.addEventListener('click', () => {
            inspectTabSegmented.classList.add('active');
            inspectTabRaw.classList.remove('active');
            inspectSegmentedAnswers.style.display = 'block';
            inspectExtractedText.style.display = 'none';
        });
    }

    if (inspectCopyTextBtn) {
        inspectCopyTextBtn.addEventListener('click', () => {
            const text = inspectExtractedText.innerText;
            if (text) {
                navigator.clipboard.writeText(text);
                showToast('Extracted student text copied to clipboard!', 'success');
            }
        });
    }

    if (inspectOverrideForm) {
        inspectOverrideForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!currentInspectSubId) return;

            const payload = {
                marks: parseFloat(inspectNewMarks.value),
                remarks: inspectRemarks.value
            };

            try {
                const res = await fetch(`/api/submissions/${currentInspectSubId}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.success) {
                    showToast('Marks & remarks updated successfully!', 'success');
                    if (state.selectedTestId) {
                        selectTeacherTest(state.selectedTestId);
                    }
                    inspectAiScore.innerText = `${payload.marks} Marks (Revised)`;
                }
            } catch (err) {
                showToast('Failed to update marks.', 'error');
            }
        });
    }

    // ─────────────────────────────────────────────────────────────
    // STUDENT SUBMISSION MODAL
    // ─────────────────────────────────────────────────────────────
    function openSubmitModal(testId) {
        const test = state.tests.find(t => t.id === testId);
        if (!test) return;

        state.selectedTestId = testId;
        modalTestTitle.innerText = test.title;
        modalTestSubject.innerText = test.subject;
        modalTestQuestion.innerText = test.question;
        modalTestMarks.innerText = test.max_marks;

        // Reset form & states
        state.studentFile = null;
        stuFileChip.style.display = 'none';
        stuSubmitBtn.disabled = true;
        evalLoading.style.display = 'none';
        evalResult.style.display = 'none';
        studentSubmitForm.style.display = 'block';

        submitModal.style.display = 'flex';
    }

    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', () => {
            submitModal.style.display = 'none';
        });
    }

    // Accordion toggle
    if (optionsToggle) {
        optionsToggle.addEventListener('click', () => {
            const isVisible = optionsBody.style.display === 'block';
            optionsBody.style.display = isVisible ? 'none' : 'block';
            optionsToggle.querySelector('.chevron').innerText = isVisible ? '▼' : '▲';
        });
    }

    // File selection for Student Form
    if (stuDropZone) {
        stuDropZone.addEventListener('click', () => stuFileInput.click());
        stuDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            stuDropZone.classList.add('dragover');
        });
        stuDropZone.addEventListener('dragleave', () => stuDropZone.classList.remove('dragover'));
        stuDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            stuDropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                setStudentFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (stuFileInput) {
        stuFileInput.addEventListener('change', () => {
            if (stuFileInput.files.length > 0) {
                setStudentFile(stuFileInput.files[0]);
            }
        });
    }

    if (stuFileRemove) {
        stuFileRemove.addEventListener('click', () => {
            state.studentFile = null;
            stuFileInput.value = '';
            stuFileChip.style.display = 'none';
            stuSubmitBtn.disabled = true;
        });
    }

    function setStudentFile(file) {
        state.studentFile = file;
        stuFileName.innerText = file.name;
        stuFileSize.innerText = formatBytes(file.size);
        stuFileChip.style.display = 'inline-flex';
        stuSubmitBtn.disabled = false;
    }

    // Quick samples
    if (stuLoadSampleImg) {
        stuLoadSampleImg.addEventListener('click', () => loadSampleForStudent('image'));
    }
    if (stuLoadSamplePdf) {
        stuLoadSamplePdf.addEventListener('click', () => loadSampleForStudent('pdf'));
    }

    async function loadSampleForStudent(type) {
        try {
            showToast(`Loading sample ${type}...`);
            const res = await fetch(`/api/samples/${type}`);
            const data = await res.json();
            const blob = await (await fetch(data.data_url)).blob();
            const file = new File([blob], data.filename, { type: data.mime_type });
            setStudentFile(file);
            showToast(`Sample ${file.name} loaded!`, 'success');
        } catch (e) {
            showToast('Failed to load sample.', 'error');
        }
    }

    // Submit & Evaluate Student Solution
    if (studentSubmitForm) {
        studentSubmitForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!state.studentFile || !state.selectedTestId) return;

            studentSubmitForm.style.display = 'none';
            evalLoading.style.display = 'block';

            const formData = new FormData();
            formData.append('file', state.studentFile);
            formData.append('student_name', document.getElementById('stu-name').value);
            formData.append('student_roll', document.getElementById('stu-roll').value);
            formData.append('ocr_engine', state.selectedOcrEngine);
            formData.append('apply_grayscale', document.getElementById('stu-grayscale').checked);
            formData.append('apply_denoise', document.getElementById('stu-denoise').checked);
            formData.append('apply_deskew', document.getElementById('stu-deskew').checked);
            formData.append('apply_threshold', document.getElementById('stu-threshold').checked);

            try {
                // Step 1: Submit and OCR
                showToast('Processing OCR (Step 1/2)...', 'info');
                const submitRes = await fetch(`/api/tests/${state.selectedTestId}/submit`, {
                    method: 'POST',
                    body: formData
                });
                const submitData = await submitRes.json().catch(() => ({}));
                
                if (!submitRes.ok || !submitData.success) {
                    throw new Error(submitData.detail || `OCR Submission failed (HTTP ${submitRes.status}).`);
                }

                // Step 2: Evaluate
                showToast('Evaluating Answers (Step 2/2)...', 'info');
                const evalRes = await fetch(`/api/submissions/${submitData.submission_id}/evaluate`, {
                    method: 'POST'
                });
                const evalData = await evalRes.json().catch(() => ({}));
                
                evalLoading.style.display = 'none';

                if (evalRes.ok && evalData.success) {
                    // Merge data for the renderer
                    submitData.evaluation = evalData.evaluation;
                    submitData.segmented_answers = evalData.segmented_answers;
                    renderEvaluationResult(submitData);
                    showToast('Evaluation completed successfully!', 'success');
                } else {
                    showToast(evalData.detail || `Evaluation failed (HTTP ${evalRes.status}).`, 'error');
                    studentSubmitForm.style.display = 'block';
                }
            } catch (err) {
                evalLoading.style.display = 'none';
                studentSubmitForm.style.display = 'block';
                showToast('Server error during processing: ' + err.message, 'error');
            }
        });
    }

    function renderEvaluationResult(data) {
        const ev = data.evaluation || {};
        const test = state.tests.find(t => t.id === state.selectedTestId) || {};

        resScore.innerText = ev.marks_obtained !== undefined ? ev.marks_obtained : '-';
        resMaxScore.innerText = ev.max_marks || test.max_marks || 10;
        
        let pct = 0;
        if (ev.max_marks > 0) {
            pct = Math.round((ev.marks_obtained / ev.max_marks) * 100);
        }
        resPercentage.innerText = `${pct}% Overall Score`;
        
        let statusBadge = ev.status || 'Evaluated';
        if (ev.review_count > 0) {
            statusBadge = `Needs Review (${ev.review_count})`;
            resStatusBadge.style.background = '#fef08a';
            resStatusBadge.style.color = '#854d0e';
        } else {
            resStatusBadge.style.background = 'var(--primary-color)';
            resStatusBadge.style.color = 'white';
        }
        resStatusBadge.innerText = statusBadge;
        
        resReasoning.innerText = `Evaluated ${ev.evaluated_count || 0} questions. ${ev.review_count || 0} answers require teacher review.`;
        resFeedback.innerText = 'See per-question breakdown for detailed feedback.';

        // Populate detailed segmented evaluation
        resCoveredList.innerHTML = '';
        resMissingList.innerHTML = ''; // We can use this for the segmented breakdown
        
        const breakdownContainer = document.createElement('div');
        breakdownContainer.style.marginTop = '1rem';
        
        const segments = data.segmented_answers || [];
        if (segments.length === 0) {
            breakdownContainer.innerHTML = '<p>No questions mapped.</p>';
        } else {
            segments.forEach(seg => {
                const card = document.createElement('div');
                card.style.border = '1px solid var(--border-color)';
                card.style.borderRadius = '6px';
                card.style.padding = '10px';
                card.style.marginBottom = '10px';
                card.style.background = '#f9fafb';
                
                const qNum = seg.detected_question_number ? `Q${seg.detected_question_number}` : 'Unknown Q';
                const subQ = seg.sub_question_identifier ? `(${seg.sub_question_identifier})` : '';
                const eres = seg.evaluation_result || {};
                
                let scoreBadge = '';
                if (eres.needs_teacher_review) {
                    scoreBadge = `<span style="background: #fef08a; padding: 2px 6px; border-radius: 12px; font-size: 0.8rem; color: #854d0e;">Review Required</span>`;
                } else {
                    scoreBadge = `<span style="background: #dcfce7; padding: 2px 6px; border-radius: 12px; font-size: 0.8rem; color: #166534;">Score: ${eres.marks_awarded}/${eres.max_marks}</span>`;
                }
                
                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px; border-bottom: 1px solid #e5e7eb; padding-bottom: 5px;">
                        <strong>${qNum}${subQ}</strong>
                        <div>${scoreBadge}</div>
                    </div>
                    <div style="font-size: 0.9rem; margin-bottom: 8px;">
                        <strong>Answer:</strong><br/>
                        <span style="color: #4b5563;">${seg.extracted_answer || '(Empty)'}</span>
                    </div>
                    <div style="font-size: 0.9rem;">
                        <strong>Reasoning:</strong> ${eres.reasoning || 'N/A'}<br/>
                        <strong>Missing/Issues:</strong> ${(eres.missing_concepts || []).join(', ') || 'None'}<br/>
                        <strong>Feedback:</strong> <span style="color: #2563eb;">${eres.feedback || 'N/A'}</span>
                    </div>
                `;
                breakdownContainer.appendChild(card);
            });
        }
        
        // Append breakdown to one of the lists to fit in the current UI without changing HTML
        resMissingList.appendChild(breakdownContainer);

        // Raw text & preview
        resExtractedText.innerText = data.extracted_text || 'No text extracted.';
        if (data.preprocessed_image) {
            resPreprocessedImg.src = data.preprocessed_image;
            resImgContainer.style.display = 'block';
        } else {
            resImgContainer.style.display = 'none';
        }

        evalResult.style.display = 'block';
    }

    // ─────────────────────────────────────────────────────────────
    // TEACHER PORTAL: CREATE TEST & OVERRIDES
    // ─────────────────────────────────────────────────────────────
    if (openCreateTestBtn) {
        openCreateTestBtn.addEventListener('click', () => {
            createTestModal.style.display = 'flex';
        });
    }
    if (closeCreateModalBtn) {
        closeCreateModalBtn.addEventListener('click', () => {
            createTestModal.style.display = 'none';
        });
    }

    // Helper to auto-extract text from PDF/Image into a target textarea
    async function extractToField(file, targetTextarea, fieldName) {
        if (!file || !targetTextarea) return;
        const currentEngine = state.selectedOcrEngine || 'standard';
        const engineLabel = getEngineLabel(currentEngine);
        showToast(`Extracting ${fieldName} from ${file.name} using ${engineLabel}...`);
        targetTextarea.placeholder = `⏳ Extracting text with ${engineLabel}...`;
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('ocr_engine', currentEngine);
        formData.append('apply_grayscale', 'true');
        formData.append('apply_denoise', 'true');
        formData.append('apply_deskew', 'true');
        formData.append('apply_threshold', 'false');

        try {
            const res = await fetch('/api/extract', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success && data.text) {
                targetTextarea.value = data.text;
                showToast(`${fieldName} extracted successfully (${getEngineLabel(data.ocr_engine || currentEngine)})!`, 'success');
            } else {
                showToast(data.detail || data.error || `Failed to extract ${fieldName}.`, 'error');
            }
        } catch (e) {
            showToast(`Error extracting ${fieldName}: ${e.message}`, 'error');
        }
    }

    // 1. Question Paper Auto-Extraction Handlers
    if (autoExtractQpBtn) {
        autoExtractQpBtn.addEventListener('click', () => qpFileInput.click());
    }
    if (qpFileInput) {
        qpFileInput.addEventListener('change', () => {
            if (qpFileInput.files.length > 0) {
                extractToField(qpFileInput.files[0], newTestQuestions, 'Question Paper');
            }
        });
    }
    if (qpDropZone) {
        qpDropZone.addEventListener('click', () => qpFileInput.click());
        qpDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            qpDropZone.classList.add('dragover');
        });
        qpDropZone.addEventListener('dragleave', () => qpDropZone.classList.remove('dragover'));
        qpDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            qpDropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                extractToField(e.dataTransfer.files[0], newTestQuestions, 'Question Paper');
            }
        });
    }

    // 2. Model Solution Auto-Extraction Handlers
    if (autoExtractModelBtn) {
        autoExtractModelBtn.addEventListener('click', () => modelPdfInput.click());
    }
    if (modelPdfInput) {
        modelPdfInput.addEventListener('change', () => {
            if (modelPdfInput.files.length > 0) {
                extractToField(modelPdfInput.files[0], newTestModelAns, 'Model Answer');
            }
        });
    }
    if (modelDropZone) {
        modelDropZone.addEventListener('click', () => modelPdfInput.click());
        modelDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            modelDropZone.classList.add('dragover');
        });
        modelDropZone.addEventListener('dragleave', () => modelDropZone.classList.remove('dragover'));
        modelDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            modelDropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                extractToField(e.dataTransfer.files[0], newTestModelAns, 'Model Answer');
            }
        });
    }

    // 3. Quick Template Auto-Fill Button
    if (btnLoadTestSample) {
        btnLoadTestSample.addEventListener('click', () => {
            const randSuffix = Math.floor(100 + Math.random() * 900);
            document.getElementById('new-test-id').value = `test_ml_opt_${randSuffix}`;
            document.getElementById('new-test-subject').value = 'Machine Learning & Optimization';
            document.getElementById('new-test-title').value = 'Gradient Descent & Cost Function Minimization';
            document.getElementById('new-test-marks').value = '10';
            document.getElementById('new-test-due').value = '2026-11-15';
            newTestQuestions.value = '1. Explain the mathematical intuition of Gradient Descent in Supervised Learning.\n2. Differentiate between Batch Gradient Descent and Stochastic Gradient Descent (SGD).\n3. Why is the learning rate hyperparameter crucial for convergence?';
            newTestModelAns.value = '1. Gradient Descent is an iterative optimization algorithm that updates model parameters in the direction of steepest descent (negative gradient of the cost function) to minimize loss.\n2. Batch Gradient Descent uses the entire training dataset to calculate the gradient at each step (stable but computationally heavy). SGD updates parameters per single training sample (fast, adds stochastic noise that can escape local minima).\n3. Learning rate determines step size. Too high causes oscillation or divergence; too low leads to very slow training or getting trapped.';
            document.getElementById('new-test-rubric').value = 'Cost function minimization concept: 3 marks; Batch vs SGD distinction: 4 marks; Learning rate tradeoffs: 3 marks.';
            showToast('Sample Machine Learning assignment loaded into form!', 'success');
        });
    }

    const saveTestBtn = document.getElementById('save-test-btn');
    let isCreatingTest = false;

    if (createTestForm) {
        createTestForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (isCreatingTest) return;

            const newIdInput = document.getElementById('new-test-id');
            const newTitleInput = document.getElementById('new-test-title');
            const newSubjectInput = document.getElementById('new-test-subject');
            const newMarksInput = document.getElementById('new-test-marks');
            const newDueInput = document.getElementById('new-test-due');
            const newRubricInput = document.getElementById('new-test-rubric');

            const payload = {
                id: newIdInput ? newIdInput.value.trim() : '',
                title: newTitleInput ? newTitleInput.value.trim() : '',
                subject: newSubjectInput ? newSubjectInput.value.trim() : '',
                max_marks: newMarksInput ? parseFloat(newMarksInput.value) : 10,
                due_date: newDueInput ? newDueInput.value : '',
                question: newTestQuestions ? newTestQuestions.value.trim() : '',
                model_answer: newTestModelAns ? newTestModelAns.value.trim() : '',
                rubric: newRubricInput ? newRubricInput.value.trim() : ''
            };

            if (!payload.title || !payload.subject || !payload.question || !payload.model_answer) {
                showToast('Please fill in all required test fields.', 'error');
                return;
            }

            isCreatingTest = true;
            if (saveTestBtn) {
                saveTestBtn.disabled = true;
                saveTestBtn.innerHTML = '<span>Publishing & Indexing...</span> ⏳';
            }

            try {
                const res = await fetch('/api/tests', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) {
                    throw new Error(data.detail || `Failed to create test (HTTP ${res.status}).`);
                }

                // If files were attached to the upload zones, ingest the original files
                if ((qpFileInput && qpFileInput.files.length > 0) || (modelPdfInput && modelPdfInput.files.length > 0)) {
                    showToast('Uploading and archiving attached documents...', 'info');
                    try {
                        const formData = new FormData();
                        if (qpFileInput && qpFileInput.files.length > 0) {
                            formData.append('question_paper', qpFileInput.files[0]);
                        }
                        if (modelPdfInput && modelPdfInput.files.length > 0) {
                            formData.append('model_answer', modelPdfInput.files[0]);
                        }
                        formData.append('question_text', payload.question);
                        formData.append('model_answer_text', payload.model_answer);
                        formData.append('rubric_text', payload.rubric);

                        await fetch(`/api/tests/${data.test?.id || payload.id}/ingest`, {
                            method: 'POST',
                            body: formData
                        });
                    } catch (ingestErr) {
                        console.warn('File ingestion notice:', ingestErr);
                    }
                }

                createTestModal.style.display = 'none';
                createTestForm.reset();
                if (qpDropZone) {
                    qpDropZone.innerHTML = '<span>📄 Drag & drop Question Paper PDF / Image here, or <strong style="color: var(--accent-indigo);">click to browse</strong></span>';
                }
                if (modelDropZone) {
                    modelDropZone.innerHTML = '<span>📑 Drag & drop Solution Key PDF here, or <strong style="color: var(--accent-indigo);">click to browse</strong></span>';
                }

                showToast(`Test '${payload.id}' published successfully!`, 'success');
                await loadTests();
                selectTeacherTest(data.test?.id || payload.id);
            } catch (err) {
                console.error('Error publishing test:', err);
                showToast(err.message || 'Error publishing test.', 'error');
            } finally {
                isCreatingTest = false;
                if (saveTestBtn) {
                    saveTestBtn.disabled = false;
                    saveTestBtn.innerHTML = 'Publish Test to Student Portal';
                }
            }
        });
    }

    // Teacher Override Modal Handlers
    function openOverrideModal(subId) {
        const sub = state.submissions.find(s => s.id === subId);
        if (!sub) return;

        currentOverrideSubId = subId;
        overrideStuName.innerText = `${sub.student_name} (${sub.student_roll})`;
        overrideAiScore.innerText = `${sub.evaluation?.marks_obtained || 0} Marks`;
        overrideNewMarks.value = sub.teacher_override !== null ? sub.teacher_override : (sub.evaluation?.marks_obtained || 0);
        overrideRemarks.value = sub.teacher_remarks || '';

        overrideModal.style.display = 'flex';
    }

    if (closeOverrideBtn) {
        closeOverrideBtn.addEventListener('click', () => overrideModal.style.display = 'none');
    }

    if (overrideForm) {
        overrideForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!currentOverrideSubId) return;

            const payload = {
                marks: parseFloat(overrideNewMarks.value),
                remarks: overrideRemarks.value
            };

            try {
                const res = await fetch(`/api/submissions/${currentOverrideSubId}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.success) {
                    overrideModal.style.display = 'none';
                    showToast('Marks updated successfully!', 'success');
                    if (state.selectedTestId) {
                        selectTeacherTest(state.selectedTestId);
                    }
                }
            } catch (err) {
                showToast('Failed to update marks.', 'error');
            }
        });
    }

    // ─────────────────────────────────────────────────────────────
    // OCR DIAGNOSTIC LAB
    // ─────────────────────────────────────────────────────────────
    if (labDropZone) {
        labDropZone.addEventListener('click', () => labFileInput.click());
        labDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            labDropZone.classList.add('dragover');
        });
        labDropZone.addEventListener('dragleave', () => labDropZone.classList.remove('dragover'));
        labDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            labDropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                setLabFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (labFileInput) {
        labFileInput.addEventListener('change', () => {
            if (labFileInput.files.length > 0) {
                setLabFile(labFileInput.files[0]);
            }
        });
    }

    function setLabFile(file) {
        state.labFile = file;
        labFileName.innerText = file.name;
        labFileChip.style.display = 'inline-flex';
        labProcessBtn.disabled = false;

        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                labOrigImg.src = e.target.result;
                labOrigImg.style.display = 'block';
                document.querySelector('#lab-orig-viewport .placeholder-text').style.display = 'none';
            };
            reader.readAsDataURL(file);
        } else {
            labOrigImg.style.display = 'none';
            document.querySelector('#lab-orig-viewport .placeholder-text').innerText = '📄 ' + file.name;
            document.querySelector('#lab-orig-viewport .placeholder-text').style.display = 'block';
        }
    }

    if (labSampleImg) {
        labSampleImg.addEventListener('click', () => loadSampleForLab('image'));
    }
    if (labSamplePdf) {
        labSamplePdf.addEventListener('click', () => loadSampleForLab('pdf'));
    }

    async function loadSampleForLab(type) {
        try {
            showToast(`Loading sample ${type}...`);
            const res = await fetch(`/api/samples/${type}`);
            const data = await res.json();
            const blob = await (await fetch(data.data_url)).blob();
            const file = new File([blob], data.filename, { type: data.mime_type });
            setLabFile(file);
            showToast(`Loaded ${file.name}`, 'success');
        } catch (e) {
            showToast('Failed to load sample', 'error');
        }
    }

    function getEngineLabel(val) {
        const canonical = (val || '').toLowerCase().trim();
        if (canonical === 'chandra') return 'Chandra OCR';
        if (canonical === 'qwen2.5vl') return 'Qwen2.5-VL OCR';
        if (canonical === 'surya') return 'Surya OCR';
        if (canonical === 'standard') return 'Standard OCR';
        return val || 'Standard OCR';
    }

    function syncOcrEngine(newEngine, source = 'system') {
        const canonical = (newEngine || 'standard').toLowerCase().trim();
        if (!CANONICAL_ENGINES.includes(canonical)) return;
        state.selectedOcrEngine = canonical;
        try {
            localStorage.setItem('selectedOcrEngine', canonical);
        } catch (e) {}

        if (globalOcrEngine && globalOcrEngine.value !== canonical) {
            globalOcrEngine.value = canonical;
        }
        if (labOcrEngine && labOcrEngine.value !== canonical) {
            labOcrEngine.value = canonical;
        }
        if (labStatEngine) {
            labStatEngine.innerText = getEngineLabel(canonical);
        }
    }

    if (globalOcrEngine) {
        globalOcrEngine.value = state.selectedOcrEngine;
        globalOcrEngine.addEventListener('change', (e) => {
            const chosen = e.target.value;
            syncOcrEngine(chosen, 'navbar');
            showToast(`Global OCR Engine: ${getEngineLabel(chosen)}`);
        });
    }

    if (labOcrEngine) {
        labOcrEngine.value = state.selectedOcrEngine;
        labOcrEngine.addEventListener('change', (e) => {
            const chosen = e.target.value;
            syncOcrEngine(chosen, 'lab');
            showToast(`Global OCR Engine: ${getEngineLabel(chosen)}`);
        });
    }

    // Initial synchronization on page load
    syncOcrEngine(state.selectedOcrEngine, 'init');


    if (labProcessBtn) {
        labProcessBtn.addEventListener('click', async () => {
            if (!state.labFile) return;

            const selectedEngine = state.selectedOcrEngine || 'standard';
            const engineLabel = getEngineLabel(selectedEngine);

            labProcessBtn.disabled = true;
            labProcessBtn.innerText = selectedEngine === 'standard' ? 'Processing Pipeline...' : `Running ${engineLabel}...`;
            labTextOutput.innerText = selectedEngine === 'standard' ? 'Running OCR and OpenCV binarization...' : `Executing ${engineLabel} inference...`;

            if (labStatEngine) {
                labStatEngine.innerText = engineLabel;
            }

            const formData = new FormData();
            formData.append('file', state.labFile);
            formData.append('ocr_engine', selectedEngine);
            formData.append('apply_grayscale', document.getElementById('lab-gray').checked);
            formData.append('apply_denoise', document.getElementById('lab-denoise').checked);
            formData.append('apply_deskew', document.getElementById('lab-deskew').checked);
            formData.append('apply_threshold', document.getElementById('lab-thresh').checked);

            try {
                const res = await fetch('/api/extract', { method: 'POST', body: formData });
                let data;
                try {
                    data = await res.json();
                } catch (parseErr) {
                    const textErr = await res.text();
                    data = { success: false, detail: textErr || 'Server returned an invalid response.' };
                }
                
                labProcessBtn.disabled = false;
                labProcessBtn.innerText = 'Run Extraction Pipeline';

                // Display the engine that actually processed the request according to the backend
                const returnedEngine = data.ocr_engine || selectedEngine;
                const engineName = getEngineLabel(returnedEngine);
                if (labStatEngine) labStatEngine.innerText = engineName;

                if (data.success) {
                    labStatTime.innerText = `${data.metadata?.processing_time_seconds || 0}s`;
                    
                    if (data.metadata?.confidence_average !== null && data.metadata?.confidence_average !== undefined) {
                        labStatConf.innerText = `${Math.round(data.metadata.confidence_average * 100)}%`;
                    } else {
                        labStatConf.innerText = 'N/A';
                    }
                    
                    labStatSkew.innerText = `${data.metadata?.skew_angle ?? 0}°`;
                    labTextOutput.innerText = data.text || 'No text extracted.';

                    if (data.preprocessed_image) {
                        labProcImg.src = data.preprocessed_image;
                        labProcImg.style.display = 'block';
                        document.querySelector('#lab-proc-viewport .placeholder-text').style.display = 'none';
                    }
                    showToast(`Extraction complete using ${engineName}!`, 'success');
                } else {
                    const errorMsg = data.error || data.detail || 'Extraction failed.';
                    labTextOutput.innerText = `[${engineName} Error]\n` + errorMsg;
                    if (data.metadata?.processing_time_seconds) {
                        labStatTime.innerText = `${data.metadata.processing_time_seconds}s`;
                    }
                    labStatConf.innerText = 'N/A';
                    labStatSkew.innerText = '-';
                    showToast(`${engineName} Error: ${errorMsg}`, 'error');
                }
            } catch (err) {
                labProcessBtn.disabled = false;
                labProcessBtn.innerText = 'Run Extraction Pipeline';
                labTextOutput.innerText = 'Error: ' + err.message;
                showToast('Server error during extraction.', 'error');
            }
        });
    }

    if (labCopyBtn) {
        labCopyBtn.addEventListener('click', () => {
            const text = labTextOutput.innerText;
            if (text) {
                navigator.clipboard.writeText(text);
                showToast('Copied to clipboard!', 'success');
            }
        });
    }

    // Utility Helpers
    function escapeHtml(str) {
        if (str === null || str === undefined) return '';
        return String(str).replace(/[&<>'"]/g, tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag));
    }

    function formatBytes(bytes, decimals = 1) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    // Initial load
    loadTests();
});
