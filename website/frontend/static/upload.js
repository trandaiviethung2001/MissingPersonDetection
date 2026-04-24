(() => {
    const form = document.getElementById("upload-form");
    const fileInput = document.getElementById("video-file");
    const thresholdInput = document.getElementById("threshold-input");
    const frameSkipInput = document.getElementById("frame-skip-input");
    const submitBtn = document.getElementById("upload-submit");
    const statusBox = document.getElementById("upload-status");
    const videoEl = document.getElementById("result-video");
    const placeholder = document.getElementById("result-placeholder");
    const summaryBox = document.getElementById("upload-summary");

    const progressBox = document.getElementById("upload-progress");
    const progressFill = document.getElementById("upload-progress-fill");
    const progressLabel = document.getElementById("upload-progress-label");
    const progressPct = document.getElementById("upload-progress-pct");

    function setStatus(text, tone) {
        statusBox.textContent = text;
        statusBox.dataset.tone = tone || "info";
    }

    function showProgress(label, pct, indeterminate = false) {
        progressBox.hidden = false;
        progressLabel.textContent = label;
        if (indeterminate) {
            progressBox.classList.add("indeterminate");
            progressFill.style.width = "100%";
            progressPct.textContent = "";
        } else {
            progressBox.classList.remove("indeterminate");
            const clamped = Math.max(0, Math.min(100, Math.round(pct)));
            progressFill.style.width = `${clamped}%`;
            progressPct.textContent = `${clamped}%`;
        }
    }

    function hideProgress() {
        progressBox.hidden = true;
        progressBox.classList.remove("indeterminate");
        progressFill.style.width = "0%";
    }

    function resetResult() {
        videoEl.removeAttribute("src");
        videoEl.load();
        videoEl.style.display = "none";
        placeholder.style.display = "";
        summaryBox.textContent = "";
    }

    /**
     * POST the multipart upload with real upload-progress events.
     * Resolves with the parsed JSON response.
     */
    function postUploadWithProgress(fd, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open("POST", "/api/upload");
            xhr.responseType = "json";

            xhr.upload.addEventListener("progress", (e) => {
                if (e.lengthComputable) {
                    onProgress(e.loaded / e.total);
                }
            });
            xhr.addEventListener("load", () => {
                const data = xhr.response || {};
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(data);
                } else {
                    reject(new Error(data.detail || `HTTP ${xhr.status}`));
                }
            });
            xhr.addEventListener("error", () => reject(new Error("Network error")));
            xhr.addEventListener("abort", () => reject(new Error("Upload aborted")));
            xhr.send(fd);
        });
    }

    async function pollJob(jobId) {
        // Poll every 500ms until done / error.
        while (true) {
            const res = await fetch(`/api/upload/${jobId}`, { cache: "no-store" });
            if (!res.ok) {
                throw new Error(`Status check failed (HTTP ${res.status})`);
            }
            const job = await res.json();

            if (job.state === "queued") {
                showProgress("Queued — waiting for the worker…", 0, true);
            } else if (job.state === "processing") {
                if (job.frames_total > 0) {
                    const pct = (job.progress || 0) * 100;
                    showProgress(
                        `Processing ${job.frames_done}/${job.frames_total} frames`,
                        pct,
                    );
                } else {
                    showProgress("Processing…", 0, true);
                }
            } else if (job.state === "done") {
                showProgress("Done", 100);
                return job;
            } else if (job.state === "error") {
                throw new Error(job.error || "Detection failed");
            }

            await new Promise((r) => setTimeout(r, 500));
        }
    }

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        const file = fileInput.files?.[0];
        if (!file) {
            setStatus("Please choose a video file first.", "error");
            return;
        }

        const fd = new FormData();
        fd.append("video", file);
        if (thresholdInput.value) fd.append("threshold", thresholdInput.value);
        if (frameSkipInput.value) fd.append("frame_skip", frameSkipInput.value);

        submitBtn.disabled = true;
        resetResult();
        const sizeMb = (file.size / 1024 / 1024).toFixed(1);
        setStatus(`Uploading ${file.name} (${sizeMb} MB)…`, "info");
        showProgress("Uploading…", 0);

        const startedAt = performance.now();

        try {
            // Phase 1: upload
            const startResp = await postUploadWithProgress(fd, (frac) => {
                showProgress(`Uploading ${Math.round(frac * 100)}%`, frac * 100);
            });
            if (!startResp.ok || !startResp.job_id) {
                throw new Error(startResp.detail || "Server did not return a job id");
            }

            // Phase 2: detection (poll until done)
            setStatus("Detecting missing persons…", "info");
            const job = await pollJob(startResp.job_id);

            // Phase 3: render result
            const result = job.result || {};
            const summary = result.summary || {};
            videoEl.src = result.video_url;
            videoEl.style.display = "";
            placeholder.style.display = "none";
            summaryBox.textContent = summary.text || "";

            const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
            const count = summary.count ?? 0;
            setStatus(
                count
                    ? `Done in ${elapsed}s — ${count} detection(s).`
                    : `Done in ${elapsed}s — no missing persons detected.`,
                count ? "success" : "info",
            );
            // Keep the bar at 100% briefly, then hide it
            setTimeout(hideProgress, 1500);
        } catch (err) {
            setStatus(`Error: ${err.message || err}`, "error");
            hideProgress();
        } finally {
            submitBtn.disabled = false;
        }
    });
})();
