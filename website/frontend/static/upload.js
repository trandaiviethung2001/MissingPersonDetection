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

    function setStatus(text, tone) {
        statusBox.textContent = text;
        statusBox.dataset.tone = tone || "info";
    }

    function resetResult() {
        videoEl.removeAttribute("src");
        videoEl.load();
        videoEl.style.display = "none";
        placeholder.style.display = "";
        summaryBox.textContent = "";
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
        setStatus(
            `Uploading ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB) and running detection…`,
            "info",
        );

        const startedAt = performance.now();
        try {
            const response = await fetch("/api/upload", { method: "POST", body: fd });
            const data = await response.json().catch(() => ({}));

            if (!response.ok) {
                const detail = data?.detail || response.statusText || "Unknown error";
                setStatus(`Error: ${detail}`, "error");
                return;
            }

            const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
            const summary = data.summary || {};
            videoEl.src = data.video_url;
            videoEl.style.display = "";
            placeholder.style.display = "none";
            summaryBox.textContent = summary.text || "";

            const count = summary.count ?? 0;
            setStatus(
                count
                    ? `Done in ${elapsed}s — ${count} detection(s).`
                    : `Done in ${elapsed}s — no missing persons detected.`,
                count ? "success" : "info",
            );
        } catch (err) {
            setStatus(`Network error: ${err.message || err}`, "error");
        } finally {
            submitBtn.disabled = false;
        }
    });
})();
