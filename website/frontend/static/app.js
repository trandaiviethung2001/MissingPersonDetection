const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const BACKEND_URL = `${wsProtocol}//${window.location.host}/ws`;

let ws = null;
let wsConnected = false;
let reconnectTimer = null;
let isRecording = false;
let tacticalMap = null;
let userMarker = null;
let lastGeocodeKey = null;
let geocodeAbortController = null;

const el = {};

function $(id) {
    return document.getElementById(id);
}

function cacheElements() {
    const ids = [
        "system-status", "battery-indicator", "gps-indicator", "signal-indicator",
        "video-stream", "feed-placeholder", "detection-box", "confidence-box",
        "confidence-overlay", "rec-badge", "feed-container",
        "target-id", "target-status", "confidence-bar", "confidence-text",
        "ai-status-box", "ai-status-text",
        "altitude-value", "altitude-bar", "speed-value", "speed-bar",
        "battery-value", "battery-bar",
        "btn-start", "btn-pause", "btn-lock", "btn-snapshot",
        "btn-record", "btn-rtl", "btn-emergency",
        "toast-container"
    ];

    for (const id of ids) {
        const key = id.replace(/-([a-z])/g, (m, c) => c.toUpperCase());
        el[key] = $(id);
    }
}

function connectWS() {
    if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }

    ws = new WebSocket(BACKEND_URL);

    ws.onopen = function() {
        wsConnected = true;
        toast("Backend connected", "success");
    };

    ws.onmessage = function(event) {
        try {
            handleMessage(JSON.parse(event.data));
        } catch (error) {
            console.error("bad message", error);
        }
    };

    ws.onclose = function() {
        wsConnected = false;
        toast("Backend disconnected, reconnecting...", "warning");
        reconnectTimer = setTimeout(connectWS, 3000);
    };

    ws.onerror = function(error) {
        console.error("ws error", error);
    };
}

function sendCommand(command) {
    if (!wsConnected || !ws) {
        toast("Backend is not connected", "danger");
        return;
    }
    ws.send(JSON.stringify(command));
}

function handleMessage(msg) {
    if (msg.type === "telemetry") {
        updateTelemetry(msg.data);
    } else if (msg.type === "detection") {
        updateDetection(msg.data);
    } else if (msg.type === "status") {
        updateStatus(msg.data.system);
    } else if (msg.type === "frame") {
        el.videoStream.src = "data:image/jpeg;base64," + msg.data;
        el.videoStream.classList.add("active");
        el.feedPlaceholder.style.display = "none";
    } else if (msg.type === "ack") {
        handleAck(msg.data);
    } else if (msg.type === "config") {
        console.log("runtime", msg.data);
    }
}

function handleAck(data) {
    if (data.command === "snapshot") {
        toast(data.success ? "Snapshot saved" : "Snapshot failed", data.success ? "success" : "danger");
    } else if (data.command === "record") {
        toast(data.enabled ? "Recording started" : "Recording stopped", data.enabled ? "danger" : "info");
    } else if (data.command === "lock_target" && !data.success) {
        toast("No active target available to lock", "warning");
    } else if (data.command === "runtime_error") {
        toast(data.detail || "Runtime error", "danger");
    }
}

function updateTelemetry(data) {
    if (data.battery != null) {
        const battery = Math.round(data.battery);
        el.batteryIndicator.textContent = battery + "%";
        el.batteryValue.textContent = battery + "%";
        el.batteryBar.style.width = battery + "%";
    }

    if (data.signal != null) {
        el.signalIndicator.textContent = Math.round(data.signal) + "%";
    }

    if (data.altitude != null) {
        el.altitudeValue.textContent = Math.round(data.altitude) + "m";
        el.altitudeBar.style.width = Math.min(100, data.altitude * 2) + "%";
    }

    if (data.speed != null) {
        el.speedValue.textContent = data.speed.toFixed(1) + " km/h";
        el.speedBar.style.width = Math.min(100, data.speed * 5) + "%";
    }
}

function updateDetection(data) {
    el.targetId.textContent = data.targetId || "-";
    el.targetStatus.textContent = data.status || "IDLE";

    const confidence = Math.round(data.confidence || 0);
    el.confidenceBar.style.width = confidence + "%";
    el.confidenceText.textContent = confidence + "%";
    el.confidenceOverlay.textContent = confidence + "%";

    if (!data.bbox || data.status === "IDLE") {
        el.detectionBox.classList.remove("visible");
        el.confidenceBox.classList.remove("visible");
        el.btnLock.disabled = true;
        return;
    }

    el.confidenceBox.classList.add("visible");
    el.detectionBox.classList.add("visible");
    el.detectionBox.style.left = (data.bbox.x * 100) + "%";
    el.detectionBox.style.top = (data.bbox.y * 100) + "%";
    el.detectionBox.style.width = (data.bbox.w * 100) + "%";
    el.detectionBox.style.height = (data.bbox.h * 100) + "%";
    el.btnLock.disabled = false;

    if (data.status === "LOCKED") {
        el.aiStatusText.textContent = "Target locked";
        el.aiStatusBox.classList.remove("scanning");
        el.aiStatusBox.classList.add("detected");
    } else {
        el.aiStatusText.textContent = "Target found - awaiting lock";
        el.aiStatusBox.classList.add("scanning");
        el.aiStatusBox.classList.remove("detected");
    }
}

function updateStatus(status) {
    el.systemStatus.textContent = status;
    el.systemStatus.classList.toggle("active", status === "SCANNING" || status === "LOCKED" || status === "RTL");

    if (status === "SCANNING") {
        el.aiStatusText.textContent = "Scanning...";
        el.aiStatusBox.classList.add("scanning");
        el.aiStatusBox.classList.remove("detected");
    } else if (status === "LOCKED") {
        el.aiStatusText.textContent = "Target locked";
        el.aiStatusBox.classList.remove("scanning");
        el.aiStatusBox.classList.add("detected");
    } else if (status === "PAUSED") {
        el.aiStatusText.textContent = "Paused";
        el.aiStatusBox.classList.remove("scanning", "detected");
    } else if (status === "EMERGENCY") {
        el.aiStatusText.textContent = "Emergency stop";
        el.aiStatusBox.classList.remove("scanning");
        el.aiStatusBox.classList.add("detected");
    } else {
        el.aiStatusText.textContent = "Standby";
        el.aiStatusBox.classList.remove("scanning", "detected");
    }

    const activeMission = status === "SCANNING" || status === "LOCKED";
    el.btnStart.disabled = activeMission;
    el.btnPause.disabled = !activeMission;
    el.btnRtl.disabled = status === "IDLE" || status === "PAUSED";
}

function setupButtons() {
    el.btnStart.onclick = function() {
        sendCommand({ command: "start" });
    };

    el.btnPause.onclick = function() {
        sendCommand({ command: "pause" });
    };

    el.btnLock.onclick = function() {
        sendCommand({ command: "lock_target", targetId: el.targetId.textContent });
    };

    el.btnSnapshot.onclick = function() {
        sendCommand({ command: "snapshot" });
        flashScreen();
    };

    el.btnRecord.onclick = function() {
        isRecording = !isRecording;
        el.recBadge.classList.toggle("active", isRecording);
        sendCommand({ command: "record", enabled: isRecording });
    };

    el.btnRtl.onclick = function() {
        if (confirm("Return to home?")) {
            sendCommand({ command: "rtl" });
        }
    };

    el.btnEmergency.onclick = function() {
        if (confirm("Emergency stop?")) {
            sendCommand({ command: "emergency_stop" });
        }
    };
}

function flashScreen() {
    const overlay = document.createElement("div");
    overlay.style.cssText = "position:absolute;inset:0;background:white;pointer-events:none;opacity:0.8;transition:opacity 0.3s;";
    el.feedContainer.appendChild(overlay);
    setTimeout(() => { overlay.style.opacity = "0"; }, 50);
    setTimeout(() => overlay.remove(), 400);
}

function toast(message, type) {
    const toastNode = document.createElement("div");
    toastNode.className = "toast " + (type || "info");
    toastNode.textContent = message;
    el.toastContainer.appendChild(toastNode);

    setTimeout(function() {
        toastNode.classList.add("hiding");
        setTimeout(() => toastNode.remove(), 300);
    }, 3500);
}

function initMap() {
    if (!window.L || tacticalMap) {
        return;
    }

    tacticalMap = L.map("tactical-map", {
        zoomControl: false,
        attributionControl: true
    }).setView([37.5665, 126.9780], 17);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors"
    }).addTo(tacticalMap);

    userMarker = L.circleMarker([37.5665, 126.9780], {
        radius: 9,
        color: "#53A2FE",
        weight: 2,
        fillColor: "#22C55E",
        fillOpacity: 0.9
    }).addTo(tacticalMap);

    userMarker.bindPopup("Current position");
}

function updateMapLocation(latitude, longitude, accuracy) {
    if (!tacticalMap) {
        return;
    }

    const location = [latitude, longitude];
    const desiredZoom = accuracy && accuracy > 80 ? 16 : 18;
    tacticalMap.setView(location, desiredZoom);

    if (userMarker) {
        userMarker.setLatLng(location);
    }
}

async function reverseGeocode(latitude, longitude) {
    const roundedLat = latitude.toFixed(5);
    const roundedLng = longitude.toFixed(5);
    const cacheKey = `${roundedLat},${roundedLng}`;

    if (cacheKey === lastGeocodeKey) {
        return;
    }
    lastGeocodeKey = cacheKey;

    if (geocodeAbortController) {
        geocodeAbortController.abort();
    }
    geocodeAbortController = new AbortController();

    try {
        const response = await fetch(
            `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${encodeURIComponent(latitude)}&lon=${encodeURIComponent(longitude)}&zoom=18&addressdetails=1`,
            {
                signal: geocodeAbortController.signal,
                headers: {
                    "Accept-Language": "ko,en"
                }
            }
        );

        if (!response.ok) {
            throw new Error(`Geocode failed: ${response.status}`);
        }

        const data = await response.json();
        const addressNode = $("map-coords");
        if (!addressNode) {
            return;
        }

        const address = data.address || {};
        const parts = [
            address.road,
            address.neighbourhood,
            address.suburb,
            address.city_district,
            address.city || address.town || address.village,
            address.state,
            address.country
        ].filter(Boolean);

        addressNode.textContent = parts.length ? parts.join(", ") : (data.display_name || `${roundedLat}, ${roundedLng}`);
    } catch (error) {
        if (error.name === "AbortError") {
            return;
        }
        const addressNode = $("map-coords");
        if (addressNode) {
            addressNode.textContent = `${roundedLat}, ${roundedLng}`;
        }
        console.error("reverse geocode error", error);
    }
}

function startGeolocation() {
    if (!navigator.geolocation) {
        el.gpsIndicator.textContent = "N/A";
        return;
    }

    navigator.geolocation.watchPosition(
        function(pos) {
            const latitude = pos.coords.latitude;
            const longitude = pos.coords.longitude;
            el.gpsIndicator.textContent = Math.round(pos.coords.accuracy) + "m";
            const mapCoords = $("map-coords");
            if (mapCoords) {
                mapCoords.textContent = "Resolving address...";
            }
            updateMapLocation(latitude, longitude, pos.coords.accuracy);
            reverseGeocode(latitude, longitude);
        },
        function() {
            el.gpsIndicator.textContent = "OFF";
            const mapCoords = $("map-coords");
            if (mapCoords) {
                mapCoords.textContent = "Location unavailable";
            }
        },
        {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 5000
        }
    );
}

window.addEventListener("DOMContentLoaded", function() {
    cacheElements();
    setupButtons();
    initMap();
    startGeolocation();
    connectWS();
    toast("Connecting to backend...", "info");
});
