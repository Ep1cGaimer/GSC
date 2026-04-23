// GSC Dashboard Logic
// Integrates Google Maps, Firebase Realtime listeners, and Gemini/KG API calls.

// 1. Firebase Config (Replace with your own)
const firebaseConfig = {
    apiKey: "YOUR_API_KEY",
    projectId: "YOUR_PROJECT_ID",
};

// Initialize Firebase
const app = firebase.initializeApp(firebaseConfig);
const db = firebase.firestore(app);

let map;
let markers = {};
let polylines = {};

// 2. Map Initialization
function initMap() {
    map = new google.maps.Map(document.getElementById("map"), {
        center: { lat: 20.5937, lng: 78.9629 }, // India center
        zoom: 5,
        styles: [
            { "elementType": "geometry", "stylers": [{ "color": "#212121" }] },
            { "elementType": "labels.icon", "stylers": [{ "visibility": "off" }] },
            { "elementType": "labels.text.fill", "stylers": [{ "color": "#757575" }] },
            { "elementType": "labels.text.stroke", "stylers": [{ "color": "#212121" }] },
            { "featureType": "administrative", "elementType": "geometry", "stylers": [{ "color": "#757575" }] },
            { "featureType": "road", "elementType": "geometry.fill", "stylers": [{ "color": "#2c2c2c" }] },
            { "featureType": "water", "elementType": "geometry", "stylers": [{ "color": "#000000" }] }
        ]
    });

    // Start Listeners
    startRealtimeListeners();
}

// 3. Realtime Listeners
function startRealtimeListeners() {
    // Listen for node updates (warehouses, factories)
    db.collection("nodes").onSnapshot((snapshot) => {
        snapshot.docChanges().forEach((change) => {
            const node = change.doc.data();
            const id = change.doc.id;
            
            if (change.type === "added" || change.type === "modified") {
                updateNodeOnMap(id, node);
            }
        });
    });

    // Listen for shield interventions
    db.collection("shield_events")
        .orderBy("timestamp", "desc")
        .limit(15)
        .onSnapshot((snapshot) => {
            const logContainer = document.getElementById("shield-log");
            logContainer.innerHTML = "";
            
            snapshot.forEach((doc) => {
                const evt = doc.data();
                const entry = document.createElement("div");
                entry.className = `log-entry ${evt.intervened ? 'blocked' : 'ok'}`;
                
                const time = new Date(evt.timestamp?.seconds * 1000).toLocaleTimeString();
                entry.innerHTML = `
                    <strong>${time} | ${evt.agent_id}</strong><br>
                    ${evt.intervened ? `🛑 <span style="color: #FF1744">${evt.reason}</span>` : '✅ Action Approved'}
                `;
                logContainer.appendChild(entry);
            });
        });

    // Listen for uncertainty metrics
    db.collection("metrics").doc("conformal").onSnapshot((doc) => {
        if (!doc.exists) return;
        const data = doc.data();
        updateUncertaintyUI(data.width, data.threshold);
    });
}

// 4. UI Update Helpers
function updateNodeOnMap(id, node) {
    const color = node.status === "disrupted" ? "#FF1744" : 
                  node.status === "stressed" ? "#FFD600" : "#00E676";
    
    if (markers[id]) {
        markers[id].setIcon(getDotIcon(color));
    } else {
        markers[id] = new google.maps.Marker({
            position: { lat: node.lat, lng: node.lng },
            map: map,
            icon: getDotIcon(color),
            title: node.name
        });
    }
}

function getDotIcon(color) {
    return {
        path: google.maps.SymbolPath.CIRCLE,
        fillColor: color,
        fillOpacity: 1,
        strokeWeight: 2,
        strokeColor: "#FFFFFF",
        scale: 8
    };
}

function updateUncertaintyUI(width, threshold) {
    const bar = document.getElementById("uncertainty-bar");
    const statusText = document.getElementById("escalation-status");
    
    const percentage = Math.min((width / (threshold * 1.5)) * 100, 100);
    bar.style.width = `${percentage}%`;
    
    if (width > threshold) {
        bar.style.backgroundColor = "#FF1744";
        statusText.textContent = "⚠️ ESCALATING TO HUMAN";
        statusText.className = "status-text escalated";
    } else {
        bar.style.backgroundColor = "#00E676";
        statusText.textContent = "NOMINAL";
        statusText.className = "status-text nominal";
    }
}

// 5. KG Signal Resolver Integration (Mocking API call for demo)
document.getElementById("resolve-btn").addEventListener("click", async () => {
    const text = document.getElementById("signal-text").value;
    if (!text) return;

    const resultsContainer = document.getElementById("kg-results");
    resultsContainer.innerHTML = '<div class="empty-state">Gemini is analyzing...</div>';

    try {
        // In GSC demo, this calls the Cloud Run endpoint running SignalResolver
        const response = await fetch('/api/resolve', {
            method: 'POST',
            body: JSON.stringify({ text })
        });
        const data = await response.json();
        renderKGResults(data);
    } catch (e) {
        // Fallback for demo if backend isn't up
        renderMockKGResults(text);
    }
});

function renderKGResults(data) {
    const resultsContainer = document.getElementById("kg-results");
    resultsContainer.innerHTML = "";
    
    data.results.forEach(res => {
        const card = document.createElement("div");
        card.className = "kg-result-card";
        card.innerHTML = `
            <div class="result-header">
                <strong>${res.entity.name}</strong>
                <span class="entity-type">${res.entity.type}</span>
            </div>
            <p style="font-size: 0.8rem; margin-bottom: 8px;">${res.entity.description}</p>
            <div class="protocols">
                ${res.protocols.map(p => `
                    <div style="margin-top: 4px;">
                        <span style="font-weight: bold; font-size: 0.8rem;">${p.protocol} (${p.severity}):</span><br>
                        ${p.actions.map(a => `<span class="protocol-action">${a}</span>`).join('')}
                    </div>
                `).join('')}
            </div>
        `;
        resultsContainer.appendChild(card);
    });
}

function renderMockKGResults(text) {
    // Demo fallback logic
    const mockData = {
        results: [{
            entity: { name: "ChemicalSpill", type: "HAZARD", description: "Spill detected at Mumbai Port" },
            protocols: [{ protocol: "HazmatRouting", severity: "CRITICAL", actions: ["reroute", "alert"] }]
        }]
    };
    renderKGResults(mockData);
}

// Disruption Injection Button
document.getElementById("inject-disruption-btn").addEventListener("click", () => {
    // Sends command to environment via API
    fetch('/api/disrupt', { method: 'POST' });
});
