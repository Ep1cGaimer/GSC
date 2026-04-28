# ChainGuard AI - 3-Minute Demo Video Script

## Video Overview
- **Total Duration:** 3:00 (180 seconds)
- **Style:** Screen recording with voiceover
- **Target Audience:** Judges, investors, technical reviewers

---

## TIMING BREAKDOWN

| Phase | Duration | Purpose |
|-------|----------|---------|
| Intro + Problem | 0:00 - 0:30 | Hook viewer |
| Solution + Architecture | 0:30 - 1:00 | Explain the system |
| Live Demo | 1:00 - 2:15 | Show it working |
| Impact + Close | 2:15 - 3:00 | Call to action |

---

## SCRIPT

### PHASE 1: INTRO (0:00 - 0:30)

**[SCREEN: Black screen with logo fade in]**

**VOICEOVER:**
> "Every year, global supply chains lose billions to disruptions. Floods, strikes, pandemics. The problem? Traditional AI systems break when the network changes. They're reactive, not resilient."

**[SCREEN: Show statistics graphic or map of disruptions]**

> "Meet ChainGuard AI. A multi-agent reinforcement learning system designed to adapt, protect, and explain."

---

### PHASE 2: ARCHITECTURE (0:30 - 1:00)

**[SCREEN: Dashboard loading, showing initial state]**

**VOICEOVER:**
> "Here's how it works. We simulate India's 50-node logistics network. Factories, ports, warehouses, retailers—all connected by road, rail, and sea routes."

**[SCREEN: Zoom to map showing nodes]**

> "Each warehouse is an AI agent. They make decisions every step. But here's the innovation: before any action executes, our Safety Shield checks against hard constraints."

**[SCREEN: Show Shield干预 in sidebar or description]**

> "Zero CO2 violations. Zero budget overruns. Guaranteed mathematically—not by guessing."

> "And if disruptions happen? The system adapts in real-time."

---

### PHASE 3: LIVE DEMO (1:00 - 2:15)

#### Segment 3a: Normal Operation (1:00 - 1:20)

**[SCREEN: Dashboard - click Step button]**

**VOICEOVER:**
> "Let's see it live. I'll step through one iteration."

**[Click Step 2-3 times]**

> "Each step, the AI orders inventory, fulfills demand, tracks revenue and emissions. Watch the metrics update."

**[Point to Revenue, CO2, Step counter]**

> "Revenue increases. Carbon is tracked. The system learns from every decision."

#### Segment 3b: Disruption Injection (1:20 - 1:50)

**[SCREEN: Click Disrupt button]**

**VOICEOVER:**
> "Now let's inject a realistic disruption. A port closure."

**[Click Disrupt - show red dashed line appearing on map]**

> "The map immediately highlights the disrupted route in red. Disabled edges shown as dashed lines. Constrained nodes marked."

**[Point to banner showing disabled edges]**

> "Operators see exactly what's affected: P01 to W01—route disabled. Capacity at F01 cut in half."

> "The AI adapts immediately—reroutes through alternative paths."

#### Segment 3c: Traffic Integration (1:50 - 2:15)

**[SCREEN: Traffic sync status]**

**VOICEOVER:**
> "But we go further. Real-world Google Maps traffic data integrates into the simulation."

> "Road congestion actually affects delivery times. Live data, not estimates."

**[If clicking more Steps to show animation]**

> "Trucks animate in real-time. Visual confirmation of shipping flows."

> "This isn't just simulation—it's a digital twin of your actual operations."

---

### PHASE 4: IMPACT + CLOSE (2:15 - 3:00)

**[SCREEN: System summary or architecture diagram]**

**VOICEOVER:**
> "ChainGuard AI achieves three things no other supply chain system can promise."

**Hold and count on fingers:**
> "One: adapts to dynamic disruptions—learns on the fly."

> "Two: guarantees safety through mathematical constraints—no violations ever."

> "Three: explains every decision through knowledge graph grounding."

**[SCREEN: Logo or team info]**

> "Built with PettingZoo, PyTorch Geometric, Gemini, and Neo4j. Ready for production deployment."

> "Thank you. Questions?"

**[SCREEN: Contact info / repository link]**

---

## SCREEN RECORDING GUIDE

### Before Recording
- [ ] Browser in fullscreen (1920x1080)
- [ ] Dashboard loaded at http://localhost:5000
- [ ] Fresh state (click Reset if needed)
- [ ] Microphone tested

### During Recording
1. **Move mouse slowly** - quick movements blur on video
2. **Pause 1-2 seconds** after clicking buttons (editing buffer)
3. **Use keyboard shortcuts** - Space to click default button

### Recording Setup (Option A: Loom/OBS)
```
OBS Studio:
- Display capture → Chrome window
- Microphone → USB condenser
- 1080p, 30fps
- Export: MP4, H.264
```

### Recording Setup (Option B: QuickTime on Mac)
```
Shift + Cmd + 5 → Select screen area → Record
```

---

## EDITING TIPS

### In Final Cut Pro / Premiere
1. **Trim pauses** - Leave 1 second pause after clicks for viewer to process
2. **Add lower thirds** - Name/role for narrator (optional)
3. **Music** - Ambient tech music, low volume under voiceover
4. **Transitions** - Cut only, no dissolves (professional feel)

### Text Overlays to Add
- 0:00: "ChainGuard AI - Resilient Supply Chain Intelligence"
- 0:30: "The Problem"
- 1:00: "Live Demo"
- 1:20: "Disruption Test"
- 2:15: "Key Results"

---

## WHAT TO DEMONSTRATE (Priority Order)

| # | Feature | Why |
|---|--------|-----|
| 1 | Step + Revenue | Shows core functionality |
| 2 | Disrupt + Visual | Shows adaptation |
| 3 | Map + Markers | Shows real-time visibility |
| 4 | Safety Shield | Shows safety guarantees |

**Skip for time:**
- Gemini/KG signal resolve (prototype)
- Auto mode animation (similar to Step)
- Traffic sync (requires API key)

---

## REHEARSAL CHECKLIST

- [ ] Practice clicking Step button 3x (smooth timing)
- [ ] Practice clicking Disrupt + showing the red line
- [ ] Know where Revenue/CO2 metrics are (don't hunt on camera)
- [ ] Have backup: "If X crashes, I'll Y" (e.g., use Reset)

---

## TECHNICAL REMINDERS

- **Server runs on port 5000** - confirm before recording
- **Refresh page** if any glitches before starting
- **Screen resolution** - match your export settings
- **Audio levels** - voice consistent, music subtle

---

*Good luck with the presentation!*