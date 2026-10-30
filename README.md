# CivicAction: From Complaints to Action

**promptwarz hackathon project**

AI-assisted civic complaint platform. Citizens report issues; the system classifies, summarises, de-duplicates and prioritises them; authorities see hotspots and move each complaint through a tracked workflow.

## Run
```
cd backend
python server.py        # Python 3.9+, no pip install needed
```
Open http://localhost:8000 (set PORT to change). The database `backend/civic.db` is created and seeded with 42 fictional complaints on first run; delete the file to reset.
The browser needs internet access for Leaflet, Chart.js, the map tiles (OpenStreetMap) and the font.

**Demo accounts (prototype only):** citizen `demo@citizen.local` / `citizen123`, authority `admin@civic.local` / `admin123`.

## Login and role separation
One login page with Citizen and Authority tabs. The server checks the account's role on every request: citizens only ever see their own complaints, and authority endpoints (stats, assign, update status) return 403 for citizens. Authority accounts cannot be self-registered; citizens can.

## Structure
- `backend/server.py`: HTTP API, SQLite schema (users, sessions, complaints, updates), AI pipeline, seed data
- `frontend/index.html`: single-page app (vanilla JS, Leaflet, Chart.js)

## AI implementation (all computed in `server.py`)
| Feature | Method |
|---|---|
| Classification | Keyword lexicon per category; hits weighted by keyword length; confidence = share of weighted hits (rule-based, not a trained model) |
| Summary | Extractive: keeps up to two sentences with the most safety/impact/severity cues |
| Duplicate detection | TF-IDF cosine similarity (pure Python) >= 0.30 AND distance <= 400 m (haversine); duplicates link to the original report |
| Priority + explanation | Transparent 0-100 score: category baseline + safety words + public-impact words + severity words + nearby similar reports. High >= 60, Medium >= 35. The explanation lists every contribution |
| Hotspots | Grid clustering (~450 m cells), cells with 3+ reports; plus recurring (area, category) pairs |
| Action brief | Template filled from the complaint's own data (priority, department, response target, area repeat count, duplicate link) |

No external AI service is called. The "AI" is classical NLP and rules by design, so every result is explainable and reproducible.

## 5-minute demo
1. Sign in as the citizen, Submit complaint: "Severe waterlogging on the main road after rain. Buses cannot pass and residents are stranded." Use "Jump map to area" > Riverside and click to drop a pin.
2. Read the result card: category, summary, duplicate link, High priority with reasons.
3. Sign out, sign in on the Authority tab. Dashboard shows the new complaint in the totals; Map & hotspots shows the Riverside drainage hotspot; AI insights lists it in the priority queue and duplicate groups.
4. Complaints > open the new ID, set department, status "Assigned" with a note, then In Progress, then Resolved.
5. Sign back in as the citizen: Dashboard "Latest updates" and the complaint timeline show every step.

## Limitations and future work
- Classifier and priority are rule-based; with real labelled data, replace with scikit-learn or sentence embeddings (`classify`, `tfidf_sim` are isolated for that).
- Image is stored and shown but not analysed. No email/SMS notifications (updates appear in-app only).
- No password reset, no HTTPS, sessions never expire. Prototype security only; do not deploy as is.
- SQLite only; schema is plain SQL and ports to PostgreSQL.
- Locations are a fictional six-ward city; edit `ZONES` in `server.py` to change.
- Frontend is vanilla JS rather than React/Tailwind, to keep setup to one command.
