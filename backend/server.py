#!/usr/bin/env python3
"""CivicAction - From Complaints to Action. Stdlib-only backend (Python 3.9+). Run: python server.py"""
import sqlite3, json, hashlib, os, re, math, secrets, random
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "civic.db")
INDEX = os.path.join(HERE, "..", "frontend", "index.html")
ST = ["Submitted", "Under Review", "Assigned", "In Progress", "Resolved"]
# Fictional, generalised city: wards around a neutral centre point
ZONES = {"Central District": (20.000, 78.000), "North Ward": (20.030, 78.002), "South Ward": (19.970, 77.998),
         "East Ward": (20.002, 78.035), "West Ward": (19.998, 77.965), "Riverside": (19.980, 78.025)}
DEPT = {"Roads & Potholes": "Public Works Dept", "Drainage & Waterlogging": "Drainage & Sewerage Board",
        "Garbage & Waste": "Solid Waste Management", "Streetlights": "Electrical & Lighting Dept",
        "Water Supply": "Water Authority", "Public Sanitation": "Public Health & Sanitation", "Other": "General Services"}
BASE = {"Roads & Potholes": 30, "Drainage & Waterlogging": 30, "Garbage & Waste": 20, "Streetlights": 25,
        "Water Supply": 30, "Public Sanitation": 25, "Other": 15}
KW = {"Roads & Potholes": ["pothole", "road", "crater", "asphalt", "pavement", "speed breaker", "tarmac", "crack"],
      "Drainage & Waterlogging": ["waterlog", "flood", "drain", "sewer", "sewage", "stagnant", "manhole", "gutter"],
      "Garbage & Waste": ["garbage", "trash", "waste", "bin", "dump", "litter", "rubbish", "stink"],
      "Streetlights": ["streetlight", "street light", "lamp", "dark", "bulb", "pole", "lighting"],
      "Water Supply": ["water supply", "tap", "pipeline", "no water", "pipe", "pressure", "contaminated"],
      "Public Sanitation": ["toilet", "sanitation", "mosquito", "hygiene", "restroom", "urinal", "dengue"]}
SAFE = ["accident", "injur", "danger", "fatal", "child", "school", "hospital", "electric", "live wire", "open manhole",
        "collapse", "fire", "disease", "dengue", "elderly", "hazard", "unsafe"]
IMP = ["main road", "highway", "market", "junction", "bus", "entire", "whole", "residents", "hundreds", "commut",
       "traffic", "days", "week"]
SEV = ["severe", "huge", "serious", "major", "blocked", "completely", "urgent", "deep"]
STOP = set("the a an and or of to in on at is are was it this that for with near from has have been very there our we i my by be as not but".split())
SCHEMA = """CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY,name,email UNIQUE,pw,role,area,phone);
CREATE TABLE IF NOT EXISTS sessions(token PRIMARY KEY,uid);
CREATE TABLE IF NOT EXISTS complaints(id INTEGER PRIMARY KEY,cid UNIQUE,uid,description,category,ai_category,confidence REAL,summary,priority,score INTEGER,reason,lat REAL,lng REAL,area,status,dept,notes,dup_of,dup_score REAL,image,created,updated);
CREATE TABLE IF NOT EXISTS updates(id INTEGER PRIMARY KEY,cid,status,note,ts);"""


class Err(Exception):
    def __init__(s, m, code=400): super().__init__(m); s.code = code


def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c


def now(): return datetime.now().isoformat(timespec="seconds")


def hp(p, salt=None):
    salt = salt or secrets.token_hex(8)
    return salt + "$" + hashlib.pbkdf2_hmac("sha256", p.encode(), salt.encode(), 50000).hex()


def chk(p, h): return secrets.compare_digest(hp(p, h.split("$")[0]), h)


# ---------------------------------------------------------------- AI / NLP
def toks(t): return [w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP and len(w) > 2]


def hav(a, b, c, d):
    p = math.pi / 180
    x = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return 12742000 * math.asin(math.sqrt(x))


def zone(lat, lng): return min(ZONES, key=lambda z: hav(lat, lng, *ZONES[z]))


def tfidf_sim(q, docs):
    """Cosine similarity between q and each doc using TF-IDF computed over all texts."""
    allt = [toks(d) for d in docs] + [toks(q)]
    N = len(allt); df = Counter(w for t in allt for w in set(t))
    vec = lambda t: {w: n * (math.log((N + 1) / (df[w] + 1)) + 1) for w, n in Counter(t).items()}
    qv = vec(allt[-1]); qn = math.sqrt(sum(v * v for v in qv.values())) or 1; out = []
    for t in allt[:-1]:
        v = vec(t); n = math.sqrt(sum(x * x for x in v.values())) or 1
        out.append(sum(qv[w] * v.get(w, 0) for w in qv) / (qn * n))
    return out


def classify(t):
    """Keyword-lexicon classifier: hits weighted by keyword length (specific terms like 'waterlog' beat generic ones like 'road');
    confidence = category's share of all weighted hits."""
    t = t.lower(); s = {c: sum(len(k) * len(re.findall(r"\b" + k, t)) for k in ks) for c, ks in KW.items()}
    b = max(s, key=s.get)
    return ("Other", 0.0) if s[b] == 0 else (b, round(s[b] / sum(s.values()), 2))


def summ(t):
    """Extractive summary: keeps up to two sentences with the most safety/impact/severity cues."""
    ss = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]
    if len(ss) <= 1: return t[:160] + ("..." if len(t) > 160 else "")
    key = SAFE + IMP + SEV
    top = sorted(range(len(ss)), key=lambda i: -(sum(k in ss[i].lower() for k in key) + (2 if i == 0 else 0)))[:2]
    return " ".join(ss[i] for i in sorted(top))[:200]


def prio(t, cat, near):
    """Transparent rule-based score (0-100); every point is listed in the explanation."""
    t = t.lower(); s = BASE[cat]; why = [f"{cat.lower()} baseline (+{s})"]
    for name, ks, w, cap in (("safety risk", SAFE, 20, 35), ("public impact", IMP, 8, 20), ("severity wording", SEV, 10, 15)):
        h = [k for k in ks if k in t]
        if h: a = min(cap, w * len(h)); s += a; why.append(f"{name}: {', '.join(h[:3])} (+{a})")
    if near: a = min(15, 5 * near); s += a; why.append(f"{near} similar report(s) within 500 m, recurring issue (+{a})")
    s = min(s, 100); p = "High" if s >= 60 else "Medium" if s >= 35 else "Low"
    return s, p, f"{p} Priority ({s}/100): " + "; ".join(why) + "."


def analyse(c, desc, cat, lat, lng):
    ai, conf = classify(desc); cat = cat or ai
    rows = c.execute("select cid,description,lat,lng,category,dup_of from complaints").fetchall()
    dup, ds = None, 0
    if rows:
        for r, s in zip(rows, tfidf_sim(desc, [r["description"] for r in rows])):
            if hav(lat, lng, r["lat"], r["lng"]) <= 400 and s >= .3 and s > ds: dup, ds = (r["dup_of"] or r["cid"]), s
    near = sum(1 for r in rows if r["category"] == cat and hav(lat, lng, r["lat"], r["lng"]) <= 500)
    sc, pr, why = prio(desc, cat, near)
    return dict(category=cat, ai_category=ai, confidence=conf, summary=summ(desc), priority=pr, score=sc, reason=why,
                dup_of=dup, dup_score=round(ds, 2))


def brief(r, n_area):
    sla = {"High": "24 hours", "Medium": "72 hours", "Low": "7 days"}[r["priority"]]
    b = f"{r['priority']} priority {r['category']} issue in {r['area']}. {r['summary']} Recommended: dispatch {DEPT[r['category']]} within {sla}."
    if n_area > 2: b += f" {n_area} open reports of this type in the area: inspect for a root cause, not a one-off fix."
    if r["dup_of"]: b += f" Likely duplicate of {r['dup_of']}: handle together and notify both citizens."
    return b


# ---------------------------------------------------------------- data ops
def add(c, uid, desc, cat, lat, lng, img=None, ts=None):
    a = analyse(c, desc, cat, lat, lng); ts = ts or now()
    cid = "CMP-%d" % (1001 + c.execute("select count(*) from complaints").fetchone()[0])
    c.execute("insert into complaints(cid,uid,description,category,ai_category,confidence,summary,priority,score,reason,lat,lng,area,status,dept,notes,dup_of,dup_score,image,created,updated) values(" + ",".join("?" * 21) + ")",
              (cid, uid, desc, a["category"], a["ai_category"], a["confidence"], a["summary"], a["priority"], a["score"], a["reason"],
               lat, lng, zone(lat, lng), "Submitted", "", "", a["dup_of"], a["dup_score"], img, ts, ts))
    c.execute("insert into updates(cid,status,note,ts) values(?,?,?,?)", (cid, "Submitted", "Complaint received and analysed by AI.", ts))
    return cid


def move(c, cid, status, note="", dept=None, ts=None):
    ts = ts or now()
    c.execute("update complaints set status=?,updated=?,dept=coalesce(?,dept),notes=case when ?!='' then ? else notes end where cid=?", (status, ts, dept, note, note, cid))
    c.execute("insert into updates(cid,status,note,ts) values(?,?,?,?)", (cid, status, note or "Status changed to " + status, ts))


TPL = {"Roads & Potholes": ["Large pothole on the main road near the junction has caused two bike accidents. It is very dangerous at night.",
                            "Deep pothole blocking one lane of the road. Traffic is slow every day and vehicles are getting damaged.",
                            "Broken pavement and cracked road along the market, unsafe for elderly pedestrians."],
       "Drainage & Waterlogging": ["Severe waterlogging on the main road after every rain. Buses cannot pass and residents are stranded.",
                                   "Drain is blocked and sewage is overflowing into the street near the school.",
                                   "Stagnant water has collected for days and an open manhole is a danger to children."],
       "Garbage & Waste": ["Garbage bin overflowing for a week near the market. Terrible smell and stray animals.",
                           "Waste is dumped on the roadside and not collected. Residents are complaining.",
                           "Trash piled up at the corner, bins not cleared since last week."],
       "Streetlights": ["Streetlight not working on the main road, the whole stretch is dark and unsafe at night.",
                        "Several street lamps are broken near the park. Women and elderly feel unsafe.",
                        "Lamp pole bulb fused, street is dark for days."],
       "Water Supply": ["No water supply in our locality for three days. Residents are buying tanker water.",
                        "Pipeline leak near the junction is wasting water and pressure is very low.",
                        "Tap water is contaminated and smells bad, dangerous for children."],
       "Public Sanitation": ["Public toilet is filthy and unusable, poor hygiene near the bus stand.",
                             "Mosquito breeding in the open area, dengue risk for residents.",
                             "Urinal blocked and toilet unhygienic near the market."]}
NOTES = ["", "Verified by field officer.", "Assigned to the responsible department.", "Crew is on site.", "Work completed and verified."]


def seed(c):
    random.seed(7)
    for n, e, pw, r in (("Authority Officer", "admin@civic.local", "admin123", "admin"), ("Demo Citizen", "demo@citizen.local", "citizen123", "citizen"),
                        ("Sam Rivers", "sam@citizen.local", "citizen123", "citizen"), ("Lena Ortiz", "lena@citizen.local", "citizen123", "citizen")):
        c.execute("insert into users(name,email,pw,role,area,phone) values(?,?,?,?,?,?)", (n, e, hp(pw), r, "Central District", ""))
    jobs = []
    for cat, area, n in (("Drainage & Waterlogging", "Riverside", 6), ("Roads & Potholes", "East Ward", 6), ("Garbage & Waste", "Central District", 5),
                         ("Streetlights", "North Ward", 4), ("Water Supply", "West Ward", 4), ("Public Sanitation", "South Ward", 3)):
        for i in range(n): jobs.append((random.randint(1, 30), cat, area, i % 3, .0012))
    for i in range(14): jobs.append((random.randint(1, 30), random.choice(list(TPL)), random.choice(list(ZONES)), random.randint(0, 2), .01))
    jobs.sort(reverse=True)
    for age, cat, area, v, j in jobs:
        la, ln = ZONES[area]
        if j < .01: la, ln = (math.floor(la / .004) + .5) * .004, (math.floor(ln / .004) + .5) * .004  # centre of a hotspot grid cell
        ts = (datetime.now() - timedelta(days=age, hours=random.randint(0, 10))).replace(microsecond=0)
        cid = add(c, random.randint(2, 4), TPL[cat][v], cat, la + random.uniform(-j, j), ln + random.uniform(-j, j), ts=ts.isoformat())
        for k in range(1, random.choices(range(5), [3, 2, 2, 2, 1 + age // 4])[0] + 1):
            move(c, cid, ST[k], NOTES[k], DEPT[cat] if k >= 2 else None, (ts + timedelta(hours=5 * k)).isoformat())


def strip(r):
    d = dict(r); d["has_image"] = bool(d.pop("image")); return d


def pub(r): return {k: r[k] for k in ("id", "name", "email", "role", "area", "phone")}


def stats(c):
    R = [dict(r) for r in c.execute("select * from complaints")]
    for r in R: r.pop("image")
    op = [r for r in R if r["status"] != "Resolved"]; cells = defaultdict(list)
    for r in R: cells[(math.floor(r["lat"] / .004), math.floor(r["lng"] / .004))].append(r)
    hot = sorted([{"lat": sum(x["lat"] for x in v) / len(v), "lng": sum(x["lng"] for x in v) / len(v), "count": len(v),
                   "open": sum(x["status"] != "Resolved" for x in v), "category": Counter(x["category"] for x in v).most_common(1)[0][0],
                   "area": zone(v[0]["lat"], v[0]["lng"])} for v in cells.values() if len(v) >= 3], key=lambda h: -h["count"])
    rec = [{"area": a, "category": k, "count": n} for (a, k), n in Counter((r["area"], r["category"]) for r in R).most_common() if n >= 3]
    g = defaultdict(list); by = {r["cid"]: r for r in R}
    for r in R:
        if r["dup_of"]: g[r["dup_of"]].append(r["cid"])
    dups = [{"root": k, "summary": by[k]["summary"], "area": by[k]["area"], "duplicates": v} for k, v in g.items() if k in by]
    hrs = [(datetime.fromisoformat(r["updated"]) - datetime.fromisoformat(r["created"])).total_seconds() / 3600 for r in R if r["status"] == "Resolved"]
    days = Counter(r["created"][:10] for r in R)
    trend = [[d, days.get(d, 0)] for d in [(datetime.now() - timedelta(days=i)).date().isoformat() for i in range(30, -1, -1)]]
    cnt = Counter((r["area"], r["category"]) for r in op); top = sorted(op, key=lambda r: -r["score"])[:8]
    for r in top: r["brief"] = brief(r, cnt[(r["area"], r["category"])])
    cn = lambda k: dict(Counter(r[k] for r in R))
    return {"total": len(R), "pending": len(op), "resolved": len(R) - len(op), "high": sum(r["priority"] == "High" for r in op),
            "avg_hours": round(sum(hrs) / len(hrs), 1) if hrs else 0, "dup_count": sum(1 for r in R if r["dup_of"]),
            "by_category": cn("category"), "by_area": cn("area"), "by_status": cn("status"), "trend": trend,
            "hotspots": hot, "recurring": rec, "duplicates": dups, "top": top}


# ---------------------------------------------------------------- API
def need(u, role=None):
    if not u: raise Err("Please sign in", 401)
    if role and u["role"] != role: raise Err("This action is not available for your account type", 403)


def api(c, m, p, q, b, u):
    if p == "/api/meta": return {"zones": {k: {"lat": v[0], "lng": v[1]} for k, v in ZONES.items()}, "depts": sorted(set(DEPT.values()))}
    if p == "/api/login" and m == "POST":
        r = c.execute("select * from users where email=?", ((b.get("email") or "").lower().strip(),)).fetchone()
        if not r or not chk(b.get("password") or "", r["pw"]): raise Err("Invalid email or password", 401)
        if r["role"] != b.get("role"):
            raise Err("This is an authority account. Use the Authority tab." if r["role"] == "admin" else "This is a citizen account. Use the Citizen tab.", 403)
        t = secrets.token_hex(16); c.execute("insert into sessions values(?,?)", (t, r["id"])); return {"token": t, "user": pub(r)}
    if p == "/api/register" and m == "POST":
        n, e, pw = (b.get("name") or "").strip(), (b.get("email") or "").lower().strip(), b.get("password") or ""
        if not n or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", e) or len(pw) < 6: raise Err("Enter a name, a valid email and a password of 6+ characters")
        if c.execute("select 1 from users where email=?", (e,)).fetchone(): raise Err("An account with this email already exists")
        i = c.execute("insert into users(name,email,pw,role,area,phone) values(?,?,?,?,?,?)", (n[:80], e, hp(pw), "citizen", "Central District", "")).lastrowid
        t = secrets.token_hex(16); c.execute("insert into sessions values(?,?)", (t, i))
        return {"token": t, "user": pub(c.execute("select * from users where id=?", (i,)).fetchone())}
    if p == "/api/me":
        need(u)
        if m == "PUT":
            if b.get("area") not in ZONES: raise Err("Choose a valid area")
            if not (b.get("name") or "").strip(): raise Err("Name is required")
            c.execute("update users set name=?,phone=?,area=? where id=?", (b["name"].strip()[:80], (b.get("phone") or "")[:20], b["area"], u["id"]))
            u = c.execute("select * from users where id=?", (u["id"],)).fetchone()
        return pub(u)
    if p == "/api/updates":
        need(u, "citizen")
        return [dict(r) for r in c.execute("select x.cid,x.status,x.note,x.ts from updates x join complaints k on k.cid=x.cid where k.uid=? order by x.id desc limit 6", (u["id"],))]
    if p == "/api/stats": need(u, "admin"); return stats(c)
    if p == "/api/complaints" and m == "GET":
        need(u); w, a = [], []
        if u["role"] == "citizen": w.append("uid=?"); a.append(u["id"])
        for k in ("category", "area", "priority", "status"):
            if q.get(k): w.append(k + "=?"); a.append(q[k])
        if q.get("q"): w.append("(description like ? or cid like ?)"); a += ["%" + q["q"] + "%"] * 2
        if q.get("dup"): w.append("dup_of is not null")
        return [strip(r) for r in c.execute("select * from complaints" + (" where " + " and ".join(w) if w else "") + " order by created desc", a)]
    if p == "/api/complaints" and m == "POST":
        need(u, "citizen"); d = (b.get("description") or "").strip()
        if len(d) < 15: raise Err("Describe the problem in at least 15 characters")
        try:
            lat, lng = float(b["lat"]), float(b["lng"]); assert -90 <= lat <= 90 and -180 <= lng <= 180
        except Exception: raise Err("Drop a pin on the map to set the location")
        cat = b.get("category") or None
        if cat and cat not in DEPT: raise Err("Unknown category")
        img = b.get("image"); img = img if isinstance(img, str) and img.startswith("data:image/") and len(img) < 900000 else None
        cid = add(c, u["id"], d[:2000], cat, lat, lng, img)
        return strip(c.execute("select * from complaints where cid=?", (cid,)).fetchone())
    mt = re.match(r"^/api/complaints/(CMP-\d+)$", p)
    if mt:
        need(u); r = c.execute("select * from complaints where cid=?", (mt[1],)).fetchone()
        if not r or (u["role"] == "citizen" and r["uid"] != u["id"]): raise Err("Complaint not found", 404)
        if m == "PUT":
            need(u, "admin"); s = b.get("status"); dp = b.get("dept") or None
            if s not in ST: raise Err("Choose a valid status")
            if dp and dp not in DEPT.values(): raise Err("Unknown department")
            if s in ST[2:] and not (dp or r["dept"]): raise Err("Assign a department before moving to this status")
            move(c, mt[1], s, (b.get("note") or "").strip()[:500], dp); r = c.execute("select * from complaints where cid=?", (mt[1],)).fetchone()
        d = dict(r); d["updates"] = [dict(x) for x in c.execute("select status,note,ts from updates where cid=? order by id", (mt[1],))]
        if u["role"] == "admin":
            n = c.execute("select count(*) from complaints where area=? and category=? and status!='Resolved'", (d["area"], d["category"])).fetchone()[0]
            d["brief"] = brief(d, n); d["suggested"] = DEPT[d["category"]]
            d["reporter"] = c.execute("select name from users where id=?", (d["uid"],)).fetchone()[0]
        return d
    raise Err("Not found", 404)


class H(BaseHTTPRequestHandler):
    def log_message(s, *a): pass

    def out(s, body, code=200, ct="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        s.send_response(code); s.send_header("Content-Type", ct); s.send_header("Content-Length", str(len(b))); s.end_headers(); s.wfile.write(b)

    def do_GET(s): s.route("GET")
    def do_POST(s): s.route("POST")
    def do_PUT(s): s.route("PUT")

    def route(s, m):
        u = urlparse(s.path); p = u.path
        if m == "GET" and not p.startswith("/api"): return s.out(open(INDEX, "rb").read(), ct="text/html; charset=utf-8")
        q = {k: v[0] for k, v in parse_qs(u.query).items()}; n = int(s.headers.get("Content-Length") or 0)
        c = db()
        try:
            b = json.loads(s.rfile.read(n) or b"{}") if n else {}
            t = s.headers.get("Authorization", "")[7:]
            usr = c.execute("select u.* from users u join sessions x on x.uid=u.id where x.token=?", (t,)).fetchone()
            res = api(c, m, p, q, b, usr); c.commit(); s.out(res)
        except Err as e: s.out({"error": str(e)}, e.code)
        except Exception as e: print("ERROR", repr(e)); s.out({"error": "Server error"}, 500)
        finally: c.close()


if __name__ == "__main__":
    fresh = not os.path.exists(DB); c = db(); c.executescript(SCHEMA)
    if fresh: seed(c); c.commit()
    port = int(os.environ.get("PORT", 8000)); print(f"CivicAction running at http://localhost:{port}")
    ThreadingHTTPServer(("", port), H).serve_forever()
