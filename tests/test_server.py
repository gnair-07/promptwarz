"""Automated tests for CivicAction. Run from the project root:  python -m unittest discover -s tests -v"""
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
import server as s


def call(c, m, p, body=None, user=None, q=None): return s.api(c, m, p, q or {}, body or {}, user)


class AITests(unittest.TestCase):
    def test_classifies_each_category(self):
        cases = {"Big pothole on the road near the junction": "Roads & Potholes",
                 "Severe waterlogging on the main road after rain": "Drainage & Waterlogging",
                 "Garbage bin overflowing near the market": "Garbage & Waste",
                 "Streetlight not working, street is dark": "Streetlights",
                 "No water supply for three days": "Water Supply",
                 "Public toilet is unhygienic": "Public Sanitation"}
        for text, cat in cases.items(): self.assertEqual(s.classify(text)[0], cat, text)

    def test_unknown_text_is_other_with_zero_confidence(self):
        self.assertEqual(s.classify("hello there"), ("Other", 0.0))

    def test_priority_levels_and_explanation(self):
        sc, p, why = s.prio("Pothole caused an accident near the school on the main road, very dangerous", "Roads & Potholes", 0)
        self.assertEqual(p, "High"); self.assertIn("safety risk", why)
        self.assertEqual(s.prio("Bin not cleared", "Garbage & Waste", 0)[1], "Low")

    def test_recurrence_raises_score(self):
        self.assertGreater(s.prio("bin not cleared", "Garbage & Waste", 3)[0], s.prio("bin not cleared", "Garbage & Waste", 0)[0])

    def test_score_capped_at_100(self):
        self.assertLessEqual(s.prio("accident danger fatal child school hospital severe huge main road highway market " * 3, "Roads & Potholes", 9)[0], 100)

    def test_tfidf_similarity(self):
        sims = s.tfidf_sim("pothole on main road", ["pothole on main road", "garbage bin overflowing"])
        self.assertGreater(sims[0], .9); self.assertLess(sims[1], .1)

    def test_haversine_distance(self):
        self.assertLess(abs(s.hav(20, 78, 20, 78)), 1)
        self.assertAlmostEqual(s.hav(20, 78, 20.01, 78), 1112, delta=15)

    def test_summary_is_short(self):
        self.assertLessEqual(len(s.summ("Big problem here. " * 40)), 200)


class APITests(unittest.TestCase):
    def setUp(self):
        self.f = tempfile.NamedTemporaryFile(suffix=".db", delete=False); self.f.close(); s.DB = self.f.name
        self.c = s.db(); self.c.executescript(s.SCHEMA); s.seed(self.c); self.c.commit()
        row = lambda e: self.c.execute("select * from users where email=?", (e,)).fetchone()
        self.admin, self.cit = row("admin@civic.local"), row("demo@citizen.local")

    def tearDown(self): self.c.close(); os.unlink(self.f.name)

    def test_seed_data(self):
        n = self.c.execute("select count(*) from complaints").fetchone()[0]
        self.assertGreaterEqual(n, 30); self.assertGreater(len(call(self.c, "GET", "/api/stats", user=self.admin)["hotspots"]), 0)

    def test_login_role_separation(self):
        with self.assertRaises(s.Err) as e: call(self.c, "POST", "/api/login", {"email": "admin@civic.local", "password": "admin123", "role": "citizen"})
        self.assertEqual(e.exception.code, 403)
        self.assertIn("token", call(self.c, "POST", "/api/login", {"email": "admin@civic.local", "password": "admin123", "role": "admin"}))

    def test_wrong_password_rejected(self):
        with self.assertRaises(s.Err) as e: call(self.c, "POST", "/api/login", {"email": "demo@citizen.local", "password": "nope", "role": "citizen"})
        self.assertEqual(e.exception.code, 401)

    def test_register_then_login_and_duplicate_email(self):
        b = {"name": "Test User", "email": "New@Test.com", "password": "secret1"}
        call(self.c, "POST", "/api/register", b)
        self.assertIn("token", call(self.c, "POST", "/api/login", {"email": "new@test.com", "password": "secret1", "role": "citizen"}))
        with self.assertRaises(s.Err): call(self.c, "POST", "/api/register", b)

    def test_register_validation(self):
        with self.assertRaises(s.Err): call(self.c, "POST", "/api/register", {"name": "A", "email": "bad", "password": "123"})

    def test_citizen_cannot_use_admin_endpoints(self):
        with self.assertRaises(s.Err) as e: call(self.c, "GET", "/api/stats", user=self.cit)
        self.assertEqual(e.exception.code, 403)
        with self.assertRaises(s.Err): call(self.c, "GET", "/api/stats")

    def test_citizen_sees_only_own_complaints(self):
        mine = call(self.c, "GET", "/api/complaints", user=self.cit)
        self.assertTrue(all(x["uid"] == self.cit["id"] for x in mine))
        other = self.c.execute("select cid from complaints where uid!=? limit 1", (self.cit["id"],)).fetchone()[0]
        with self.assertRaises(s.Err) as e: call(self.c, "GET", "/api/complaints/" + other, user=self.cit)
        self.assertEqual(e.exception.code, 404)

    def test_submit_runs_ai_and_detects_duplicate(self):
        r = call(self.c, "POST", "/api/complaints", {"description": s.TPL["Drainage & Waterlogging"][0], "lat": 19.9805, "lng": 78.0245}, self.cit)
        self.assertEqual(r["category"], "Drainage & Waterlogging"); self.assertEqual(r["priority"], "High")
        self.assertIsNotNone(r["dup_of"]); self.assertTrue(r["cid"].startswith("CMP-")); self.assertEqual(r["status"], "Submitted")

    def test_submit_validation(self):
        for body in ({"description": "short", "lat": 20, "lng": 78}, {"description": "A long enough description here", "lat": "x", "lng": 78},
                     {"description": "A long enough description here", "lat": 20, "lng": 78, "category": "Bogus"}):
            with self.assertRaises(s.Err): call(self.c, "POST", "/api/complaints", body, self.cit)

    def test_admin_workflow_visible_to_citizen(self):
        r = call(self.c, "POST", "/api/complaints", {"description": "Garbage bin overflowing near the market for a week", "lat": 20, "lng": 78}, self.cit)
        u = call(self.c, "PUT", "/api/complaints/" + r["cid"], {"status": "Assigned", "dept": "Solid Waste Management", "note": "Crew booked"}, self.admin)
        self.assertEqual((u["status"], u["dept"]), ("Assigned", "Solid Waste Management"))
        d = call(self.c, "GET", "/api/complaints/" + r["cid"], user=self.cit)
        self.assertEqual(d["status"], "Assigned"); self.assertEqual(d["updates"][-1]["note"], "Crew booked")

    def test_status_rules(self):
        cid = call(self.c, "POST", "/api/complaints", {"description": "Streetlight broken and the street is dark", "lat": 20, "lng": 78}, self.cit)["cid"]
        with self.assertRaises(s.Err): call(self.c, "PUT", "/api/complaints/" + cid, {"status": "Assigned"}, self.admin)
        with self.assertRaises(s.Err): call(self.c, "PUT", "/api/complaints/" + cid, {"status": "Bogus"}, self.admin)
        with self.assertRaises(s.Err): call(self.c, "PUT", "/api/complaints/" + cid, {"status": "Under Review"}, self.cit)

    def test_filters(self):
        r = call(self.c, "GET", "/api/complaints", user=self.admin, q={"priority": "High", "category": "Roads & Potholes"})
        self.assertTrue(r and all(x["priority"] == "High" and x["category"] == "Roads & Potholes" for x in r))
        self.assertTrue(all(x["dup_of"] for x in call(self.c, "GET", "/api/complaints", user=self.admin, q={"dup": "1"})))

    def test_profile_update(self):
        p = call(self.c, "PUT", "/api/me", {"name": "New Name", "phone": "123", "area": "North Ward"}, self.cit)
        self.assertEqual((p["name"], p["area"]), ("New Name", "North Ward"))
        with self.assertRaises(s.Err): call(self.c, "PUT", "/api/me", {"name": "X", "area": "Nowhere"}, self.cit)


if __name__ == "__main__": unittest.main()
