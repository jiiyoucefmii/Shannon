"""SQLite store (data/shannon.db): question bank, assessments, submissions, chat-answer ratings.

items.status: pending (generated, waiting for the instructor) | approved | rejected | practice (made on demand for
one student's practice, never reviewed). Only approved items can go into an assessment.
"""
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  id INTEGER PRIMARY KEY, created REAL, kind TEXT, difficulty TEXT, topic TEXT, status TEXT,
  body TEXT, checks TEXT, sources TEXT, reviewed REAL);
CREATE TABLE IF NOT EXISTS assessments (
  id INTEGER PRIMARY KEY, created REAL, title TEXT, kind TEXT, item_ids TEXT, published INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS submissions (
  id INTEGER PRIMARY KEY, created REAL, student TEXT, item_id INTEGER, assessment_id INTEGER,
  answer TEXT, auto_score REAL, max_score REAL, result TEXT, flags TEXT,
  instructor_score REAL, instructor_comment TEXT, reviewed REAL);
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY, created REAL, student TEXT, question TEXT, answer TEXT, rating INTEGER, comment TEXT,
  style TEXT, model TEXT);
"""
JSON_COLS = {"body", "checks", "sources", "item_ids", "answer", "result", "flags"}


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _rows(self, sql, args=()):
        with self.lock:
            rows = self.db.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in JSON_COLS & d.keys():
                d[k] = json.loads(d[k]) if d[k] else None
            out.append(d)
        return out

    def _write(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur.lastrowid

    # items
    def add_item(self, kind, difficulty, topic, status, body, checks, sources):
        return self._write("INSERT INTO items (created, kind, difficulty, topic, status, body, checks, sources) "
                           "VALUES (?,?,?,?,?,?,?,?)", (time.time(), kind, difficulty, topic, status,
                                                        json.dumps(body), json.dumps(checks), json.dumps(sources)))

    def item(self, item_id):
        rows = self._rows("SELECT * FROM items WHERE id=?", (item_id,))
        return rows[0] if rows else None

    def items(self, status=None):
        if status:
            return self._rows("SELECT * FROM items WHERE status=? ORDER BY id DESC", (status,))
        return self._rows("SELECT * FROM items WHERE status != 'practice' ORDER BY id DESC")

    def update_item(self, item_id, status=None, body=None):
        if body is not None:
            self._write("UPDATE items SET body=? WHERE id=?", (json.dumps(body), item_id))
        if status is not None:
            self._write("UPDATE items SET status=?, reviewed=? WHERE id=?", (status, time.time(), item_id))

    # assessments
    def add_assessment(self, title, kind, item_ids):
        return self._write("INSERT INTO assessments (created, title, kind, item_ids) VALUES (?,?,?,?)",
                           (time.time(), title, kind, json.dumps(item_ids)))

    def assessment(self, aid):
        rows = self._rows("SELECT * FROM assessments WHERE id=?", (aid,))
        return rows[0] if rows else None

    def assessments(self, published_only=False):
        return self._rows("SELECT * FROM assessments" + (" WHERE published=1" if published_only else "")
                          + " ORDER BY id DESC")

    def publish(self, aid, published):
        self._write("UPDATE assessments SET published=? WHERE id=?", (1 if published else 0, aid))

    # submissions
    def add_submission(self, student, item_id, assessment_id, answer, auto_score, max_score, result, flags):
        return self._write("INSERT INTO submissions (created, student, item_id, assessment_id, answer, auto_score, "
                           "max_score, result, flags) VALUES (?,?,?,?,?,?,?,?,?)",
                           (time.time(), student, item_id, assessment_id, json.dumps(answer), auto_score, max_score,
                            json.dumps(result), json.dumps(flags)))

    def submission(self, sid):
        rows = self._rows("SELECT * FROM submissions WHERE id=?", (sid,))
        return rows[0] if rows else None

    def submissions(self, item_id=None, assessment_id=None, student=None):
        where, args = [], []
        for col, v in (("item_id", item_id), ("assessment_id", assessment_id), ("student", student)):
            if v is not None:
                where.append(f"{col}=?"); args.append(v)
        return self._rows("SELECT * FROM submissions" + (" WHERE " + " AND ".join(where) if where else "")
                          + " ORDER BY id DESC", args)

    def review(self, sid, score, comment):
        self._write("UPDATE submissions SET instructor_score=?, instructor_comment=?, reviewed=? WHERE id=?",
                    (score, comment, time.time(), sid))

    # chat ratings
    def add_feedback(self, student, question, answer, rating, comment, style, model):
        return self._write("INSERT INTO feedback (created, student, question, answer, rating, comment, style, model) "
                           "VALUES (?,?,?,?,?,?,?,?)", (time.time(), student, question, answer, rating, comment,
                                                        style, model))

    def feedback(self):
        return self._rows("SELECT * FROM feedback ORDER BY id DESC")
