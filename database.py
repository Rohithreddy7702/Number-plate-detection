"""
DATABASE MODULE
SQLite database for storing detections, blacklist, and stats
"""

import sqlite3
import os
from datetime import datetime, timedelta
import random

DB_PATH = 'plates.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create all tables if they don't exist."""
    conn = get_db()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text  TEXT    NOT NULL,
            confidence  REAL    DEFAULT 0,
            image_path  TEXT,
            is_blacklisted INTEGER DEFAULT 0,
            detected_at TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS blacklist (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT    NOT NULL UNIQUE,
            reason     TEXT,
            added_at   TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print("[DB] Database initialized!")

def seed_demo_data():
    """Add demo detections if DB is empty (for presentation)."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM detections")
    count = c.fetchone()[0]

    if count == 0:
        demo_plates = [
            ("MH12AB1234", 91.5), ("GJ01BC5678", 88.0), ("DL4CAF9012", 76.5),
            ("KA03MN3456", 94.2), ("TN09XY7890", 82.3), ("MH14CD2345", 89.1),
            ("UP32EF6789", 71.8), ("RJ14GH0123", 95.0), ("WB02IJ4567", 67.4),
            ("MP09KL8901", 83.6), ("HR26MN2345", 92.1), ("PB10OP6789", 78.9),
        ]
        blacklisted = ["DL4CAF9012", "UP32EF6789"]

        for i, (plate, conf) in enumerate(demo_plates):
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            det_time = (datetime.now() - timedelta(days=days_ago, hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")
            is_bl = 1 if plate in blacklisted else 0
            c.execute(
                "INSERT INTO detections (plate_text, confidence, image_path, is_blacklisted, detected_at) VALUES (?,?,?,?,?)",
                (plate, conf, "demo_image.jpg", is_bl, det_time)
            )

        # Seed blacklist
        c.execute("INSERT OR IGNORE INTO blacklist (plate_text, reason) VALUES (?,?)",
                  ("DL4CAF9012", "Reported stolen vehicle"))
        c.execute("INSERT OR IGNORE INTO blacklist (plate_text, reason) VALUES (?,?)",
                  ("UP32EF6789", "Traffic violation - unpaid fines"))

        conn.commit()
        print("[DB] Demo data seeded!")
    conn.close()

def save_detection(plate_text, confidence, image_path, is_blacklisted):
    conn = get_db()
    conn.execute(
        "INSERT INTO detections (plate_text, confidence, image_path, is_blacklisted) VALUES (?,?,?,?)",
        (plate_text, confidence, image_path, 1 if is_blacklisted else 0)
    )
    conn.commit()
    conn.close()

def is_blacklisted(plate_text):
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM blacklist WHERE plate_text = ?", (plate_text.upper().strip(),)
    ).fetchone()
    conn.close()
    return row is not None

def get_recent_detections(limit=5):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM detections ORDER BY detected_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return rows

def get_all_detections(page=1, search=''):
    conn = get_db()
    offset = (page - 1) * 20
    if search:
        rows = conn.execute(
            "SELECT * FROM detections WHERE plate_text LIKE ? ORDER BY detected_at DESC LIMIT 20 OFFSET ?",
            (f'%{search}%', offset)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM detections ORDER BY detected_at DESC LIMIT 20 OFFSET ?",
            (offset,)
        ).fetchall()
    conn.close()
    return rows

def get_total_count(search=''):
    conn = get_db()
    if search:
        row = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE plate_text LIKE ?", (f'%{search}%',)
        ).fetchone()
    else:
        row = conn.execute("SELECT COUNT(*) FROM detections").fetchone()
    conn.close()
    return row[0]

def get_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    alerts = conn.execute("SELECT COUNT(*) FROM detections WHERE is_blacklisted=1").fetchone()[0]
    avg_conf = conn.execute("SELECT AVG(confidence) FROM detections WHERE confidence > 0").fetchone()[0]
    blacklist_count = conn.execute("SELECT COUNT(*) FROM blacklist").fetchone()[0]
    conn.close()
    return {
        "total": total,
        "alerts": alerts,
        "avg_confidence": round(avg_conf or 0, 1),
        "blacklist_count": blacklist_count
    }

def get_full_stats():
    conn = get_db()
    stats = get_stats()

    # Detections per day (last 7 days)
    daily = conn.execute("""
        SELECT DATE(detected_at) as day, COUNT(*) as count
        FROM detections
        WHERE detected_at >= DATE('now', '-7 days')
        GROUP BY DATE(detected_at)
        ORDER BY day ASC
    """).fetchall()

    # Confidence distribution
    conf_bins = {
        "0-50%": conn.execute("SELECT COUNT(*) FROM detections WHERE confidence < 50").fetchone()[0],
        "50-70%": conn.execute("SELECT COUNT(*) FROM detections WHERE confidence >= 50 AND confidence < 70").fetchone()[0],
        "70-90%": conn.execute("SELECT COUNT(*) FROM detections WHERE confidence >= 70 AND confidence < 90").fetchone()[0],
        "90-100%": conn.execute("SELECT COUNT(*) FROM detections WHERE confidence >= 90").fetchone()[0],
    }

    # Top plates
    top = conn.execute("""
        SELECT plate_text, COUNT(*) as count
        FROM detections GROUP BY plate_text
        ORDER BY count DESC LIMIT 5
    """).fetchall()

    conn.close()
    return {
        **stats,
        "daily_labels": [r["day"] for r in daily],
        "daily_counts": [r["count"] for r in daily],
        "conf_labels": list(conf_bins.keys()),
        "conf_counts": list(conf_bins.values()),
        "top_plates": [{"plate": r["plate_text"], "count": r["count"]} for r in top],
    }

def get_blacklist():
    conn = get_db()
    rows = conn.execute("SELECT * FROM blacklist ORDER BY added_at DESC").fetchall()
    conn.close()
    return rows

def add_to_blacklist(plate_text, reason=''):
    conn = get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO blacklist (plate_text, reason) VALUES (?,?)",
            (plate_text.upper(), reason)
        )
        # Mark all existing detections of this plate
        conn.execute(
            "UPDATE detections SET is_blacklisted=1 WHERE plate_text=?",
            (plate_text.upper(),)
        )
        conn.commit()
    except Exception as e:
        print(f"[DB Error] {e}")
    conn.close()

def remove_from_blacklist(plate_id):
    conn = get_db()
    plate = conn.execute("SELECT plate_text FROM blacklist WHERE id=?", (plate_id,)).fetchone()
    if plate:
        conn.execute("DELETE FROM blacklist WHERE id=?", (plate_id,))
        conn.execute("UPDATE detections SET is_blacklisted=0 WHERE plate_text=?", (plate["plate_text"],))
        conn.commit()
    conn.close()

def delete_detection(det_id):
    conn = get_db()
    conn.execute("DELETE FROM detections WHERE id=?", (det_id,))
    conn.commit()
    conn.close()
