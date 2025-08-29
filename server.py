from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
import os

# -------------------- CONFIG --------------------
DB_PATH = os.environ.get("DETECTIONS_DB", "sqlite:///detections.db")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_PATH
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# -------------------- MODEL ---------------------
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    label = db.Column(db.String(128), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    x1 = db.Column(db.Float, nullable=True)
    y1 = db.Column(db.Float, nullable=True)
    x2 = db.Column(db.Float, nullable=True)
    y2 = db.Column(db.Float, nullable=True)
    source = db.Column(db.String(128), nullable=True)  # e.g., "camera_0" or video filename
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "source": self.source,
            "timestamp": self.timestamp.isoformat()
        }

# Create tables at startup (Flask 3+ safe)
with app.app_context():
    db.create_all()

# -------------------- API -----------------------
def _insert_detection(d, default_source="unknown"):
    """Insert a single detection dict into DB."""
    label = d.get("label")
    confidence = float(d.get("confidence", 0.0))
    bbox = d.get("bbox", [None, None, None, None])
    ts_str = d.get("timestamp")
    ts = datetime.fromisoformat(ts_str) if ts_str else datetime.utcnow()
    det = Detection(
        label=label,
        confidence=confidence,
        x1=float(bbox[0]) if bbox[0] is not None else None,
        y1=float(bbox[1]) if bbox[1] is not None else None,
        x2=float(bbox[2]) if bbox[2] is not None else None,
        y2=float(bbox[3]) if bbox[3] is not None else None,
        source=d.get("source", default_source),
        timestamp=ts
    )
    db.session.add(det)
    return det

@app.route("/detections", methods=["POST"])
def add_detections():
    """
    Accepts EITHER:
    1) Batch payload:
       {
         "source": "camera_0",
         "detections": [
           {"label":"person","confidence":0.9,"bbox":[x1,y1,x2,y2],"timestamp":"..."},
           ...
         ]
       }
    2) Single detection payload:
       {"label":"person","confidence":0.9,"bbox":[x1,y1,x2,y2],"timestamp":"...","source":"camera_0"}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "missing json body"}), 400

    # batch
    if isinstance(data, dict) and "detections" in data and isinstance(data["detections"], list):
        source = data.get("source", "unknown")
        created_ids = []
        for d in data["detections"]:
            try:
                det = _insert_detection({**d, "source": d.get("source", source)}, default_source=source)
                created_ids.append(det.id)  # det.id is None until commit, but we return count anyway
            except Exception as e:
                app.logger.warning(f"Skipping invalid detection entry: {e}")
                continue
        db.session.commit()
        return jsonify({"inserted": len(created_ids)}), 201

    # single
    try:
        det = _insert_detection(data, default_source=data.get("source", "unknown"))
        db.session.commit()
        return jsonify({"inserted": 1, "id": det.id}), 201
    except Exception as e:
        app.logger.exception("Invalid detection payload")
        return jsonify({"error": str(e)}), 400

@app.route("/detections", methods=["GET"])
def get_detections():
    """
    Query params:
      label=<str>
      source=<str>
      start=<ISO timestamp>
      end=<ISO timestamp>
      limit=<int>   (default 200)
    """
    q = Detection.query
    label = request.args.get("label")
    source = request.args.get("source")
    start = request.args.get("start")
    end = request.args.get("end")
    limit = int(request.args.get("limit", 200))

    if label:
        q = q.filter(Detection.label == label)
    if source:
        q = q.filter(Detection.source == source)
    if start:
        q = q.filter(Detection.timestamp >= datetime.fromisoformat(start))
    if end:
        q = q.filter(Detection.timestamp <= datetime.fromisoformat(end))

    rows = q.order_by(Detection.timestamp.desc()).limit(limit).all()
    return jsonify([r.to_dict() for r in rows])

@app.route("/stats", methods=["GET"])
def stats():
    """
    Returns counts per label, total detections, and recent rate.
    Optional: ?minutes=5 (defaults to 5)
    """
    total = db.session.query(func.count(Detection.id)).scalar()
    counts = db.session.query(Detection.label, func.count(Detection.id)).group_by(Detection.label).all()
    counts_dict = {label: cnt for label, cnt in counts}

    minutes = int(request.args.get("minutes", 5))
    since = datetime.utcnow() - timedelta(minutes=minutes)
    recent_count = db.session.query(func.count(Detection.id)).filter(Detection.timestamp >= since).scalar()
    per_minute = (recent_count / minutes) if minutes > 0 else None

    return jsonify({
        "total_detections": int(total),
        "counts_per_label": counts_dict,
        "recent_count_last_minutes": int(recent_count),
        "recent_per_minute": per_minute
    })

# -------------------- DASHBOARD -----------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/health")
def health():
    return jsonify({"ok": True})

# -------------------- MAIN ----------------------
if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=5000, debug=True)
