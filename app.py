from datetime import datetime
from functools import lru_cache
import os
import threading
import time

import cv2
from docx import Document
from flask import Flask, Response, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from core.count_vehicle import count_vehicle
from core.db import get_connection, init_db

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DOC_PATH = os.path.join(BASE_DIR, "thong tin du an.docx")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
init_db()

latest_result = None
latest_processing = False
latest_error = None
processing_lock = threading.Lock()


def reset_realtime_state():
    import core.count_vehicle as cv_module

    cv_module.reset_realtime_state()


def build_result_payload(counts, result_video, flow_per_minute, summary):
    return {
        "motorbike": int(counts.get("motorbike", 0)),
        "car": int(counts.get("car", 0)),
        "bus": int(counts.get("bus", 0)),
        "truck": int(counts.get("truck", 0)),
        "video": result_video,
        "flow": flow_per_minute,
        "traffic_status": summary.get("traffic_status", "Binh thuong"),
        "auto_comment": summary.get("auto_comment", ""),
        "peak_minute": summary.get("peak_minute"),
        "peak_flow": summary.get("peak_flow", 0),
        "replay_timeline": summary.get("replay_timeline", []),
        "processed_frames": summary.get("processed_frames", 0),
        "frame_stride": summary.get("frame_stride", 1),
    }


def process_video(video_path):
    global latest_error, latest_processing, latest_result

    try:
        with processing_lock:
            counts, flow_per_minute, result_video, _flow_rate, summary = count_vehicle(video_path)

            print("FINAL COUNTS:", counts)
            latest_result = build_result_payload(counts, result_video, flow_per_minute, summary)

            if sum(counts.values()) == 0:
                print("Khong co xe -> bo qua luu DB")
                return

            try:
                with get_connection() as conn:
                    cursor = conn.cursor()
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    cursor.execute(
                        """
                        INSERT INTO results (
                            video_name, motorbike, car, bus, truck, traffic_status,
                            auto_comment, peak_minute, peak_flow, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            result_video,
                            int(counts.get("motorbike", 0)),
                            int(counts.get("car", 0)),
                            int(counts.get("bus", 0)),
                            int(counts.get("truck", 0)),
                            summary.get("traffic_status", "Binh thuong"),
                            summary.get("auto_comment", ""),
                            summary.get("peak_minute"),
                            summary.get("peak_flow", 0),
                            now,
                        ),
                    )

                print("Luu SQLite thanh cong:", counts)

            except Exception as e:
                print("Loi DB:", e)

    except Exception as e:
        latest_result = None
        latest_error = f"Loi xu ly video: {e}"
        print("Loi xu ly video:", e)
    finally:
        latest_processing = False


@app.route("/", methods=["GET", "POST"])
def index():
    global latest_error, latest_processing, latest_result

    if request.method == "POST":
        if latest_processing:
            return render_template(
                "index.html",
                error="He thong dang xu ly mot video khac. Vui long doi video hien tai hoan tat.",
            ), 409

        video = request.files.get("video")
        if video is None or not video.filename:
            return render_template("index.html", error="Vui long chon mot video hop le."), 400

        safe_name = secure_filename(video.filename)
        if not safe_name:
            return render_template("index.html", error="Ten file khong hop le."), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stored_name = f"{timestamp}_{safe_name}"
        video_path = os.path.join(UPLOAD_FOLDER, stored_name)
        video.save(video_path)

        latest_processing = True
        latest_result = None
        latest_error = None
        reset_realtime_state()

        threading.Thread(
            target=process_video,
            args=(video_path,),
            daemon=True,
        ).start()

        return redirect(url_for("realtime"))

    return render_template("index.html", error=latest_error)


def generate_frames():
    import core.count_vehicle as cv_module

    while True:
        frame, _, _, _ = cv_module.get_realtime_snapshot()

        if frame is None:
            time.sleep(0.03)
            continue

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            time.sleep(0.03)
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/realtime_data")
def realtime_data():
    import core.count_vehicle as cv_module

    _, counts, fps, status = cv_module.get_realtime_snapshot()
    return jsonify(
        {
            "motorbike": counts["motorbike"],
            "car": counts["car"],
            "bus": counts["bus"],
            "truck": counts["truck"],
            "fps": fps,
            "status": status,
        }
    )


@app.route("/realtime_status")
def realtime_status():
    return jsonify(
        {
            "processing": bool(latest_processing),
            "done": (latest_result is not None) and (not latest_processing),
            "result_video": (latest_result or {}).get("video"),
            "traffic_status": (latest_result or {}).get("traffic_status"),
            "auto_comment": (latest_result or {}).get("auto_comment"),
            "peak_minute": (latest_result or {}).get("peak_minute"),
            "peak_flow": (latest_result or {}).get("peak_flow"),
            "replay_timeline": (latest_result or {}).get("replay_timeline", []),
            "processed_frames": (latest_result or {}).get("processed_frames"),
            "frame_stride": (latest_result or {}).get("frame_stride"),
            "error": latest_error,
        }
    )


@app.route("/dashboard")
def dashboard():
    global latest_result

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                video_name, motorbike, car, bus, truck,
                traffic_status, auto_comment, peak_minute, peak_flow, created_at
            FROM results
            ORDER BY created_at DESC
            """
        )
        rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT
                COALESCE(SUM(motorbike), 0),
                COALESCE(SUM(car), 0),
                COALESCE(SUM(bus), 0),
                COALESCE(SUM(truck), 0)
            FROM results
            """
        )
        total = list(cursor.fetchone() or (0, 0, 0, 0))

        cursor.execute("SELECT COUNT(*) FROM results")
        total_videos = cursor.fetchone()[0]

    total_vehicles = sum(total)

    flow_labels = []
    flow_data = []

    if latest_result and "flow" in latest_result:
        flow_dict = latest_result["flow"]
        flow_labels = list(flow_dict.keys())
        flow_data = list(flow_dict.values())

    if flow_data:
        peak_value = max(flow_data)
        peak_minute = flow_labels[flow_data.index(peak_value)]
    elif rows:
        peak_minute = rows [0][7]
        peak_value = rows [0][8]
    else:
        peak_minute = "-"
        peak_value = 0

    return render_template(
        "dashboard.html",
        rows=rows,
        total=total,
        flow_labels=flow_labels,
        flow_data=flow_data,
        total_vehicles=total_vehicles,
        total_videos=total_videos,
        peak_minute=peak_minute,
        peak_value=peak_value,
        latest_result=latest_result,
    )


@app.route("/replay_timeline")
def replay_timeline():
    timeline = (latest_result or {}).get("replay_timeline", [])
    return jsonify({"timeline": timeline})


@app.route("/realtime")
def realtime():
    return render_template("realtime.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@lru_cache(maxsize=1)
def _load_project_info(doc_mtime):
    project_intro = ""
    members = []

    try:
        doc = Document(DOC_PATH)
        lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    except Exception:
        lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if "giới thiệu" in line.lower() and ":" in line:
            project_intro = line.split(":", 1)[1].strip()

        if "MSSV" in line:
            mssv = line.replace("MSSV:", "").strip()

            for j in range(i + 1, min(i + 3, len(lines))):
                if "Họ tên" in lines[j]:
                    name = lines[j].replace("Họ tên:", "").strip()
                    members.append(
                        {
                            "name": name,
                            "id": mssv,
                        }
                    )
                    break

        i += 1

    return project_intro, members


def get_project_info():
    doc_mtime = os.path.getmtime(DOC_PATH) if os.path.exists(DOC_PATH) else None
    return _load_project_info(doc_mtime)


@app.route("/about")
def about():
    project_name = "Hệ thống nhận diện & đếm lưu lượng giao thông"
    project_intro, members = get_project_info()

    return render_template(
        "about.html",
        project_name=project_name,
        project_intro=project_intro,
        members=members,
    )


if __name__ == "__main__":
    app.run(debug=True)
