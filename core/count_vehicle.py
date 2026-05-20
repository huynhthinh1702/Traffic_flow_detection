import os
import threading
import time

import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train6", "weights", "last.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "uploads")
EMPTY_COUNTS = {
    "motorbike": 0,
    "car": 0,
    "bus": 0,
    "truck": 0,
}

model = YOLO(MODEL_PATH)

current_counts = EMPTY_COUNTS.copy()
total_counts = EMPTY_COUNTS.copy()
current_frame = None
current_fps = 0
current_status = "Binh thuong"
state_lock = threading.Lock()


def reset_realtime_state():
    global current_counts, current_fps, current_frame, total_counts, current_status

    with state_lock:
        current_counts = EMPTY_COUNTS.copy()
        total_counts = EMPTY_COUNTS.copy()
        current_frame = None
        current_fps = 0
        current_status = "Binh thuong"


def get_realtime_snapshot():
    with state_lock:
        frame = current_frame
        counts = current_counts.copy()
        fps = current_fps
        status = current_status

    return frame, counts, fps, status


def classify_traffic(flow_value):
    if flow_value > 150:
        return "Tac duong"
    if flow_value > 90:
        return "Dong xe"
    return "Binh thuong"


def build_auto_comment(total_counts, peak_minute, flow_per_minute):
    total_vehicles = sum(total_counts.values())
    if total_vehicles == 0:
        return "Khong phat hien phuong tien trong video."

    dominant_type = max(total_counts.items(), key=lambda item: item[1])[0]
    dominant_ratio = (total_counts[dominant_type] / total_vehicles) * 100
    peak_flow = flow_per_minute.get(peak_minute, 0) if peak_minute is not None else 0
    peak_text = f"phut {peak_minute}" if peak_minute is not None else "khong xác dinh"
    traffic_status = classify_traffic(peak_flow)

    return (
        f"{dominant_type} chiem da so ({dominant_ratio:.1f}%). "
        f"Luu luong cao nhat vao {peak_text} ({peak_flow} xe/phut). "
        f"Danh gia chung: {traffic_status.lower()}."
    )


def create_video_writer(output_path, fps_video, frame_size):
    codecs = ("avc1", "mp4v", "XVID")

    for codec in codecs:
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*codec),
            fps_video,
            frame_size,
        )
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError("Khong the khoi tao bo ghi video voi cac codec hien co.")


def count_vehicle(video_path):
    global current_counts, current_frame, total_counts, current_fps, current_status

    reset_realtime_state()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Khong the mo video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Video khong hop le hoac khong doc duoc kich thuoc khung hinh.")

    frame_stride = 2
    output_fps = max(1.0, fps_video / frame_stride)

    stem, _ext = os.path.splitext(os.path.basename(video_path))
    filename = f"{stem}_result.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = create_video_writer(output_path, output_fps, (width, height))

    line_y = height // 2
    counted_ids = set()
    track_history = {}
    max_missing = 15
    missing_counter = {}

    class_names = {
        0: "motorbike",
        1: "car",
        2: "bus",
        3: "truck",
    }

    frame_count = 0
    processed_frames = 0
    last_elapsed_time = 0.0
    flow_per_second = {}

    try:
        while cap.isOpened():
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            elapsed_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if elapsed_time <= 0:
                elapsed_time = frame_count / fps_video if fps_video > 0 else 0.0
            last_elapsed_time = elapsed_time
            current_second = int(elapsed_time)

            if frame_count % frame_stride != 0:
                # Giữ nguyên khung đã annotate gần nhất để tránh nhấp nháy
                # giữa frame có detect và frame bị skip.
                continue

            processed_frames += 1

            cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.5,
                imgsz=640,
                verbose=False,
            )

            current_ids = []
            boxes = results[0].boxes if results else None
            track_ids = None if boxes is None else boxes.id

            if boxes is not None and track_ids is not None:
                current_ids = track_ids.tolist()

                for box, cls, track_id in zip(boxes.xyxy, boxes.cls, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cls = int(cls)
                    label = class_names.get(cls, "unknown")
                    track_id = int(track_id)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    if track_id not in track_history:
                        track_history[track_id] = center_y

                    prev_y = track_history[track_id]
                    track_history[track_id] = center_y


                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    if track_id not in counted_ids:
                        crossed_line = (prev_y < line_y and center_y >= line_y) or (
                            prev_y > line_y and center_y <= line_y
                        )

                        if crossed_line:
                            counted_ids.add(track_id)

                            if label in total_counts:
                                total_counts[label] += 1

                            flow_per_second[current_second] = flow_per_second.get(current_second, 0) + 1

            for tid in list(track_history.keys()):
                if tid not in current_ids:
                    missing_counter[tid] = missing_counter.get(tid, 0) + 1
                    if missing_counter[tid] > max_missing:
                        track_history.pop(tid, None)
                        missing_counter.pop(tid, None)
                else:
                    missing_counter[tid] = 0

            counts_snapshot = total_counts.copy()
            minute_flow = flow_per_second.get(current_second, 0) * 60
            status_snapshot = classify_traffic(minute_flow)

            # Realtime frame chỉ hien tracking + line, khong ve text vang.
            realtime_frame = frame.copy()
            y_offset = 30
            for key, value in counts_snapshot.items():
                cv2.putText(
                    frame,
                    f"{key}: {value}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                y_offset += 30
            cv2.putText(
                frame,
                f"Trang thai: {status_snapshot}",
                (10, y_offset + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            end_time = time.time()
            fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0

            with state_lock:
                current_counts = counts_snapshot
                current_frame = realtime_frame
                current_fps = round(fps, 1)
                current_status = status_snapshot

            out.write(frame)

    finally:
        cap.release()
        out.release()

    flow_per_minute = {}
    for sec, value in flow_per_second.items():
        minute = sec // 60
        flow_per_minute[minute] = flow_per_minute.get(minute, 0) + value

    flow_per_minute = dict(sorted(flow_per_minute.items()))

    total_time_seconds = last_elapsed_time if frame_count > 0 else 0.0
    if total_time_seconds <= 0 and fps_video > 0:
        total_time_seconds = frame_count / fps_video
    total_time_hours = total_time_seconds / 3600 if total_time_seconds > 0 else 0
    total_vehicles = sum(total_counts.values())
    flow_rate = total_vehicles / total_time_hours if total_time_hours > 0 else 0
    peak_minute = max(flow_per_minute, key=flow_per_minute.get) if flow_per_minute else None
    peak_flow = flow_per_minute.get(peak_minute, 0) if peak_minute is not None else 0
    traffic_status = classify_traffic(peak_flow)




    replay_timeline = []
    for minute, flow in flow_per_minute.items():
        replay_timeline.append(
            {
                "minute": minute,
                "flow": flow,
                "status": classify_traffic(flow),
            }
        )

    summary = {
        "peak_minute": peak_minute,
        "peak_flow": peak_flow,
        "traffic_status": traffic_status,
        "auto_comment": build_auto_comment(total_counts, peak_minute, flow_per_minute),
        "replay_timeline": replay_timeline,
        "processed_frames": processed_frames,
        "frame_stride": frame_stride,
    }

    print("TOTAL:", total_counts)
    return total_counts, flow_per_minute, filename, flow_rate, summary
