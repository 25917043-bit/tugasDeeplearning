import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO


def process_video(input_path, output_path):
    # Load model YOLOv8 (yolov8s = small, jauh lebih cepat di CPU dibanding yolov8x)
    model = YOLO('yolov8s.pt')

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka video.")
        return

    # Properti video asli
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Konfigurasi VideoWriter untuk menyimpan video demo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ---- Parameter yang bisa diatur ----
    FLOOR_Y_THRESHOLD = int(height * 0.55)   # garis lantai (sesuaikan)
    PEAK_WINDOW = 3                           # jendela deteksi puncak (local maxima)
    BOUNCE_COOLDOWN = 6                       # frame minimal antar pantulan per bola
    CONF_THRESHOLD = 0.05                     # ambang kepercayaan deteksi (default 0.25)
    MATCH_RADIUS = 200                        # jarak maksimum (px) mencocokkan bola antar frame
    MAX_MISSED = 30                           # frame toleransi saat bola hilang dari deteksi

    # ---- State tracker sederhana (per bola) ----
    bounce_count = 0
    tracks = []   # list of dict: {id, x, y, y2_buf, last_bounce, missed}
    next_id = 0

    frame_idx = 0
    print("Memproses video dan menghitung pantulan...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi bola (kelas 32 = sports ball). Tracking diatur manual di bawah.
        results = model.predict(frame, classes=[32], conf=CONF_THRESHOLD, verbose=False)

        # Kumpulkan deteksi sebagai (cx, cy, y2, x1, y1, x2, y2)
        detections = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                detections.append(((x1 + x2) / 2, (y1 + y2) / 2, y2, x1, y1, x2, y2))

        # ---- Cocokkan deteksi dengan track yang ada (nearest neighbor) ----
        matched_track = set()
        matched_det = set()

        for di, det in enumerate(detections):
            cx, cy, y2 = det[0], det[1], det[2]
            best_ti = None
            best_d = MATCH_RADIUS
            for ti, t in enumerate(tracks):
                if ti in matched_track:
                    continue
                d = np.hypot(cx - t['x'], cy - t['y'])
                if d < best_d:
                    best_d = d
                    best_ti = ti
            if best_ti is not None:
                matched_track.add(best_ti)
                matched_det.add(di)
                t = tracks[best_ti]

                # Jika bola sempat hilang, reset buffer agar tidak memicu puncak palsu.
                if t['missed'] > 0:
                    t['y2_buf'].clear()

                t['x'], t['y'], t['missed'] = cx, cy, 0

                # Deteksi pantulan: cari PUNCAK (local maxima) dari tepi bawah bola (y2).
                # Pantulan terjadi saat bola mencapai titik terendah lalu naik kembali.
                t['y2_buf'].append(y2)
                if len(t['y2_buf']) == PEAK_WINDOW:
                    a, b, c = list(t['y2_buf'])
                    if (b >= a and b >= c and b >= FLOOR_Y_THRESHOLD
                            and (frame_idx - t['last_bounce']) > BOUNCE_COOLDOWN):
                        bounce_count += 1
                        t['last_bounce'] = frame_idx

        # ---- Buat track baru untuk deteksi yang tidak cocok ----
        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            cx, cy, y2 = det[0], det[1], det[2]
            tracks.append({
                'id': next_id,
                'x': cx, 'y': cy,
                'y2_buf': deque([y2], maxlen=PEAK_WINDOW),
                'last_bounce': -10**9,
                'missed': 0,
            })
            next_id += 1

        # ---- Naikkan hitungan missed untuk track yang tidak terdeteksi ----
        for ti, t in enumerate(tracks):
            if ti not in matched_track:
                t['missed'] += 1

        # Hapus track yang hilang terlalu lama
        tracks = [t for t in tracks if t['missed'] <= MAX_MISSED]

        # ---- Gambar semua bola yang terdeteksi di frame ini ----
        for det in detections:
            x1, y1, x2, y2 = det[3], det[4], det[5], det[6]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # ---- Gambar Garis Batas Lantai ----
        cv2.line(frame, (0, FLOOR_Y_THRESHOLD), (width, FLOOR_Y_THRESHOLD), (0, 0, 255), 2)
        cv2.putText(frame, "Floor Threshold", (50, FLOOR_Y_THRESHOLD - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ---- Tampilkan Live Counter ----
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (350, 100), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, "BOUNCE COUNTER", (35, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Count: {bounce_count}", (35, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Selesai! Video demo disimpan sebagai '{output_path}'. "
          f"Total Pantulan Terdeteksi: {bounce_count} | Bola aktif di akhir: {len(tracks)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = input("Masukkan path video input: ")

    input_path = input_path.strip().strip('"').strip("'")

    if not input_path:
        print("Error: Tidak ada video input.")
        sys.exit(1)

    process_video(input_path, "output_demo.mp4")
