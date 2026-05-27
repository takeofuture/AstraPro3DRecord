import os
import time
import csv
import argparse
from pathlib import Path

import cv2
import numpy as np
from primesense import openni2
from primesense import _openni2 as c_api


def find_openni2_redist():
    candidates = [
        r"C:\Program Files\OpenNI2\Redist",
        r"C:\Program Files (x86)\OpenNI2\Redist",
        r"C:\Program Files\Orbbec\OpenNI2\Redist",
        r"C:\Program Files\Orbbec\ASTRA\Redist",
    ]

    env_path = os.environ.get("OPENNI2_REDIST")
    if env_path:
        candidates.insert(0, env_path)

    for p in candidates:
        if os.path.exists(p):
            return p

    raise RuntimeError(
        "OpenNI2 Redist folder が見つかりません。\n"
        "例: C:\\Program Files\\OpenNI2\\Redist\n"
        "見つからない場合は環境変数 OPENNI2_REDIST に指定してください。"
    )


def make_depth_vis(depth_mm, max_mm=5000):
    """
    16bit depth(mm)を表示用のカラーマップに変換。
    保存用depthには使わない。表示用だけ。
    """
    depth = depth_mm.copy()

    # 0は欠損として扱う
    valid = depth > 0

    vis = np.zeros_like(depth, dtype=np.uint8)

    if np.any(valid):
        clipped = np.clip(depth, 0, max_mm)
        vis = (255 - (clipped.astype(np.float32) / max_mm * 255)).astype(np.uint8)
        vis[~valid] = 0

    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    color[~valid] = (0, 0, 0)

    return color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-index", type=int, default=0, help="Astra Pro HD Camera の OpenCV index")
    parser.add_argument("--out", type=str, default="astra_recording", help="出力フォルダ")
    parser.add_argument("--seconds", type=float, default=10.0, help="録画秒数。0ならqキーまで録画")
    parser.add_argument("--rgb-width", type=int, default=1280)
    parser.add_argument("--rgb-height", type=int, default=720)
    parser.add_argument("--rgb-fps", type=int, default=30)
    parser.add_argument("--preview-max-depth-mm", type=int, default=5000)
    args = parser.parse_args()

    out_dir = Path(args.out)
    color_dir = out_dir / "color"
    depth_dir = out_dir / "depth"
    depth_vis_dir = out_dir / "depth_vis"

    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    depth_vis_dir.mkdir(parents=True, exist_ok=True)

    timestamps_path = out_dir / "timestamps.csv"
    preview_video_path = out_dir / "preview_side_by_side.mp4"

    # =========================
    # OpenNI2 / Depth 初期化
    # =========================

    openni_path = find_openni2_redist()
    print("OpenNI2 Redist:", openni_path)

    openni2.initialize(openni_path)

    dev = openni2.Device.open_any()
    print("Depth device opened.")

    depth_stream = dev.create_depth_stream()

    # 可能なら 640x480@30fps にする
    # Astra Proの対応モードによって失敗する場合があるのでtryにする
    try:
        depth_stream.set_video_mode(
            c_api.OniVideoMode(
                pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                resolutionX=640,
                resolutionY=480,
                fps=30,
            )
        )
        print("Depth mode set to 640x480@30, 1mm")
    except Exception as e:
        print("Depth mode set failed. Use default mode.")
        print(e)

    depth_stream.start()

    # =========================
    # RGBカメラ初期化
    # =========================

    cap = cv2.VideoCapture(args.rgb_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"RGB camera を開けません: index={args.rgb_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.rgb_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.rgb_height)
    cap.set(cv2.CAP_PROP_FPS, args.rgb_fps)

    ret, test_frame = cap.read()
    if not ret:
        raise RuntimeError("RGB frame を取得できません。rgb-indexを変えてください。")

    rgb_h, rgb_w = test_frame.shape[:2]
    print(f"RGB opened: index={args.rgb_index}, size={rgb_w}x{rgb_h}")

    # preview動画用
    preview_w = 1280
    preview_h = 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    preview_writer = cv2.VideoWriter(
        str(preview_video_path),
        fourcc,
        min(args.rgb_fps, 30),
        (preview_w, preview_h),
    )

    # =========================
    # 録画ループ
    # =========================

    print()
    print("Recording started.")
    print("qキーで停止します。")
    print("Depthは16bit PNGで保存します。")

    start_perf = time.perf_counter()
    start_wall = time.time()
    frame_idx = 0

    with open(timestamps_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "time_sec",
            "wall_time",
            "color_path",
            "depth_path",
            "depth_vis_path",
            "rgb_width",
            "rgb_height",
            "depth_width",
            "depth_height",
            "depth_min_mm",
            "depth_max_mm",
            "depth_valid_ratio",
        ])

        while True:
            now_perf = time.perf_counter()
            elapsed = now_perf - start_perf

            if args.seconds > 0 and elapsed >= args.seconds:
                break

            # Depth取得
            depth_frame = depth_stream.read_frame()
            depth_w = depth_frame.width
            depth_h = depth_frame.height

            depth_mm = np.frombuffer(
                depth_frame.get_buffer_as_uint16(),
                dtype=np.uint16
            ).reshape((depth_h, depth_w))
            depth_mm = cv2.flip(depth_mm, 1)
            # RGB取得
            ret, color_frame = cap.read()
            if not ret:
                print("RGB frame failed")
                continue

            # 表示用Depth
            depth_vis = make_depth_vis(depth_mm, max_mm=args.preview_max_depth_mm)

            # 保存パス
            color_path = color_dir / f"{frame_idx:06d}.jpg"
            depth_path = depth_dir / f"{frame_idx:06d}.png"
            depth_vis_path = depth_vis_dir / f"{frame_idx:06d}.jpg"

            # 保存
            cv2.imwrite(str(color_path), color_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            cv2.imwrite(str(depth_path), depth_mm)  # 16bit PNG
            cv2.imwrite(str(depth_vis_path), depth_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            valid = depth_mm > 0
            if np.any(valid):
                depth_min = int(depth_mm[valid].min())
                depth_max = int(depth_mm[valid].max())
                valid_ratio = float(valid.mean())
            else:
                depth_min = 0
                depth_max = 0
                valid_ratio = 0.0

            writer.writerow([
                frame_idx,
                elapsed,
                start_wall + elapsed,
                str(color_path),
                str(depth_path),
                str(depth_vis_path),
                rgb_w,
                rgb_h,
                depth_w,
                depth_h,
                depth_min,
                depth_max,
                valid_ratio,
            ])

            # =========================
            # 2画面プレビュー
            # =========================

            rgb_preview = cv2.resize(color_frame, (640, 480))
            depth_preview = cv2.resize(depth_vis, (640, 480))

            cv2.putText(
                rgb_preview,
                f"RGB frame={frame_idx} time={elapsed:.2f}s",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                depth_preview,
                f"DEPTH min={depth_min} max={depth_max} valid={valid_ratio:.2f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            side_by_side = np.hstack([rgb_preview, depth_preview])

            cv2.imshow("Astra Pro RGB-D Preview  左:RGB  右:Depth", side_by_side)

            if preview_writer.isOpened():
                preview_writer.write(side_by_side)

            if frame_idx % 30 == 0:
                print(
                    f"frame={frame_idx}, time={elapsed:.2f}s, "
                    f"depth={depth_w}x{depth_h}, "
                    f"valid={valid_ratio:.2f}, min={depth_min}, max={depth_max}"
                )

            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    print()
    print("Recording finished.")
    print("Saved to:", out_dir.resolve())
    print("timestamps:", timestamps_path)
    print("preview video:", preview_video_path)

    # =========================
    # 終了処理
    # =========================

    cap.release()
    preview_writer.release()
    depth_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()