import cv2
import json
import argparse
from pathlib import Path
import numpy as np


points_rgb = []
points_depth = []
mode = "rgb"


def mouse_callback(event, x, y, flags, param):
    global mode, points_rgb, points_depth

    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == "rgb":
            points_rgb.append([x, y])
            print(f"RGB point {len(points_rgb)}: ({x}, {y})")
        elif mode == "depth":
            points_depth.append([x, y])
            print(f"DEPTH point {len(points_depth)}: ({x}, {y})")


def draw_points(img, points, color):
    out = img.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(out, (x, y), 6, color, -1)
        cv2.putText(
            out,
            str(i + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
    return out


def compute_reprojection_error(rgb_points, depth_points, H):
    """
    RGB点をHでDepth座標に変換し、実際にクリックしたDepth点との誤差を計算
    """
    rgb_pts = np.array(rgb_points, dtype=np.float32)
    depth_pts = np.array(depth_points, dtype=np.float32)

    rgb_homo = cv2.convertPointsToHomogeneous(rgb_pts).reshape(-1, 3).T
    projected = H @ rgb_homo
    projected = projected / projected[2, :]
    projected_xy = projected[:2, :].T

    errors = np.linalg.norm(projected_xy - depth_pts, axis=1)

    return {
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        "max_error_px": float(np.max(errors)),
        "errors_px": errors.tolist(),
    }


def warp_depth_vis_to_rgb(depth_vis, H_rgb_to_depth, rgb_size):
    """
    depth_visをRGB座標へワープして確認用画像を作る。
    Hは RGB→Depth なので、Depth→RGB には逆行列を使う。
    """
    rgb_w, rgb_h = rgb_size
    H_depth_to_rgb = np.linalg.inv(H_rgb_to_depth)
    aligned = cv2.warpPerspective(depth_vis, H_depth_to_rgb, (rgb_w, rgb_h))
    return aligned


def main():
    global mode

    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", type=str, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--points-json", type=str, default="calibration_points.json")
    parser.add_argument("--transform-json", type=str, default="rgb_to_depth_homography.json")
    args = parser.parse_args()

    record_dir = Path(args.record_dir)

    color_path = record_dir / "color" / f"{args.frame:06d}.jpg"
    depth_vis_path = record_dir / "depth_vis" / f"{args.frame:06d}.jpg"

    rgb = cv2.imread(str(color_path))
    depth_vis = cv2.imread(str(depth_vis_path))

    if rgb is None:
        raise RuntimeError(f"RGB画像が読めません: {color_path}")

    if depth_vis is None:
        raise RuntimeError(f"Depth可視化画像が読めません: {depth_vis_path}")

    rgb_h, rgb_w = rgb.shape[:2]
    depth_h, depth_w = depth_vis.shape[:2]

    print("RGB image:", color_path, rgb_w, rgb_h)
    print("Depth image:", depth_vis_path, depth_w, depth_h)
    print()
    print("操作:")
    print("  r : RGB画像に点を打つモード")
    print("  d : Depth画像に点を打つモード")
    print("  u : 今のモードの直前点を削除")
    print("  s : 点を保存し、homographyを計算して終了")
    print("  q : 保存せず終了")
    print()
    print("重要:")
    print("  RGBで点1を打ったら、Depthでも同じ実世界点を点1として打つ")
    print("  RGB点数とDepth点数は同じにする")
    print("  homographyは最低4点、実用上は8〜20点がおすすめ")
    print("  3m付近を測るなら、3m付近の静止物の角を中心に取る")
    print()

    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("DEPTH", cv2.WINDOW_NORMAL)

    cv2.setMouseCallback("RGB", mouse_callback)
    cv2.setMouseCallback("DEPTH", mouse_callback)

    while True:
        rgb_show = draw_points(rgb, points_rgb, (0, 255, 0))
        depth_show = draw_points(depth_vis, points_depth, (0, 255, 255))

        cv2.putText(
            rgb_show,
            f"MODE={mode.upper()} RGB points={len(points_rgb)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            depth_show,
            f"MODE={mode.upper()} DEPTH points={len(points_depth)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        cv2.imshow("RGB", rgb_show)
        cv2.imshow("DEPTH", depth_show)

        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            mode = "rgb"
            print("Mode: RGB")

        elif key == ord("d"):
            mode = "depth"
            print("Mode: DEPTH")

        elif key == ord("u"):
            if mode == "rgb" and points_rgb:
                removed = points_rgb.pop()
                print("Removed RGB:", removed)
            elif mode == "depth" and points_depth:
                removed = points_depth.pop()
                print("Removed DEPTH:", removed)

        elif key == ord("s"):
            if len(points_rgb) != len(points_depth):
                print("ERROR: RGB点数とDepth点数が違います")
                print("RGB:", len(points_rgb), "DEPTH:", len(points_depth))
                continue

            if len(points_rgb) < 4:
                print("ERROR: homographyには最低4点必要です")
                continue

            if len(points_rgb) < args.min_points:
                print(f"WARNING: 点数が少なめです。現在 {len(points_rgb)} 点。推奨は {args.min_points} 点以上です。")
                print("それでも保存するなら、もう一度 s を押してください。")
                args.min_points = len(points_rgb)
                continue

            rgb_points_np = np.array(points_rgb, dtype=np.float32)
            depth_points_np = np.array(points_depth, dtype=np.float32)

            # RGB座標 → Depth座標 のHomography
            H, mask = cv2.findHomography(
                rgb_points_np,
                depth_points_np,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
            )

            if H is None:
                print("ERROR: homographyを計算できませんでした。点を取り直してください。")
                continue

            mask_list = mask.ravel().astype(int).tolist() if mask is not None else None

            error_info = compute_reprojection_error(points_rgb, points_depth, H)

            points_data = {
                "record_dir": str(record_dir),
                "frame": args.frame,
                "rgb_image": str(color_path),
                "depth_vis_image": str(depth_vis_path),
                "rgb_points": points_rgb,
                "depth_points": points_depth,
                "rgb_size": [rgb_w, rgb_h],
                "depth_size": [depth_w, depth_h],
                "note": "point i in rgb_points corresponds to point i in depth_points",
            }

            transform_data = {
                "method": "homography",
                "direction": "rgb_to_depth",
                "matrix": H.tolist(),
                "rgb_size": [rgb_w, rgb_h],
                "depth_size": [depth_w, depth_h],
                "source_points_json": args.points_json,
                "frame_used": args.frame,
                "num_points": len(points_rgb),
                "ransac_inlier_mask": mask_list,
                "reprojection_error": error_info,
                "usage": "Apply this matrix to RGB keypoint coordinates to obtain corresponding Depth pixel coordinates.",
            }

            points_json_path = record_dir / args.points_json
            transform_json_path = record_dir / args.transform_json

            with open(points_json_path, "w", encoding="utf-8") as f:
                json.dump(points_data, f, indent=2, ensure_ascii=False)

            with open(transform_json_path, "w", encoding="utf-8") as f:
                json.dump(transform_data, f, indent=2, ensure_ascii=False)

            print()
            print("Saved points:", points_json_path)
            print("Saved transform:", transform_json_path)
            print()
            print("Homography RGB -> Depth:")
            print(H)
            print()
            print("Reprojection error:")
            print("  mean  :", error_info["mean_error_px"])
            print("  median:", error_info["median_error_px"])
            print("  max   :", error_info["max_error_px"])

            # 確認用: Depth可視化画像をRGB座標へワープして重ねる
            aligned_depth_to_rgb = warp_depth_vis_to_rgb(depth_vis, H, (rgb_w, rgb_h))

            overlay = cv2.addWeighted(rgb, 0.65, aligned_depth_to_rgb, 0.35, 0)

            aligned_path = record_dir / "depth_vis_aligned_to_rgb.jpg"
            overlay_path = record_dir / "rgb_depth_overlay_check.jpg"

            cv2.imwrite(str(aligned_path), aligned_depth_to_rgb)
            cv2.imwrite(str(overlay_path), overlay)

            print()
            print("Saved check image:", aligned_path)
            print("Saved overlay image:", overlay_path)

            cv2.imshow("Aligned Depth to RGB", aligned_depth_to_rgb)
            cv2.imshow("RGB + Aligned Depth Overlay", overlay)
            print()
            print("確認画像を表示しました。何かキーを押すと終了します。")
            cv2.waitKey(0)

            break

        elif key == ord("q"):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()