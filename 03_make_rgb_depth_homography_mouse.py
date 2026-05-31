import cv2
import json
import argparse
from pathlib import Path
import numpy as np


WINDOW_NAME = "RGB-D Homography Calibration"

points_rgb = []
points_depth = []

rgb = None
depth_vis = None

rgb_scale = 0.4
depth_scale = 1.0

rgb_w = rgb_h = 0
depth_w = depth_h = 0
rgb_disp_w = rgb_disp_h = 0
depth_disp_w = depth_disp_h = 0

button_undo = None
button_save = None
button_quit = None

record_dir = None
args_global = None


def draw_points(img, points, color):
    out = img.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(out, (int(x), int(y)), 6, color, -1)
        cv2.putText(
            out,
            str(i + 1),
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
    return out


def compute_reprojection_error(rgb_points, depth_points, H):
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


def warp_depth_vis_to_rgb(depth_vis_img, H_rgb_to_depth, rgb_size):
    rgb_w_, rgb_h_ = rgb_size
    H_depth_to_rgb = np.linalg.inv(H_rgb_to_depth)
    aligned = cv2.warpPerspective(depth_vis_img, H_depth_to_rgb, (rgb_w_, rgb_h_))
    return aligned


def make_display_image():
    global rgb_disp_w, rgb_disp_h, depth_disp_w, depth_disp_h
    global button_undo, button_save, button_quit

    rgb_show = cv2.resize(rgb, None, fx=rgb_scale, fy=rgb_scale)
    depth_show = cv2.resize(depth_vis, None, fx=depth_scale, fy=depth_scale)

    rgb_disp_h, rgb_disp_w = rgb_show.shape[:2]
    depth_disp_h, depth_disp_w = depth_show.shape[:2]

    rgb_points_disp = [[x * rgb_scale, y * rgb_scale] for x, y in points_rgb]
    depth_points_disp = [[x * depth_scale, y * depth_scale] for x, y in points_depth]

    rgb_show = draw_points(rgb_show, rgb_points_disp, (0, 255, 0))
    depth_show = draw_points(depth_show, depth_points_disp, (0, 255, 255))

    max_h = max(rgb_show.shape[0], depth_show.shape[0])

    if rgb_show.shape[0] < max_h:
        pad = max_h - rgb_show.shape[0]
        rgb_show = cv2.copyMakeBorder(
            rgb_show, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    if depth_show.shape[0] < max_h:
        pad = max_h - depth_show.shape[0]
        depth_show = cv2.copyMakeBorder(
            depth_show, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    cv2.putText(
        rgb_show,
        f"RGB points={len(points_rgb)} scale={rgb_scale}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        depth_show,
        f"DEPTH points={len(points_depth)} scale={depth_scale}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    combined = np.hstack([rgb_show, depth_show])

    bar_h = 70
    bar = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
    bar[:] = (35, 35, 35)

    y1, y2 = 15, 55
    button_undo = (20, y1 + combined.shape[0], 170, y2 + combined.shape[0])
    button_save = (190, y1 + combined.shape[0], 380, y2 + combined.shape[0])
    button_quit = (400, y1 + combined.shape[0], 560, y2 + combined.shape[0])

    cv2.rectangle(bar, (20, y1), (170, y2), (90, 90, 90), -1)
    cv2.putText(
        bar,
        "UNDO",
        (55, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.rectangle(bar, (190, y1), (380, y2), (0, 120, 0), -1)
    cv2.putText(
        bar,
        "SAVE",
        (250, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.rectangle(bar, (400, y1), (560, y2), (0, 0, 160), -1)
    cv2.putText(
        bar,
        "QUIT",
        (455, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        bar,
        "Click RGB point on left, then matching DEPTH point on right.",
        (590, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        2,
    )

    return np.vstack([combined, bar])


def point_in_rect(x, y, rect):
    if rect is None:
        return False
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def undo_last():
    if points_rgb and points_depth:
        r = points_rgb.pop()
        d = points_depth.pop()
        print("Removed pair:", r, d)
    elif points_rgb:
        r = points_rgb.pop()
        print("Removed RGB only:", r)
    elif points_depth:
        d = points_depth.pop()
        print("Removed DEPTH only:", d)


def save_homography():
    if len(points_rgb) != len(points_depth):
        print("ERROR: RGB点数とDEPTH点数が違います")
        print("RGB:", len(points_rgb), "DEPTH:", len(points_depth))
        return False

    if len(points_rgb) < 4:
        print("ERROR: homographyには最低4点必要です")
        return False

    if len(points_rgb) < args_global.min_points:
        print(
            f"ERROR: 点数が少ないです。現在 {len(points_rgb)} 点。"
            f"最低 {args_global.min_points} 点。"
        )
        return False

    rgb_points_np = np.array(points_rgb, dtype=np.float32)
    depth_points_np = np.array(points_depth, dtype=np.float32)

    H, mask = cv2.findHomography(
        rgb_points_np,
        depth_points_np,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )

    if H is None:
        print("ERROR: homographyを計算できませんでした。点を取り直してください。")
        return False

    error_info = compute_reprojection_error(points_rgb, points_depth, H)
    mask_list = mask.ravel().astype(int).tolist() if mask is not None else None

    color_path = record_dir / "color" / f"{args_global.frame:06d}.jpg"
    depth_vis_path = record_dir / "depth_vis" / f"{args_global.frame:06d}.jpg"

    points_data = {
        "record_dir": str(record_dir),
        "frame": args_global.frame,
        "rgb_image": str(color_path),
        "depth_vis_image": str(depth_vis_path),
        "rgb_points": points_rgb,
        "depth_points": points_depth,
        "rgb_size": [rgb_w, rgb_h],
        "depth_size": [depth_w, depth_h],
        "rgb_scale_display": rgb_scale,
        "depth_scale_display": depth_scale,
        "note": "point i in rgb_points corresponds to point i in depth_points",
    }

    transform_data = {
        "method": "homography",
        "direction": "rgb_to_depth",
        "matrix": H.tolist(),
        "rgb_size": [rgb_w, rgb_h],
        "depth_size": [depth_w, depth_h],
        "source_points_json": args_global.points_json,
        "frame_used": args_global.frame,
        "num_points": len(points_rgb),
        "ransac_inlier_mask": mask_list,
        "reprojection_error": error_info,
        "usage": "Apply this matrix to RGB keypoint coordinates to obtain corresponding Depth pixel coordinates.",
    }

    points_json_path = record_dir / args_global.points_json
    transform_json_path = record_dir / args_global.transform_json

    with open(points_json_path, "w", encoding="utf-8") as f:
        json.dump(points_data, f, indent=2, ensure_ascii=False)

    with open(transform_json_path, "w", encoding="utf-8") as f:
        json.dump(transform_data, f, indent=2, ensure_ascii=False)

    aligned_depth_to_rgb = warp_depth_vis_to_rgb(depth_vis, H, (rgb_w, rgb_h))
    overlay = cv2.addWeighted(rgb, 0.65, aligned_depth_to_rgb, 0.35, 0)

    aligned_path = record_dir / "depth_vis_aligned_to_rgb.jpg"
    overlay_path = record_dir / "rgb_depth_overlay_check.jpg"

    cv2.imwrite(str(aligned_path), aligned_depth_to_rgb)
    cv2.imwrite(str(overlay_path), overlay)

    print()
    print("Saved points:", points_json_path)
    print("Saved transform:", transform_json_path)
    print("Saved check image:", aligned_path)
    print("Saved overlay image:", overlay_path)
    print()
    print("Homography RGB -> Depth:")
    print(H)
    print()
    print("Reprojection error:")
    print("  mean  :", error_info["mean_error_px"])
    print("  median:", error_info["median_error_px"])
    print("  max   :", error_info["max_error_px"])

    cv2.imshow("RGB + Aligned Depth Overlay", overlay)
    cv2.waitKey(1)

    return True


def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if point_in_rect(x, y, button_undo):
        undo_last()
        return

    if point_in_rect(x, y, button_save):
        ok = save_homography()
        if ok:
            print("SAVE finished. You can close the window or click QUIT.")
        return

    if point_in_rect(x, y, button_quit):
        print("Quit clicked.")
        cv2.destroyAllWindows()
        raise SystemExit

    # 左側 RGB 画像クリック
    if x < rgb_disp_w:
        orig_x = int(round(x / rgb_scale))
        orig_y = int(round(y / rgb_scale))

        if 0 <= orig_x < rgb_w and 0 <= orig_y < rgb_h:
            points_rgb.append([orig_x, orig_y])
            print(f"RGB point {len(points_rgb)}: ({orig_x}, {orig_y})")

    # 右側 DEPTH 画像クリック
    elif x < rgb_disp_w + depth_disp_w:
        x_depth_disp = x - rgb_disp_w
        orig_x = int(round(x_depth_disp / depth_scale))
        orig_y = int(round(y / depth_scale))

        if 0 <= orig_x < depth_w and 0 <= orig_y < depth_h:
            points_depth.append([orig_x, orig_y])
            print(f"DEPTH point {len(points_depth)}: ({orig_x}, {orig_y})")


def main():
    global rgb, depth_vis
    global rgb_w, rgb_h, depth_w, depth_h
    global record_dir, args_global
    global rgb_scale, depth_scale

    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", type=str, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--rgb-scale", type=float, default=0.4)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--points-json", type=str, default="calibration_points.json")
    parser.add_argument("--transform-json", type=str, default="rgb_to_depth_homography.json")
    args = parser.parse_args()

    args_global = args
    rgb_scale = args.rgb_scale
    depth_scale = args.depth_scale
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
    print("RGB display scale:", rgb_scale)
    print("Depth display scale:", depth_scale)
    print()
    print("操作:")
    print("  左側RGB画像をクリック → RGB点")
    print("  右側DEPTH画像をクリック → DEPTH点")
    print("  下のUNDO/SAVE/QUITボタンをクリック")
    print()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        display = make_display_image()
        cv2.imshow(WINDOW_NAME, display)

        if cv2.waitKey(30) == 27:
            break

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()