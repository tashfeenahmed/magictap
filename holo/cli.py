"""Command line entry point: `uv run holo <command>`."""

import argparse


def main():
    p = argparse.ArgumentParser(prog="holo", description="Camera-based touch input for a MacBook screen.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("cameras", help="list cameras and their indexes")

    sp = sub.add_parser("preview", help="show the camera view and detected body signals (setup)")
    sp.add_argument("--camera", type=int)
    sp.add_argument("--video", help="replay a recorded clip instead of the live camera")

    sp = sub.add_parser("calibrate", help="look at and touch dots so holo learns your eyes and body")
    sp.add_argument("--camera", type=int)
    sp.add_argument("--seed", type=int)
    sp.add_argument("--mode", choices=["cells", "point"], default="cells",
                    help="cells: reliable grid cells (default); point: estimate exact positions")
    sp.add_argument("--grid", default="4x3", help="cells mode: finest grid to calibrate, e.g. 4x3")
    sp.add_argument("--rounds", type=int, default=3, help="cells mode: touches per cell")
    sp.add_argument("--touch-samples", type=int, default=2, help="point mode: captures per touch dot")
    sp.add_argument("--auto", action="store_true", help="also start recording when you hold still (besides Space)")

    sp = sub.add_parser("buttons", help="big buttons on the best learned grid, with right/wrong feedback")
    sp.add_argument("--camera", type=int)
    sp.add_argument("--seed", type=int)

    sp = sub.add_parser("taps", help="teach holo what your screen taps sound like (vs. typing, talking, noise)")
    sp.add_argument("--seed", type=int)
    sp.add_argument("--retrain", action="store_true", help="retrain from the last saved tap recording instead")

    sp = sub.add_parser("retrain", help="retrain from a saved calibration recording (latest by default)")
    sp.add_argument("run", nargs="?", help="data/calibration_runs/<time> folder")

    sp = sub.add_parser("experiment", help="tap targets across the screen and measure accuracy")
    sp.add_argument("--camera", type=int)
    sp.add_argument("--cols", type=int, default=4)
    sp.add_argument("--rows", type=int, default=3)
    sp.add_argument("--radius", type=int, help="target radius in points (default: from calibration accuracy)")
    sp.add_argument("--seed", type=int)
    sp.add_argument("--reps", type=int, default=2, help="cells mode: times each cell is tested")

    sp = sub.add_parser("live", help="control the real mouse pointer (needs Accessibility)")
    sp.add_argument("--camera", type=int)
    sp.add_argument("--no-preview", action="store_true")
    sp.add_argument("--no-cursor", action="store_true", help="only click; don't move the cursor to where you look")

    args = p.parse_args()
    if args.cmd == "cameras":
        from .camera import list_cameras

        for i, name in enumerate(list_cameras()):
            print(i, name)
    elif args.cmd == "preview":
        from .preview import run_preview

        run_preview(args.camera, args.video)
    elif args.cmd == "calibrate":
        from .calibration import run_calibration

        if args.mode == "cells":
            from .calibration import run_cell_calibration

            cols, rows = map(int, args.grid.lower().split("x"))
            run_cell_calibration(args.camera, cols, rows, args.rounds, args.seed)
        else:
            run_calibration(args.camera, args.seed, args.auto, args.touch_samples)
    elif args.cmd == "buttons":
        from .buttons import run_buttons

        run_buttons(args.camera, args.seed)
    elif args.cmd == "taps":
        from .tapcal import retrain_taps, run_tap_calibration

        retrain_taps() if args.retrain else run_tap_calibration(args.seed)
    elif args.cmd == "retrain":
        from .calibration import RUNS_DIR, retrain

        runs = sorted(p for p in RUNS_DIR.iterdir() if (p / "dataset.npz").exists()) if RUNS_DIR.exists() else []
        if not args.run and not runs:
            raise SystemExit("No calibration recordings found.")
        retrain(args.run or runs[-1])
    elif args.cmd == "experiment":
        from .experiment import run_experiment

        run_experiment(args.camera, args.cols, args.rows, args.radius, args.seed, args.reps)
    elif args.cmd == "live":
        from .control import run_live

        run_live(args.camera, not args.no_preview, not args.no_cursor)


if __name__ == "__main__":
    main()
