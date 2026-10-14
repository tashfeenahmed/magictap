"""Command line entry point: `magictap <command>`."""

import argparse

from . import __version__


def main():
    p = argparse.ArgumentParser(
        prog="magictap",
        description="Magic Tap: touch input for a laptop screen with no touchscreen. "
                    "Your eyes pick the button, the sound of your tap says when.",
        epilog="First time? Run `magictap setup`, then `magictap buttons`. New session? `magictap recenter` (15 s).")
    p.add_argument("--version", action="version", version=f"magictap {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # --- get started
    sub.add_parser("doctor", help="check camera, microphone, permissions and models")
    sp = sub.add_parser("setup", help="guided first run: doctor, then calibration, then tap training")
    sp.add_argument("--camera", type=int)
    sp = sub.add_parser("calibrate", help="look at and touch dots so Magic Tap learns your eyes (~6 min)")
    sp.add_argument("--camera", type=int)
    sp.add_argument("--seed", type=int)
    sp.add_argument("--mode", choices=["cells", "point"], default="cells",
                    help="cells: reliable grid cells (default); point: estimate exact positions")
    sp.add_argument("--grid", default="4x3", help="cells mode: finest grid to calibrate, e.g. 4x3")
    sp.add_argument("--rounds", type=int, default=3, help="cells mode: touches per cell")
    sp.add_argument("--touch-samples", type=int, default=2, help="point mode: captures per touch dot")
    sp.add_argument("--auto", action="store_true", help="point mode: also record when you hold still")
    sp = sub.add_parser("recenter", help="15-second re-centre at the start of a session (removes calibration drift)")
    sp.add_argument("--camera", type=int)
    sp.add_argument("--seed", type=int)
    sp = sub.add_parser("taps", help="teach Magic Tap your screen taps vs. typing, talking and noise (~2 min)")
    sp.add_argument("--seed", type=int)
    sp.add_argument("--retrain", action="store_true", help="retrain from the last saved tap recording instead")

    # --- try it / use it
    sp = sub.add_parser("buttons", help="try it: big on-screen buttons with right/wrong feedback")
    sp.add_argument("--camera", type=int)
    sp.add_argument("--seed", type=int)
    sp = sub.add_parser("mouse", help="use it in any app: tap the screen to click where you look (needs Accessibility)")
    sp.add_argument("--camera", type=int)
    sp.add_argument("--min-confidence", type=float, default=0.6)
    sp = sub.add_parser("serve", help="integrate it: local WebSocket service for apps (ws://127.0.0.1:8765)")
    sp.add_argument("--camera", type=int)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--no-camera", action="store_true", help="start without camera/mic (protocol testing)")

    # --- tools
    sub.add_parser("cameras", help="list cameras and their indexes")
    sp = sub.add_parser("preview", help="camera view with face, eyes and body overlay")
    sp.add_argument("--camera", type=int)
    sp.add_argument("--video", help="replay a recorded clip instead of the live camera")
    sp = sub.add_parser("retrain", help="retrain from a saved calibration recording (latest by default)")
    sp.add_argument("run", nargs="?", help="calibration_runs/<time> folder")
    sp = sub.add_parser("experiment", help="measure accuracy on a grid of targets")
    sp.add_argument("--camera", type=int)
    sp.add_argument("--cols", type=int, default=4)
    sp.add_argument("--rows", type=int, default=3)
    sp.add_argument("--radius", type=int, help="target radius in points (default: from calibration accuracy)")
    sp.add_argument("--seed", type=int)
    sp.add_argument("--reps", type=int, default=2, help="cells mode: times each cell is tested")
    sp = sub.add_parser("evaluate-demo", help="offline accuracy on a recorded demo session (with/without re-centre)")
    sp.add_argument("run", nargs="?", help="demo_runs/<time> folder (latest by default)")
    sp = sub.add_parser("record-demo", help="guided session that records the camera and press times for a demo video")
    sp.add_argument("--camera", type=int)
    sp.add_argument("--taps", type=int, default=24)
    sp.add_argument("--seed", type=int)

    args = p.parse_args()
    if args.cmd == "doctor":
        from .doctor import run_doctor

        run_doctor()
    elif args.cmd == "setup":
        from .doctor import run_setup

        run_setup(args.camera)
    elif args.cmd == "calibrate":
        if args.mode == "cells":
            from .calibration import run_cell_calibration

            cols, rows = map(int, args.grid.lower().split("x"))
            run_cell_calibration(args.camera, cols, rows, args.rounds, args.seed)
        else:
            from .calibration import run_calibration

            run_calibration(args.camera, args.seed, args.auto, args.touch_samples)
    elif args.cmd == "recenter":
        from .recenter import run_recenter

        run_recenter(args.camera, args.seed)
    elif args.cmd == "evaluate-demo":
        from .evaluate import run_evaluate_demo

        run_evaluate_demo(args.run)
    elif args.cmd == "taps":
        from .tapcal import retrain_taps, run_tap_calibration

        retrain_taps() if args.retrain else run_tap_calibration(args.seed)
    elif args.cmd == "buttons":
        from .buttons import run_buttons

        run_buttons(args.camera, args.seed)
    elif args.cmd == "mouse":
        from .mouse import run_mouse

        run_mouse(args.camera, args.min_confidence)
    elif args.cmd == "serve":
        from .serve import run_serve

        run_serve(args.camera, args.host, args.port, args.no_camera)
    elif args.cmd == "cameras":
        from .camera import list_cameras

        for i, name in enumerate(list_cameras()):
            print(i, name)
    elif args.cmd == "preview":
        from .preview import run_preview

        run_preview(args.camera, args.video)
    elif args.cmd == "retrain":
        from .calibration import RUNS_DIR, retrain

        runs = sorted(p for p in RUNS_DIR.iterdir() if (p / "dataset.npz").exists()) if RUNS_DIR.exists() else []
        if not args.run and not runs:
            raise SystemExit("No calibration recordings found.")
        retrain(args.run or runs[-1])
    elif args.cmd == "experiment":
        from .experiment import run_experiment

        run_experiment(args.camera, args.cols, args.rows, args.radius, args.seed, args.reps)
    elif args.cmd == "record-demo":
        from .demo import run_record_demo

        run_record_demo(args.camera, args.taps, args.seed)


if __name__ == "__main__":
    main()
