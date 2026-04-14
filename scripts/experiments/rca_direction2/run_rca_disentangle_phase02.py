from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "rca_direction2" / "train_rca_disentangle_re2tt.py"
BASE_INIT = (
    PROJECT_ROOT
    / "checkpoints"
    / "experiments"
    / "rca_direction1"
    / "v6_root_head_re2tt_init_s42_e3"
    / "best_model.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print recommended Phase 0.2 experiment commands")
    parser.add_argument(
        "--mode",
        type=str,
        default="print",
        choices=["print"],
    )
    return parser.parse_args()


def build_cmd(
    tag: str,
    *,
    victim_label_mode: str,
    train_mode: str,
    victim_loss_weight: float,
    rank_loss_weight: float,
    victim_two_hop_weight: float = 0.5,
) -> str:
    save_dir = (
        PROJECT_ROOT
        / "checkpoints"
        / "experiments"
        / "rca_direction2"
        / tag
    )
    return (
        f"& 'D:\\anaconda\\envs\\paper_env\\python.exe' '{TRAIN_SCRIPT}' "
        f"--data-dir '{PROJECT_ROOT / 'data_rcaeval' / 'processed' / 're2-tt_lazy'}' "
        f"--save-dir '{save_dir}' "
        f"--init-checkpoint '{BASE_INIT}' "
        f"--selection-strategy first_3 "
        f"--seed 42 "
        f"--batch-size 4 "
        f"--epochs 20 "
        f"--patience 8 "
        f"--victim-label-mode {victim_label_mode} "
        f"--victim-two-hop-weight {victim_two_hop_weight} "
        f"--train-mode {train_mode} "
        f"--victim-loss-weight {victim_loss_weight} "
        f"--rank-loss-weight {rank_loss_weight}"
    )


def main() -> None:
    parse_args()
    commands = {
        "phase02_topology_only_headonly": build_cmd(
            "v6_disentangle_re2tt_phase02_topology_only_headonly_s42",
            victim_label_mode="topology_only",
            train_mode="disentangle_head_only",
            victim_loss_weight=1.0,
            rank_loss_weight=0.5,
        ),
        "phase02_topology_decay_headonly": build_cmd(
            "v6_disentangle_re2tt_phase02_topology_decay_headonly_s42",
            victim_label_mode="topology_decay",
            train_mode="disentangle_head_only",
            victim_loss_weight=1.0,
            rank_loss_weight=0.5,
            victim_two_hop_weight=0.25,
        ),
        "phase02_topology_decay_headspluscls": build_cmd(
            "v6_disentangle_re2tt_phase02_topology_decay_headspluscls_s42",
            victim_label_mode="topology_decay",
            train_mode="heads_plus_classifier",
            victim_loss_weight=1.0,
            rank_loss_weight=1.0,
            victim_two_hop_weight=0.25,
        ),
    }

    print("Recommended Phase 0.2 commands:\n")
    for name, cmd in commands.items():
        print(f"[{name}]")
        print(cmd)
        print()


if __name__ == "__main__":
    main()
