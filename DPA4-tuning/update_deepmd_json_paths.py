#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
import shutil
import re


def natural_key(s):
    """
    用于自然排序，例如 O24Sr8Ti8 排在 O120Sr40Ti40 前面。
    """
    s = str(s)
    return [
        int(x) if x.isdigit() else x
        for x in re.split(r"(\d+)", s)
    ]


def is_deepmd_system_dir(path: Path) -> bool:
    """
    判断一个目录是否是 DeePMD npy system 目录。

    典型结构：
        system/
        ├── type.raw
        ├── type_map.raw
        └── set.000/
            ├── box.npy
            ├── coord.npy
            ├── energy.npy
            ├── force.npy
            └── virial.npy
    """
    if not path.is_dir():
        return False

    if not (path / "type.raw").is_file():
        return False

    set_dirs = sorted(path.glob("set.*"))
    if len(set_dirs) == 0:
        return False

    required = [
        "box.npy",
        "coord.npy",
        "energy.npy",
        "force.npy",
    ]

    for set_dir in set_dirs:
        if not set_dir.is_dir():
            continue
        ok = True
        for name in required:
            if not (set_dir / name).is_file():
                ok = False
                break
        if ok:
            return True

    return False


def collect_systems(root: Path, relative_to: Path | None = None, use_absolute: bool = False):
    """
    收集 root 下所有 DeePMD system 目录。
    """
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    systems = []

    for p in root.iterdir():
        if is_deepmd_system_dir(p):
            systems.append(p)

    systems = sorted(systems, key=lambda x: natural_key(x.name))

    if len(systems) == 0:
        raise RuntimeError(f"No DeePMD system directories found under: {root}")

    out = []

    for p in systems:
        if use_absolute:
            out.append(str(p.resolve()))
        else:
            if relative_to is not None:
                try:
                    rp = p.resolve().relative_to(relative_to.resolve())
                    out.append("./" + str(rp))
                except ValueError:
                    out.append(str(p))
            else:
                out.append(str(p))

    return out


def update_existing_data_blocks(obj, train_systems, valid_systems):
    """
    递归查找 JSON 中的 training_data / validation_data，并更新 systems。
    """
    updated_train = 0
    updated_valid = 0

    def walk(x):
        nonlocal updated_train, updated_valid

        if isinstance(x, dict):
            for k, v in x.items():
                if k == "training_data" and isinstance(v, dict):
                    v["systems"] = train_systems
                    updated_train += 1

                elif k == "validation_data" and isinstance(v, dict):
                    v["systems"] = valid_systems
                    updated_valid += 1

                else:
                    walk(v)

        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)

    return updated_train, updated_valid


def force_create_training_blocks(obj, train_systems, valid_systems):
    """
    如果 JSON 里没有 training_data / validation_data，尝试在顶层 training 里面创建。
    """
    if "training" not in obj or not isinstance(obj["training"], dict):
        obj["training"] = {}

    if "training_data" not in obj["training"]:
        obj["training"]["training_data"] = {}

    if "validation_data" not in obj["training"]:
        obj["training"]["validation_data"] = {}

    obj["training"]["training_data"]["systems"] = train_systems
    obj["training"]["validation_data"]["systems"] = valid_systems


def main():
    parser = argparse.ArgumentParser(
        description="Automatically update DeePMD training_data/validation_data systems paths in JSON."
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input DeePMD json file, e.g. input_dpa4_air_finetune.json"
    )

    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output json file. If not set, overwrite input json."
    )

    parser.add_argument(
        "--data-root",
        default="deepmd_data",
        help="Root directory of DeePMD data. Default: deepmd_data"
    )

    parser.add_argument(
        "--train-subdir",
        default="training_data",
        help="Training data subdirectory under data-root. Default: training_data"
    )

    parser.add_argument(
        "--valid-subdir",
        default="validation_data",
        help="Validation data subdirectory under data-root. Default: validation_data"
    )

    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Use absolute paths in JSON."
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not make .bak backup when overwriting input json."
    )

    parser.add_argument(
        "--force-create",
        action="store_true",
        help="If training_data/validation_data blocks are not found, create them under top-level training."
    )

    args = parser.parse_args()

    input_json = Path(args.input)
    if not input_json.is_file():
        raise FileNotFoundError(f"Input json not found: {input_json}")

    output_json = Path(args.output) if args.output is not None else input_json

    workdir = input_json.parent.resolve()
    data_root = Path(args.data_root)

    if not data_root.is_absolute():
        data_root = workdir / data_root

    train_root = data_root / args.train_subdir
    valid_root = data_root / args.valid_subdir

    train_systems = collect_systems(
        train_root,
        relative_to=workdir,
        use_absolute=args.absolute,
    )

    valid_systems = collect_systems(
        valid_root,
        relative_to=workdir,
        use_absolute=args.absolute,
    )

    print("Found training systems:")
    for s in train_systems:
        print("  ", s)

    print()
    print("Found validation systems:")
    for s in valid_systems:
        print("  ", s)

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_train, updated_valid = update_existing_data_blocks(
        data,
        train_systems,
        valid_systems,
    )

    if updated_train == 0 or updated_valid == 0:
        msg = (
            f"Found training_data blocks: {updated_train}, "
            f"validation_data blocks: {updated_valid}."
        )

        if args.force_create:
            print()
            print("WARNING:", msg)
            print("Creating missing blocks under top-level 'training'.")
            force_create_training_blocks(data, train_systems, valid_systems)
        else:
            raise RuntimeError(
                msg
                + "\nCannot safely update JSON.\n"
                + "You can rerun with --force-create, or check your JSON structure."
            )

    if output_json == input_json and not args.no_backup:
        backup = input_json.with_suffix(input_json.suffix + ".bak")
        shutil.copy2(input_json, backup)
        print()
        print(f"Backup written to: {backup}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print()
    print(f"Updated JSON written to: {output_json}")
    print(f"Updated training_data blocks: {updated_train}")
    print(f"Updated validation_data blocks: {updated_valid}")


if __name__ == "__main__":
    main()
