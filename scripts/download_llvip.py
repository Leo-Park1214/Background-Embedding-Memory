from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

from tqdm import tqdm

SCENES: Tuple[str, ...] = ("19", "20", "21", "22", "23", "24")
MODALITIES: Tuple[str, ...] = ("visible", "infrared")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DOWNLOADER_VERSION = "2026-06-06-scene-filter-v3"


def _extract_archives(root: Path) -> None:
    for archive in root.rglob("*.zip"):
        marker = archive.with_suffix(archive.suffix + ".extracted")
        if marker.exists():
            continue
        with zipfile.ZipFile(archive, "r") as handle:
            handle.extractall(archive.parent)
        marker.touch()

    try:
        import py7zr  # type: ignore
    except ImportError:
        return

    for archive in root.rglob("*.7z"):
        marker = archive.with_suffix(archive.suffix + ".extracted")
        if marker.exists():
            continue
        with py7zr.SevenZipFile(archive, "r") as handle:
            handle.extractall(path=archive.parent)
        marker.touch()


def download_from_hf(raw_root: Path) -> Path:
    raw_root.mkdir(parents=True, exist_ok=True)

    # Reuse an already downloaded/extracted LLVIP tree. This avoids downloading
    # the 4 GB snapshot again after a layout-conversion failure.
    try:
        existing_root = _locate_llvip_root(raw_root)
        print(f"[Info] Reusing existing LLVIP files under: {existing_root}")
        return existing_root
    except FileNotFoundError:
        pass

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="UserNae3/LLVIP",
        repo_type="dataset",
        local_dir=raw_root.as_posix(),
    )
    _extract_archives(raw_root)
    return _locate_llvip_root(raw_root)


def download_from_official(raw_root: Path, links_file: Path) -> Path:
    import gdown
    import requests

    if not links_file.exists():
        raise FileNotFoundError(f"Official-link file not found: {links_file}")

    raw_root.mkdir(parents=True, exist_ok=True)
    for index, line in enumerate(links_file.read_text(encoding="utf-8").splitlines()):
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        name = url.split("?")[0].rstrip("/").split("/")[-1] or f"download_{index}"
        destination = raw_root / name
        if "drive.google.com" in url:
            gdown.download(url=url, output=destination.as_posix(), quiet=False, fuzzy=True)
        else:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with destination.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)

    _extract_archives(raw_root)
    return _locate_llvip_root(raw_root)


def _find_named_dir(root: Path, names: Iterable[str]) -> Optional[Path]:
    lowered = {name.lower() for name in names}
    candidates = [path for path in root.rglob("*") if path.is_dir() and path.name.lower() in lowered]
    if not candidates:
        return None
    # Prefer the shallowest matching directory.
    return min(candidates, key=lambda path: len(path.parts))


def _locate_llvip_root(search_root: Path) -> Path:
    annotations = _find_named_dir(search_root, ("Annotations", "annotations", "Annotation"))
    visible = _find_named_dir(search_root, ("visible", "Visible", "VI", "vi"))
    infrared = _find_named_dir(search_root, ("infrared", "Infrared", "IR", "ir"))
    if annotations is None or visible is None or infrared is None:
        raise FileNotFoundError(
            "Could not locate LLVIP Annotations, visible, and infrared directories under "
            f"{search_root}."
        )

    common = Path(*Path(annotations).parts[:0])
    # Find the nearest common parent of all three components.
    common_parts = []
    for parts in zip(annotations.parts, visible.parts, infrared.parts):
        if len(set(parts)) == 1:
            common_parts.append(parts[0])
        else:
            break
    if not common_parts:
        raise RuntimeError("LLVIP components do not share a common root.")
    return Path(*common_parts)


def _locate_components(llvip_root: Path) -> Tuple[Path, Dict[str, Path]]:
    annotations = _find_named_dir(llvip_root, ("Annotations", "annotations", "Annotation"))
    visible = _find_named_dir(llvip_root, ("visible", "Visible", "VI", "vi"))
    infrared = _find_named_dir(llvip_root, ("infrared", "Infrared", "IR", "ir"))
    if annotations is None or visible is None or infrared is None:
        raise FileNotFoundError("LLVIP component directories are missing after extraction.")
    return annotations, {"visible": visible, "infrared": infrared}


def _scene_from_filename(path: Path) -> Optional[str]:
    """Map LLVIP names such as 190001.jpg to scene 19.

    LLVIP also contains sequences outside the paper's selected scenes, such as
    260001.jpg. Those files intentionally return None and are skipped.
    """
    match = re.match(r"^(\d{2})\d+", path.stem)
    if not match:
        return None
    scene = match.group(1)
    return scene if scene in SCENES else None


def _split_from_path(path: Path, modality_root: Path) -> Optional[str]:
    try:
        parts = [part.lower() for part in path.relative_to(modality_root).parts]
    except ValueError:
        parts = [part.lower() for part in path.parts]
    if "train" in parts:
        return "train"
    if "test" in parts or "val" in parts or "validation" in parts:
        return "val"
    return None


def _collect_required_images(modality_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    grouped: Dict[str, Dict[str, List[Path]]] = {
        scene: {"train": [], "val": []} for scene in SCENES
    }

    for image in sorted(modality_root.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        split = _split_from_path(image, modality_root)
        if split is None:
            continue
        scene = _scene_from_filename(image)
        if scene is None:
            # Example: 260001.jpg belongs to sequence 26, not to scenes 19--24.
            continue
        grouped[scene][split].append(image)

    missing_scenes = [
        scene for scene in SCENES
        if not grouped[scene]["train"] and not grouped[scene]["val"]
    ]
    if missing_scenes:
        raise RuntimeError(
            f"No images were found for required scenes: {', '.join(missing_scenes)} in {modality_root}. "
            "Scene assignment uses the first two filename digits."
        )
    return grouped


def _annotation_index(annotation_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for xml_path in annotation_root.rglob("*.xml"):
        index.setdefault(xml_path.stem, xml_path)
    if not index:
        raise RuntimeError(f"No XML annotations found under {annotation_root}")
    return index


def voc_to_yolo(xml_path: Path) -> str:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing image size in {xml_path}")
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    lines: List[str] = []
    for obj in root.findall("object"):
        if (obj.findtext("name", "").strip().lower()) != "person":
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.findtext("xmin", "0"))
        ymin = float(bbox.findtext("ymin", "0"))
        xmax = float(bbox.findtext("xmax", "0"))
        ymax = float(bbox.findtext("ymax", "0"))

        xmin = min(max(xmin, 0.0), width)
        xmax = min(max(xmax, 0.0), width)
        ymin = min(max(ymin, 0.0), height)
        ymax = min(max(ymax, 0.0), height)
        if xmax <= xmin or ymax <= ymin:
            continue

        cx = ((xmin + xmax) / 2.0) / width
        cy = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.hardlink_to(source)
    except (OSError, FileExistsError):
        shutil.copy2(source, destination)


def _prepare_scene(
    output_root: Path,
    modality: str,
    scene: str,
    images_by_split: Dict[str, List[Path]],
    annotations: Dict[str, Path],
    overwrite: bool,
) -> Path:
    yolo_root = output_root / modality / scene / "yolo"
    if overwrite and yolo_root.exists():
        shutil.rmtree(yolo_root)

    for split in ("train", "val"):
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        for image in tqdm(images_by_split[split], desc=f"{modality}-scene{scene}-{split}"):
            annotation = annotations.get(image.stem)
            if annotation is None:
                raise FileNotFoundError(f"Missing annotation for {image.name}")
            image_out = yolo_root / "images" / split / image.name
            label_out = yolo_root / "labels" / split / f"{image.stem}.txt"
            if not image_out.exists():
                _copy_or_link(image, image_out)
            label_out.write_text(voc_to_yolo(annotation), encoding="utf-8")

    (yolo_root / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {yolo_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/val",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yolo_root


def build_scene_layouts(llvip_root: Path, data_root: Path, overwrite: bool) -> None:
    annotation_root, modality_roots = _locate_components(llvip_root)
    annotations = _annotation_index(annotation_root)

    print("[Info] Only scenes 19,20,21,22,23,24 are selected; all other LLVIP sequences are skipped.")
    for modality in MODALITIES:
        grouped = _collect_required_images(modality_roots[modality])
        for scene in SCENES:
            yolo_root = _prepare_scene(
                data_root, modality, scene, grouped[scene], annotations, overwrite
            )
            print(
                f"[Done] {modality}/scene{scene}: "
                f"train={len(grouped[scene]['train'])}, val={len(grouped[scene]['val'])} -> {yolo_root}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download LLVIP and create visible/infrared scene 19--24 YOLO datasets."
    )
    parser.add_argument("--source", choices=("hf", "official"), default="hf")
    parser.add_argument("--links", type=Path, default=Path("links.txt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/LLVIP"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"[Info] download_llvip.py version: {DOWNLOADER_VERSION}")
    data_root = args.data_root.resolve()
    raw_root = data_root / "_raw"
    llvip_root = (
        download_from_hf(raw_root)
        if args.source == "hf"
        else download_from_official(raw_root, args.links)
    )
    build_scene_layouts(llvip_root, data_root, overwrite=args.overwrite)

    print("\nEvaluation command:")
    print(
        "python scripts/run_llvip.py "
        f"--data-root {data_root} --scenes 19,20,21,22,23,24 "
        "--modalities visible,infrared"
    )


if __name__ == "__main__":
    main()
