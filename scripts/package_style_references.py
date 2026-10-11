"""Package the existing human-selected calibration images for portable deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def package(source: Path, destination: Path) -> None:
    module_path = source / "src/fashion_200k_graphdb/fashion200k_global_axes.py"
    # Read constants without importing the sibling project's runtime dependencies.
    import ast
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    specs = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"DEFAULT_AXES", "STYLE_REFERENCE_AXES"}:
                    class ExpandLevels(ast.NodeTransformer):
                        def visit_Call(self, call):
                            if (isinstance(call.func, ast.Name) and call.func.id == "list"
                                    and len(call.args) == 1 and isinstance(call.args[0], ast.Name)
                                    and call.args[0].id == "STYLE_REFERENCE_LEVELS"):
                                return ast.List(elts=[ast.Constant(level) for level in
                                    ("0.00", "0.25", "0.50", "0.75", "1.00")], ctx=ast.Load())
                            raise ValueError("Unsupported expression in axis specification")
                    specs[target.id] = ast.literal_eval(ExpandLevels().visit(node.value))
    calibration = source / "src/fashion_200k_graphdb/axis_calibration"
    anchors = json.loads((calibration / "anchors.json").read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    axes = []
    for spec in [*specs["DEFAULT_AXES"], *specs["STYLE_REFERENCE_AXES"]]:
        axis = spec["id"]
        levels = list(spec["score_hints"])
        rows = []
        for level in levels:
            if axis in anchors["selections"]:
                examples = []
                for item_id in anchors["selections"][axis][level]:
                    with Image.open(calibration / "images" / f"{item_id}.jpg") as image:
                        examples.append((item_id, image.convert("RGB")))
            else:
                path = source / "src/fashion_200k_graphdb/style_references" / spec["reference_image"]
                with Image.open(path) as atlas:
                    row = levels.index(level)
                    examples = [(f"{axis}_{level.replace('.', '_')}_ref_{col + 1}",
                                 atlas.crop((98 + col * 152, 40 + row * 194,
                                             98 + (col + 1) * 152, 40 + row * 194 + 190)).convert("RGB"))
                                for col in range(4)]
            rows.append((level, examples))
        columns = 5 if axis in anchors["selections"] else 4
        subrows = 2 if columns == 5 else 1
        cell_width, cell_height, label_width = 160, 190, 70
        sheet = Image.new("RGB", (label_width + columns * cell_width,
                                   40 + len(levels) * subrows * cell_height), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 10), axis, fill="black")
        references = []
        for row, (level, examples) in enumerate(rows):
            draw.text((8, 50 + row * subrows * cell_height), f"{float(level):.2f}", fill="black")
            for index, (ref_id, image) in enumerate(examples):
                x = label_width + (index % columns) * cell_width
                y = 40 + (row * subrows + index // columns) * cell_height
                thumb = ImageOps.contain(image, (cell_width - 8, cell_height - 25))
                sheet.paste(thumb, (x + (cell_width - thumb.width) // 2, y))
                caption = ref_id if columns == 5 else f"ref_{index + 1}"
                draw.text((x + 4, y + cell_height - 20), caption, fill="black")
                references.append({"id": ref_id, "level": float(level)})
        image_path = destination / f"{axis}.jpg"
        sheet.save(image_path, quality=92)
        axes.append({"id": axis, "label": spec["english_name"], "question": spec["question"],
                     "score_hints": spec["score_hints"], "neutral_hint": spec.get("neutral_hint", ""),
                     "levels": [float(level) for level in levels], "references": references,
                     "image": image_path.name, "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
    manifest = {"version": "query_style_references_v1", "source": "Fashion-Retrieval/fashion200k_global_axes.py",
                "source_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
                "gallery_id": anchors["gallery_id"], "axes": axes}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Packaged {len(axes)} axes and {sum(len(a['references']) for a in axes)} references")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "web/style-references")
    args = parser.parse_args()
    package(args.source, args.destination)
