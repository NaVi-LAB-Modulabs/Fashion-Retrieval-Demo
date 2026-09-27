"""Static taxonomy and Step1 common-attribute definitions for Fashion-200K.

Common attributes apply to all five Fashion-200K garment categories. Values are
English, lowercase identifiers shared by the VLM schema and graph node IDs.
The public constants, node format, schema structure, and metric fields are kept
compatible with the existing extractor. Category-specific visual attributes
remain in ``category_taxonomy.py``.

Category labels follow https://huggingface.co/datasets/Marqo/fashion200k
(``category1`` and ``category2``). Subcategories describe the source metadata;
they are not additional VLM predictions or new top-level graph categories.
They do not provide a normalized garment-type vocabulary. GARMENT_TYPE_VALUES
defines visual predictions extracted in Step2, independently of category2.
Image-view selection belongs to the loader, not this taxonomy.
"""

from __future__ import annotations

from typing import Any

COLORS = [
    "black",
    "white",
    "ivory",
    "beige",
    "gray",
    "brown",
    "navy",
    "blue",
    "light_blue",
    "green",
    "khaki",
    "yellow",
    "orange",
    "red",
    "burgundy",
    "pink",
    "purple",
    "silver",
    "gold",
]

# Keep materials with useful visible fabric or surface cues. An image cannot
# establish exact fiber content; use low confidence or unknown when uncertain.
MATERIALS = [
    "cotton",
    "linen",
    "denim",
    "chiffon",
    "silk",
    "wool",
    "lace",
    "leather",
    "suede",
    "tweed",
    "corduroy",
    "satin",
    "velvet",
    "knit",
    "fleece",
    "mesh",
    "fur",
    "unknown",
]

PATTERNS = [
    "argyle",
    "plaid_check",
    "stripes",
    "polka_dots",
    "floral",
    "animal_print",
    "geometric",
    "graphic",
    "camouflage",
    "none",
]

SEASONS = [
    "spring",
    "summer",
    "fall",
    "winter",
    "unknown",
]

CATEGORY_NAMES = {
    "dresses": "dresses",
    "jackets": "jackets",
    "pants": "pants",
    "skirts": "skirts",
    "tops": "tops",
}

# Exact source category2 labels, grouped under their category1 parent. Keep
# these distinct from visual attributes: e.g. a "cocktail dresses" source label
# does not prove that a particular material or silhouette is visible.
CATEGORY_SUBCATEGORIES = {
    "dresses": [
        "casual and day dresses",
        "cocktail dresses",
        "gowns",
        "maxi and long dresses",
        "mini and short dresses",
        "prom and formal dresses",
    ],
    "jackets": [
        "blazers and suit jackets",
        "casual jackets",
        "denim jackets",
        "fur jackets",
        "leather jackets",
        "padded and down jackets",
        "waistcoats and gilets",
    ],
    "pants": [
        "cargo pants",
        "cropped pants",
        "full length pants",
        "harem pants",
        "leggings",
        "skinny pants",
        "straight-leg pants",
        "wide-leg and palazzo pants",
    ],
    "skirts": [
        "knee length skirts",
        "maxi skirts",
        "mid length skirts",
        "mini skirts",
    ],
    "tops": [
        "blouses",
        "long sleeved tops",
        "shirts",
        "short sleeve tops",
        "sleeveless and tank tops",
        "t-shirts",
    ],
}

# Visual garment types, not source category2 labels. Candidate sets are disjoint
# and retain the source category1. In the local Marqo/fashion200k view-0 titles,
# cardigan and hoodie occur in jackets, not tops; waistcoats and gilets are a
# jackets category2. These guide candidate ownership, not automatic predictions.
# Based on fashion-how/codebook.wst.txt garment types (including KN: jersey).
# Sleeveless tops are the only additional type; finer styles share a broad type.
GARMENT_TYPE_VALUES = {
    "jackets": ["coat", "cardigan", "vest", "jacket", "hoodie"],
    "tops": [
        "jersey",
        "sweater",
        "shirt",
        "blouse",
        "sleeveless_top",
    ],
}

GARMENT_TYPE_DESCRIPTIONS = {
    "coat": "Substantial outerwear with coverage typically extending below the hips; length alone does not establish a coat.",
    "jacket": "Outerwear with jacket construction, including tailored blazers; excludes coats, cardigans, and vests.",
    "shirt": "A shirt-style top with a collar or functional placket and shirt construction, including polo shirts.",
    "blouse": "A blouse-style top with soft draping or decorative construction rather than a conventional shirt or T-shirt structure.",
    "jersey": "A casual jersey-style pullover, including T-shirts, sports jerseys, and non-hooded sweatshirts. Excludes visibly sweater-knit pullovers and polo shirts.",
    "sweater": "A knitted pullover without a full front opening.",
    "cardigan": "A knitted garment with a full front opening, whether buttoned, zipped, or open-front.",
    "hoodie": "A sweatshirt-style top with an attached hood; a hooded coat or jacket is not a hoodie.",
    "sleeveless_top": "A sleeveless top, including tank tops and narrow-strap camisoles; excludes layering vests.",
    "vest": "A sleeveless layering garment with vest construction, including waistcoats and sweater vests; not an ordinary tank top.",
}

COMMON_ATTRIBUTE_VALUES = {
    "colors": COLORS,
    "materials": MATERIALS,
    "patterns": PATTERNS,
    "seasons": SEASONS,
}

# A fallback cannot coexist with a supported value in the same group.
COMMON_ATTRIBUTE_EXCLUSIVE_VALUES = {
    "materials": {"unknown"},
    "patterns": {"none"},
    "seasons": {"unknown"},
}

COMMON_ATTRIBUTE_METRIC_FIELDS = {
    "colors": ["confidence", "coverage"],
    "patterns": ["confidence", "prominence"],
    "materials": ["confidence"],
    "seasons": ["confidence", "score"],
}


def common_attribute_nodes(values: list[str]) -> list[dict[str, str]]:
    return [{"id": value, "name": value} for value in values]


def common_attribute_schema() -> dict[str, Any]:
    properties = {}
    for group_name, fields in COMMON_ATTRIBUTE_METRIC_FIELDS.items():
        item_properties: dict[str, Any] = {
            "name": {
                "type": "string",
                "enum": list(COMMON_ATTRIBUTE_VALUES[group_name]),
            },
        }
        for field in fields:
            item_properties[field] = {"type": "number"}
        properties[group_name] = {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", *fields],
                "properties": item_properties,
            },
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(COMMON_ATTRIBUTE_METRIC_FIELDS),
        "properties": properties,
    }
