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
    "jackets": ["coat", "jacket", "parka", "puffer", "cardigan", "vest", "hoodie"],
    "tops": [
        "t-shirt",
        "sweatshirt",
        "bustier",
        "sweater",
        "shirt",
        "blouse",
        "sleeveless_top",
    ],
}

GARMENT_TYPE_DESCRIPTIONS = {
    "coat": "Outerwear with coat construction and coverage typically extending below the hips, including trench coats and pea coats. Excludes parkas and visibly lofted puffers; a long puffer remains puffer. Length alone does not establish a coat.",
    "jacket": "Outerwear with jacket construction, including tailored blazers, bomber jackets, biker jackets, denim jackets, and non-puffer quilted jackets. Excludes coats, parkas, puffers, cardigans, vests, and sweatshirt-style hoodies.",
    "parka": "A sleeved utilitarian outerwear garment, usually covering the hips or thighs, often with a hood, substantial pockets, and a drawstring or cinched waist or hem. A hood alone does not establish a parka. Select puffer instead when the target has a visibly lofted padded body.",
    "puffer": "A sleeved outerwear garment with a visibly lofted padded body, often arranged in voluminous filled baffles; includes short puffer jackets and long puffer coats. Flat quilting, padded shoulders, or a padded collar alone do not establish a puffer. Sleeveless padded layering garments are vest. Do not infer whether the filling is down or synthetic.",
    "cardigan": "A knitted garment with sleeves and a full front opening, whether buttoned, zipped, or open-front. Visible knit construction is required; an open front alone does not establish a cardigan. Sleeveless knitted layering garments are vest.",
    "vest": "A sleeveless layering garment with vest construction, including waistcoats, sweater vests, padded vests, and fur vests; not an ordinary tank top. Select vest for sleeveless padded outerwear rather than puffer. Ignore sleeves belonging to an inner garment.",
    "hoodie": "A sweatshirt-style top with an attached hood, including zip-front hoodies. A hooded coat, parka, puffer, or jacket is not a hoodie; do not classify an inner hoodie layered beneath the target outerwear.",
    "t-shirt": "A casual lightweight jersey-knit top with T-shirt construction, typically featuring a crew neck, V-neck, or similar simple neckline. Includes short-sleeved and long-sleeved T-shirts, graphic tees, and sports jerseys with T-shirt construction. Excludes heavier sweatshirt-style pullovers, visibly sweater-knit garments, collared shirts, and sleeveless tops.",
    "sweatshirt": "A casual pullover with sweatshirt construction, typically made from medium- to heavyweight knit fabric such as fleece or French terry, often with ribbed cuffs, collar, or hem. Includes crewneck sweatshirts and non-hooded athletic sweatshirts. Excludes hoodies, lightweight T-shirts, and visibly sweater-knit pullovers. Ribbed trim alone does not establish a sweatshirt.",
    "bustier": "A fitted top with structured bust or torso construction, often featuring shaped cups, corset-like seams, boning, or a defined bodice. Includes sleeveless and strapless bustier tops. Excludes ordinary tank tops, camisoles, and fitted sleeveless tops without visible bustier construction. Tight fit or cropped length alone does not establish a bustier.",
    "sweater": "A visibly knitted pullover without a full front opening, including fine-knit, ribbed-knit, cable-knit, and chunky-knit sweaters. Includes short-sleeved and sleeveless sweater-style pullovers when the garment has sweater construction. Excludes cardigans, sweatshirt-style pullovers, and lightweight T-shirts. Ribbed fabric alone does not establish a sweater.",
    "shirt": "A shirt-style top with conventional shirt construction, typically featuring a structured collar, button placket, or both. Includes dress shirts, Oxford shirts, casual button-down shirts, and polo shirts. Excludes soft draped blouses, T-shirts, and sweatshirt-style pullovers. A collar alone does not establish a shirt.",
    "blouse": "A top with blouse-style construction, typically featuring soft draping, lightweight woven fabric, gathers, ruffles, pleats, or decorative neckline and sleeve details. Includes both collared and collarless blouses. Excludes conventional button-down shirts, basic T-shirts, and structured bustier tops. Fabric softness or decorative details alone do not establish a blouse.",
    "sleeveless_top": "A top without sleeves, including tank tops, camisoles, halter tops, and simple strapless or tube tops. Excludes structured bustiers, sleeveless layering vests, and visibly sweater-knit pullovers. Select bustier when the garment has structured bust or corset-like construction. Sleeveless design alone does not override a more specific garment type.",
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
