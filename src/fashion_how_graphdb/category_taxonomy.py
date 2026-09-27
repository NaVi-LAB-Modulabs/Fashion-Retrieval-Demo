"""Fashion-200K category attributes and graph IDs used by search."""

from __future__ import annotations

from typing import Any

from .taxonomy import CATEGORY_NAMES, GARMENT_TYPE_DESCRIPTIONS, GARMENT_TYPE_VALUES

CATEGORY_ATTRIBUTE_DEFAULT_REQUIRED_FIELDS = ["name", "confidence", "prominence"]

CATEGORY_ATTRIBUTES: dict[str, dict[str, Any]] = {
    "pants": {
        "domain": "pants",
        "exclude_note": "Ignore attributes that belong to other garments, such as neckline, sleeve, collar, or shoulder.",
        "groups": {
            "pants_length": {
                "name": "pants length",
                "cardinality": "single",
                "values": [
                    "short_shorts",
                    "knee_length",
                    "capri",
                    "ankle_length",
                    "full_length",
                ],
                "value_descriptions": {
                    "short_shorts": "Hem reaches the upper to middle thigh.",
                    "knee_length": "Hem reaches just above or around the knee.",
                    "capri": "Hem falls below the knee to mid-calf.",
                    "ankle_length": "Hem exposes the ankle and does not cover the top of the foot.",
                    "full_length": "Hem covers the ankle or reaches the shoe.",
                },
                "required": ["name", "confidence"],
            },
            "pants_fit": {
                "name": "pants fit and silhouette",
                "cardinality": "single",
                "values": [
                    "skinny",
                    "straight_leg",
                    "wide_leg",
                    "bootcut",
                    "baggy",
                    "jogger",
                ],
                "required": ["name", "confidence"],
            },
            "pants_rise": {
                "name": "pants rise",
                "cardinality": "single",
                "values": ["low_rise", "mid_rise", "high_rise"],
                "required": ["name", "confidence"],
            },
            "pants_waist": {
                "name": "pants waist closure",
                "cardinality": "multi",
                "values": ["buttons", "zipper", "elastic_waist", "none"],
                "value_descriptions": {
                    "buttons": "Visible buttons function as a closure at the waist, front, or side.",
                    "zipper": "A visible zipper track or opening functions as a closure.",
                    "elastic_waist": "The waistband shows elastic gathering or a stretch-band structure.",
                    "none": "No button, zipper, or elastic waistband is visibly identifiable; select alone.",
                },
                "exclusive_values": ["none"],
                "required": ["name", "confidence"],
            },
            "pants_design": {
                "name": "pants design details",
                "cardinality": "multi",
                "values": [
                    "pintucks",
                    "pleats",
                    "cuffed_hem",
                    "cargo",
                    "washed",
                    "distressed",
                    "overalls",
                    "jumpsuit",
                    "none",
                ],
                "value_descriptions": {
                    "pintucks": "Narrow folds stitched into the waistband or front panel.",
                    "pleats": "Repeated folded pleats visibly shape or decorate the garment.",
                    "cuffed_hem": "Hem is visibly folded or rolled up.",
                    "cargo": "Large patch or flap pockets define the cargo design.",
                    "washed": "Fabric, especially denim, shows intentional fading or washed color effects.",
                    "distressed": "Intentional rips, holes, or frayed areas are visible.",
                    "overalls": "Pants have shoulder straps or a bib front.",
                    "jumpsuit": "The bodice and pants form a single garment.",
                    "none": "None of the listed design details is clearly visible; select alone.",
                },
                "exclusive_values": ["none"],
            },
            "pants_pocket": {
                "name": "pants pocket visibility",
                "cardinality": "single",
                "values": ["present", "none"],
            },
        },
    },
    "skirts": {
        "domain": "skirt",
        "exclude_note": "Ignore attributes that belong to other garments, such as neckline, sleeve, collar, shoulder, or pants leg shape.",
        "groups": {
            "skirt_length": {
                "name": "skirt length",
                "cardinality": "single",
                "values": ["mini", "knee_length", "midi", "maxi"],
                "value_descriptions": {
                    "mini": "Hem reaches the upper to middle thigh.",
                    "knee_length": "Hem reaches the lower thigh or knee area.",
                    "midi": "Hem falls below the knee to mid-calf.",
                    "maxi": "Hem reaches the lower calf or ankle area.",
                },
            },
            "skirt_silhouette": {
                "name": "skirt silhouette",
                "cardinality": "single",
                "values": ["a_line", "straight", "mermaid"],
                "value_descriptions": {
                    "a_line": "Skirt gradually widens from the waist toward the hem.",
                    "straight": "Width changes little from the hips to the hem, forming a straight outline.",
                    "mermaid": "Skirt is fitted at the hips or thighs and flares below the knees or near the hem.",
                },
            },
            "skirt_waist_fit": {
                "name": "skirt waist closure",
                "cardinality": "multi",
                "values": ["buttons", "zipper", "elastic_waist", "none"],
                "value_descriptions": {
                    "buttons": "Visible buttons function as a closure at the waist, front, or side.",
                    "zipper": "A visible zipper track or opening functions as a closure.",
                    "elastic_waist": "The waistband shows elastic gathering or a stretch-band structure.",
                    "none": "No button, zipper, or elastic waistband is visibly identifiable; select alone.",
                },
                "exclusive_values": ["none"],
            },
            "skirt_detail": {
                "name": "skirt details",
                "cardinality": "multi",
                "values": [
                    "pleats",
                    "ruffles",
                    "slit",
                    "pockets",
                    "lace",
                    "fringe",
                    "wrap",
                    "asymmetric",
                    "belt",
                    "sheer",
                ],
                "value_descriptions": {
                    "pleats": "Repeated folded pleats visibly shape or decorate the skirt.",
                    "ruffles": "Wavy or layered ruffled trim appears at the hem, waist, or seams.",
                    "slit": "A visible opening at the hem or a seam exposes the leg or inner space.",
                    "pockets": "A pocket opening, patch, or flap is visibly identifiable.",
                    "lace": "Decorative openwork, net, or floral lace fabric is visible.",
                    "fringe": "Loose threads, cords, or tassels hang from the hem or seams.",
                    "wrap": "One front panel overlaps the other in a wrap construction.",
                    "asymmetric": "Unequal hem lengths or asymmetric panel construction are clearly visible.",
                    "belt": "A belt, tie, or buckle is visible around the waist.",
                    "sheer": "Skin, lining, or background is visible through the fabric.",
                },
            },
        },
    },
    "jackets": {
        "domain": "outer",
        "groups": {
            "outer_type": {
                "name": "outerwear garment type",
                "cardinality": "single",
                "values": GARMENT_TYPE_VALUES["jackets"],
                "value_descriptions": {
                    value: GARMENT_TYPE_DESCRIPTIONS[value]
                    for value in GARMENT_TYPE_VALUES["jackets"]
                },
                "required": ["name", "confidence"],
            },
            "outer_collar": {
                "name": "jacket collar",
                "cardinality": "single",
                "values": ["collarless", "collared"],
            },
            "outer_sleeve": {
                "name": "jacket sleeve length",
                "cardinality": "single",
                "values": ["long_sleeve", "short_sleeve", "sleeveless"],
                "required": ["name", "confidence"],
            },
            "outer_fit": {
                "name": "jacket fit and silhouette",
                "cardinality": "single",
                "values": ["slim_fit", "regular_fit", "oversized"],
                "required": ["name", "confidence"],
            },
            "outer_length": {
                "name": "jacket length",
                "cardinality": "single",
                "values": ["cropped", "standard", "long"],
                "required": ["name", "confidence"],
            },
            "outer_closure": {
                "name": "jacket closure",
                "cardinality": "multi",
                "values": [
                    "single_breasted",
                    "double_breasted",
                    "zipper",
                    "belt",
                    "none",
                ],
                "value_descriptions": {
                    "single_breasted": "One row of functional buttons closes the overlapping front panels.",
                    "double_breasted": "Two parallel rows of buttons and a broad front overlap form a double-breasted closure.",
                    "zipper": "A zipper track, slider, or zipper opening is the main visible closure.",
                    "belt": "A belt ties or cinches the garment as a main closure, including a trench-coat belt.",
                    "none": "No closing device is visible, or the garment has an open-front design; select alone.",
                },
                "exclusive_values": ["none"],
                "required": ["name", "confidence"],
            },
            "outer_pocket": {
                "name": "jacket pocket location",
                "cardinality": "multi",
                "values": ["none", "chest_pockets", "waist_pockets", "lower_pockets"],
                "value_descriptions": {
                    "none": "No external pocket is visible; do not combine with pocket locations.",
                    "chest_pockets": "Pockets sit at chest height on the upper front panels.",
                    "waist_pockets": "Hand pockets sit around waist or hip height.",
                },
                "exclusive_values": ["none"],
            },
        },
    },
    "tops": {
        "domain": "top",
        "exclude_note": "Judge only the target top. Ignore outerwear, bottoms, accessories, and any inner or layered garment that is not the target item.",
        "prompt_guidance": """tops-specific visual decision rules:
- Treat neckline, sleeve length, fit, length, and closure as independent groups.
- For every single group, return exactly one best-supported value. When the image is ambiguous, choose the closest visible value and lower its confidence instead of inventing a value.
- For the detail group, return every clearly visible applicable detail and return an empty array when none is visible. Do not add a detail merely because it is common for the garment type.
- Determine neckline from the shape of the neck opening. Use collared only when a distinct folded or standing collar is visibly attached to the neckline.
- Determine sleeve length separately from sleeve shape. puff_sleeves and balloon_sleeves are details and may coexist with long_sleeve, short_sleeve, or three_quarter_sleeve.
- Determine fit from the garment silhouette, ease, body width, and shoulder position. Do not infer the wearer's body shape or use pose as evidence.
- Determine top length from the hem position relative to the natural waist and hips, not from image cropping.
- Select buttons or zipper only when a functional opening is visibly supported. Decorative hardware alone is not a closure.
- Select sheer only when skin, an underlayer, or the background is visibly discernible through a meaningful area of the fabric; sheen or light color alone is insufficient.""",
        "groups": {
            "top_type": {
                "name": "top garment type",
                "cardinality": "single",
                "values": GARMENT_TYPE_VALUES["tops"],
                "value_descriptions": {
                    value: GARMENT_TYPE_DESCRIPTIONS[value]
                    for value in GARMENT_TYPE_VALUES["tops"]
                },
                "required": ["name", "confidence"],
            },
            "top_neckline": {
                "name": "top neckline",
                "cardinality": "single",
                "values": [
                    "v_neck",
                    "round_neck",
                    "square_neck",
                    "boat_neck",
                    "halter",
                    "turtleneck",
                    "collared",
                ],
                "value_descriptions": {
                    "v_neck": "The opening descends diagonally toward the center chest in a V shape.",
                    "round_neck": "The neck opening forms a round or gently curved U shape.",
                    "square_neck": "A relatively horizontal lower edge and angular sides form a square opening.",
                    "boat_neck": "A wide, shallow opening extends horizontally toward the shoulders.",
                    "halter": "Straps or fabric around the neck support the top and expose the shoulders.",
                    "turtleneck": "Raised fabric surrounds the neck in a standing or folded high neckline.",
                    "collared": "A distinct folded or standing collar, such as a shirt or polo collar, is attached at the neckline.",
                },
            },
            "top_sleeve_length": {
                "name": "top sleeve length",
                "cardinality": "single",
                "values": [
                    "long_sleeve",
                    "short_sleeve",
                    "sleeveless",
                    "three_quarter_sleeve",
                ],
                "value_descriptions": {
                    "long_sleeve": "Sleeves extend past the elbow to near the wrist.",
                    "short_sleeve": "Sleeves end on the upper arm or above the elbow.",
                    "sleeveless": "No sleeve covers the arm; the garment ends at the armhole, including vest-style tops.",
                    "three_quarter_sleeve": "Sleeves end below the elbow and above the wrist, usually around mid-forearm.",
                },
            },
            "top_fit": {
                "name": "top fit and silhouette",
                "cardinality": "single",
                "values": ["slim_fit", "regular_fit", "oversized"],
                "value_descriptions": {
                    "slim_fit": "The body and sleeves closely follow the body with little ease.",
                    "regular_fit": "The body and shoulders have moderate ease without a tight or intentionally wide silhouette.",
                    "oversized": "The body and sleeves have substantial ease, often with dropped shoulders or a visibly wide cut.",
                },
            },
            "top_length": {
                "name": "top length",
                "cardinality": "single",
                "values": ["cropped", "standard", "long"],
                "value_descriptions": {
                    "cropped": "The hem sits at or above the natural waist and may expose the midriff.",
                    "standard": "The hem sits below the waist around the upper hip.",
                    "long": "The hem covers much of the hips or extends below them in a tunic-like length.",
                },
            },
            "top_closure": {
                "name": "top closure",
                "cardinality": "single",
                "values": ["buttons", "zipper", "none"],
                "value_descriptions": {
                    "buttons": "Visible functional buttons open and close the main garment opening.",
                    "zipper": "A visible zipper track and slider open and close the main garment opening.",
                    "none": "No functional button or zipper opening is visible, as in a pullover.",
                },
            },
            "top_detail": {
                "name": "top decorative details",
                "cardinality": "multi",
                "values": [
                    "ruffles",
                    "bow",
                    "ruching",
                    "lace",
                    "tie",
                    "embroidery",
                    "puff_sleeves",
                    "balloon_sleeves",
                    "upper_pockets",
                    "lower_pockets",
                    "hood",
                    "sheer",
                ],
                "value_descriptions": {
                    "ruffles": "Wavy or layered ruffled trim follows a fabric edge or seam.",
                    "bow": "A ribbon or fabric strip is visibly tied or formed into a bow.",
                    "ruching": "Stitching or elastic gathers fabric into repeated small folds and volume.",
                    "lace": "Decorative openwork, net, or floral lace structure is visible.",
                    "tie": "A distinct long strip at the neckline or front is designed to hang or tie.",
                    "embroidery": "Thread-stitched motifs, lettering, or logos are visible on the fabric surface.",
                    "puff_sleeves": "Gathering at the shoulder or sleeve head creates localized rounded volume.",
                    "balloon_sleeves": "The sleeve body has substantial volume and narrows again at the cuff.",
                    "upper_pockets": "Pockets sit on the upper front of the top around chest height.",
                    "lower_pockets": "Pockets sit on the lower front of the top around the waist or hem.",
                    "hood": "An attached fabric hood behind the neckline can cover the head.",
                    "sheer": "Skin, an underlayer, or background is visible through a meaningful area of fabric.",
                },
            },
        },
    },
    "dresses": {
        "domain": "dress",
        "exclude_note": "Judge only the target dress. Ignore jackets, separate tops, separate bottoms, accessories, and background styling.",
        "groups": {
            "dress_length": {
                "name": "dress length",
                "cardinality": "single",
                "values": ["mini", "knee_length", "midi", "maxi"],
                "value_descriptions": {
                    "mini": "Hem sits above the knee.",
                    "knee_length": "Hem sits around the knee.",
                    "midi": "Hem sits below the knee and above the ankle.",
                    "maxi": "Hem reaches the ankle or floor.",
                },
                "required": ["name", "confidence"],
            },
            "dress_silhouette": {
                "name": "dress silhouette",
                "cardinality": "single",
                "values": ["fitted", "straight", "flared"],
                "value_descriptions": {
                    "fitted": "The dress closely follows the body through most of its length.",
                    "straight": "The dress falls with little change in width and does not closely follow the body.",
                    "flared": "The dress becomes visibly wider from the waist or upper body toward the hem.",
                },
                "required": ["name", "confidence"],
            },
            "dress_neckline": {
                "name": "dress neckline",
                "cardinality": "single",
                "values": [
                    "v_neck",
                    "round_neck",
                    "square_neck",
                    "boat_neck",
                    "halter",
                    "collared",
                    "strapless",
                ],
                "required": ["name", "confidence"],
            },
            "dress_sleeve_length": {
                "name": "dress sleeve length",
                "cardinality": "single",
                "values": [
                    "sleeveless",
                    "short_sleeve",
                    "three_quarter_sleeve",
                    "long_sleeve",
                ],
                "required": ["name", "confidence"],
            },
            "dress_waist": {
                "name": "dress waist",
                "cardinality": "single",
                "values": [
                    "defined_waist",
                    "dropped_waist",
                    "empire_waist",
                    "straight_waist",
                ],
                "required": ["name", "confidence"],
            },
            "dress_detail": {
                "name": "dress detail",
                "cardinality": "multi",
                "values": [
                    "pleats",
                    "ruffles",
                    "lace",
                    "mesh",
                    "cutout",
                    "buttons",
                    "belt",
                    "pockets",
                    "slit",
                    "none",
                ],
                "exclusive_values": ["none"],
            },
        },
    },
}


def category_attribute_id(group_id: str, name: str) -> str:
    return f"{group_id}:{name}"
