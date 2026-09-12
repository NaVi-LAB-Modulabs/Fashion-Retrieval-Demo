"""Fashion-200K category attribute catalog and graph IDs used by search."""

CATEGORY_ATTRIBUTES = {
    "pants": {
        "domain": "pants",
        "groups": {
            "pants_length": {
                "name": "pants length",
                "cardinality": "single",
                "values": [
                    "short_shorts",
                    "knee_length",
                    "capri",
                    "ankle_length",
                    "full_length"
                ]
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
                    "jogger"
                ]
            },
            "pants_rise": {
                "name": "pants rise",
                "cardinality": "single",
                "values": [
                    "low_rise",
                    "mid_rise",
                    "high_rise"
                ]
            },
            "pants_waist": {
                "name": "pants waistband",
                "cardinality": "multi",
                "values": [
                    "elastic_waist",
                    "drawstring",
                    "none"
                ]
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
                    "none"
                ]
            },
            "pants_pocket": {
                "name": "pants pocket visibility",
                "cardinality": "single",
                "values": [
                    "present",
                    "none"
                ]
            }
        }
    },
    "skirts": {
        "domain": "skirt",
        "groups": {
            "skirt_length": {
                "name": "skirt length",
                "cardinality": "single",
                "values": [
                    "mini",
                    "knee_length",
                    "midi",
                    "maxi"
                ]
            },
            "skirt_silhouette": {
                "name": "skirt silhouette",
                "cardinality": "single",
                "values": [
                    "a_line",
                    "straight",
                    "mermaid"
                ]
            },
            "skirt_waist_fit": {
                "name": "skirt waist closure",
                "cardinality": "multi",
                "values": [
                    "buttons",
                    "zipper",
                    "elastic_waist",
                    "none"
                ]
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
                    "sheer"
                ]
            }
        }
    },
    "jackets": {
        "domain": "outer",
        "groups": {
            "outer_collar": {
                "name": "jacket collar",
                "cardinality": "single",
                "values": [
                    "collarless",
                    "collared"
                ]
            },
            "outer_sleeve": {
                "name": "jacket sleeve length",
                "cardinality": "single",
                "values": [
                    "long_sleeve",
                    "short_sleeve",
                    "sleeveless"
                ]
            },
            "outer_fit": {
                "name": "jacket fit and silhouette",
                "cardinality": "single",
                "values": [
                    "slim_fit",
                    "regular_fit",
                    "oversized"
                ]
            },
            "outer_length": {
                "name": "jacket length",
                "cardinality": "single",
                "values": [
                    "cropped",
                    "standard",
                    "long"
                ]
            },
            "outer_closure": {
                "name": "jacket closure",
                "cardinality": "multi",
                "values": [
                    "single_breasted",
                    "double_breasted",
                    "zipper",
                    "belt",
                    "none"
                ]
            },
            "outer_pocket": {
                "name": "jacket pocket location",
                "cardinality": "multi",
                "values": [
                    "none",
                    "chest_pockets",
                    "waist_pockets",
                    "lower_pockets"
                ]
            }
        }
    },
    "tops": {
        "domain": "top",
        "groups": {
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
                    "collared"
                ]
            },
            "top_sleeve_length": {
                "name": "top sleeve length",
                "cardinality": "single",
                "values": [
                    "long_sleeve",
                    "short_sleeve",
                    "sleeveless",
                    "three_quarter_sleeve"
                ]
            },
            "top_fit": {
                "name": "top fit and silhouette",
                "cardinality": "single",
                "values": [
                    "slim_fit",
                    "regular_fit",
                    "oversized"
                ]
            },
            "top_length": {
                "name": "top length",
                "cardinality": "single",
                "values": [
                    "cropped",
                    "standard",
                    "long"
                ]
            },
            "top_closure": {
                "name": "top closure",
                "cardinality": "single",
                "values": [
                    "buttons",
                    "zipper",
                    "none"
                ]
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
                    "pockets",
                    "hood",
                    "sheer"
                ]
            }
        }
    },
    "dresses": {
        "domain": "dress",
        "groups": {
            "dress_length": {
                "name": "dress length",
                "cardinality": "single",
                "values": [
                    "mini",
                    "knee_length",
                    "midi",
                    "maxi"
                ]
            },
            "dress_silhouette": {
                "name": "dress silhouette",
                "cardinality": "single",
                "values": [
                    "bodycon",
                    "sheath",
                    "shift",
                    "fit_and_flare",
                    "a_line",
                    "wrap",
                    "empire"
                ]
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
                    "strapless"
                ]
            },
            "dress_sleeve_length": {
                "name": "dress sleeve length",
                "cardinality": "single",
                "values": [
                    "sleeveless",
                    "short_sleeve",
                    "three_quarter_sleeve",
                    "long_sleeve"
                ]
            },
            "dress_waist": {
                "name": "dress waist",
                "cardinality": "single",
                "values": [
                    "defined_waist",
                    "dropped_waist",
                    "empire_waist",
                    "straight_waist"
                ]
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
                    "none"
                ]
            }
        }
    }
}


def category_attribute_id(group_id: str, name: str) -> str:
    return f"{group_id}:{name}"
