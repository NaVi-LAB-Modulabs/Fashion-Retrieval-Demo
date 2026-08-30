"""Static taxonomy and Step1 common-attribute definitions for Fashion-How.

Common attributes (colors, materials, styles, occasions) apply to every garment
category. Category-specific attributes are defined separately in the Step2
package (jungyj/fashionhow)."""

from __future__ import annotations

from typing import Any

COLORS = [
    "검은색",
    "화이트",
    "아이보리",
    "베이지색",
    "회색",
    "갈색",
    "네이비",
    "파란색",
    "하늘색",
    "초록색",
    "카키색",
    "노란색",
    "주황색",
    "빨간색",
    "분홍색",
    "보라색",
    "실버",
    "골드",
]

MATERIALS = [
    "면",
    "린넨",
    "데님",
    "쉬폰",
    "실크",
    "울",
    "레이스",
    "가죽",
    "스웨이드",
    "트위드",
    "코듀로이",
    "폴리에스테르",
    "나일론",
]

STYLES = [
    "캐주얼한",
    "포멀한",
    "로맨틱한, 페미닌한",
    "모던한, 시크한",
    "클래식한, 미니멀한, 깔끔한",
    "빈티지한",
    "스포티한",
    "보헤미안",
    "고급스러운, 우아한",
    "화려한",
    "귀여운, 러블리한",
    "힙한, 스트릿한",
]

OCCASIONS = [
    "데일리",
    "하객",
    "비즈니스",
    "데이트",
    "파티",
    "여행, 휴양지",
    "캠퍼스",
    "운동",
]

PATTERNS = [
    "아가일",
    "체크",
    "스트라이프",
    "도트",
    "플라워",
    "없음",
]

SEASONS = [
    "봄",
    "여름",
    "가을",
    "겨울",
    "알 수 없음",
]

TYPE_NAMES = {
    "CT": "코트",
    "CD": "가디건",
    "VT": "조끼",
    "JK": "자켓",
    "JP": "점퍼",
    "KN": "니트",
    "SW": "스웨터",
    "SH": "셔츠",
    "BL": "블라우스",
    "SK": "스커트",
    "PT": "팬츠",
    "OP": "원피스",
    "SE": "신발",
}

SLOT_NAMES = {
    "O": "겉옷",
    "T": "상의",
    "B": "하의",
    "S": "신발",
}

META_ATTRIBUTE_NAMES = {
    "F": "features",
    "M": "materials",
    "C": "colors",
    "E": "sensibilities",
}

COMMON_ATTRIBUTE_VALUES = {
    "colors": COLORS,
    "materials": MATERIALS,
    "styles": STYLES,
    "occasions": OCCASIONS,
    "patterns": PATTERNS,
    "seasons": SEASONS,
}

COMMON_ATTRIBUTE_METRIC_FIELDS = {
    "colors": ["confidence", "coverage"],
    "patterns": ["confidence", "prominence"],
    "materials": ["confidence"],
    "styles": ["confidence", "score"],
    "occasions": ["confidence", "score"],
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
