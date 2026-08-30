"""Category-specific (Step2) attribute taxonomy for Fashion-How.

Common attributes (see taxonomy.py) apply to every garment. The category
attributes defined here are specialized per item ``type_code`` and are extracted
in a second VLM pass after the common attributes.

To support a new category, add an entry to ``CATEGORY_ATTRIBUTES`` keyed by its
Fashion-How ``type_code``; the extractor and graph builders pick it up
automatically. Each entry has:

- ``domain``: short, stable label stored on every node/edge for filtering.
- ``exclude_note`` (optional): one extra prompt line guarding against attributes
  that belong to other garments.
- ``prompt_guidance`` (optional): category-specific visual decision rules
  appended to the system prompt.
- ``groups``: ``group_id -> {name, cardinality ("single"|"multi"), values}``.
  Groups may also define ``value_descriptions`` for prompt guidance and
  ``exclusive_values`` for values that cannot coexist with other values.
  Use ``required`` to override the response fields required for that group.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .prompts import CATEGORY_ATTRIBUTE_SYSTEM_PROMPT_TEMPLATE
from .taxonomy import TYPE_NAMES

CATEGORY_ATTRIBUTE_DEFAULT_REQUIRED_FIELDS = ["name", "confidence", "prominence"]

CATEGORY_ATTRIBUTES: dict[str, dict[str, Any]] = {
    "PT": {
        "domain": "pants",
        "exclude_note": (
            "Ignore attributes that belong to other garments, such as neckline, "
            "sleeve, collar, or shoulder."
        ),
        "groups": {
            "pants_length": {
                "name": "팬츠 기장",
                "cardinality": "single",
                "values": ["3부", "5부", "7부", "9부", "10부"],
                "value_descriptions": {
                    "3부": "허벅지 위쪽에서 중간 정도까지 오는 매우 짧은 쇼츠 기장.",
                    "5부": "무릎 위나 무릎선 부근까지 오는 하프 팬츠 기장.",
                    "7부": "무릎 아래에서 종아리 중간 정도까지 오는 크롭 기장.",
                    "9부": "발목이 보이는 앵클/크롭 기장으로, 발등을 덮지 않음.",
                    "10부": "발목을 덮거나 발등/신발에 닿는 긴 기장.",
                },
                "required": ["name", "confidence"],
            },
            "pants_fit": {
                "name": "팬츠 핏/실루엣",
                "cardinality": "single",
                "values": [
                    "스키니핏",
                    "스트레이트핏",
                    "와이드핏",
                    "부츠컷",
                    "배기핏",
                    "조거핏",
                ],
                "required": ["name", "confidence"],
            },
            "pants_rise": {
                "name": "팬츠 밑위",
                "cardinality": "single",
                "values": ["로우웨이스트", "미드웨이스트", "하이웨이스트"],
                "required": ["name", "confidence"],
            },
            "pants_waist": {
                "name": "팬츠 허리 처리",
                "cardinality": "multi",
                "values": ["허리밴딩", "허리끈", "없음"],
                "exclusive_values": ["없음"],
                "required": ["name", "confidence"],
            },
            "pants_design": {
                "name": "팬츠 주요 디자인",
                "cardinality": "multi",
                "values": [
                    "핀턱",
                    "플리츠",
                    "롤업",
                    "카고",
                    "워싱",
                    "데미지",
                    "오버롤",
                    "점프수트",
                    "없음",
                ],
                "value_descriptions": {
                    "핀턱": "허리나 앞판에 접어 박은 좁은 선 주름 디테일.",
                    "플리츠": "접힌 주름이 반복되어 실루엣이나 장식으로 보이는 디테일.",
                    "롤업": "밑단을 접어 올린 디자인.",
                    "카고": "큰 패치 포켓이나 플랩 포켓이 강조된 카고 디자인.",
                    "워싱": "데님 등 원단에 탈색, 물빠짐, 색 바랜 효과가 있는 디자인.",
                    "데미지": "찢김, 구멍, 올풀림 등 의도적인 손상 디테일.",
                    "오버롤": "어깨끈이나 앞가슴받이가 있는 멜빵형 팬츠.",
                    "점프수트": "상의와 팬츠가 하나로 이어진 원피스형 팬츠.",
                    "없음": "위 디자인 요소가 뚜렷하지 않은 기본형이며, 선택 시 단독 사용.",
                },
                "exclusive_values": ["없음"],
            },
            "pants_pocket": {
                "name": "팬츠 포켓 유무",
                "cardinality": "single",
                "values": ["있음", "없음"],
            },
        },
    },
    "SK": {
        "domain": "skirt",
        "exclude_note": (
            "Ignore attributes that belong to other garments, such as neckline, "
            "sleeve, collar, shoulder, or pants leg shape."
        ),
        "groups": {
            "skirt_length": {
                "name": "스커트 기장",
                "cardinality": "single",
                "values": ["3부", "5부", "7부", "9부"],
                "value_descriptions": {
                    "3부": "허벅지 위쪽에서 중간 정도까지 오는 매우 짧은 미니 스커트 기장.",
                    "5부": "허벅지 아래쪽에서 무릎선 부근까지 오는 일반적인 무릎 위 또는 무릎 길이.",
                    "7부": "무릎 아래에서 종아리 중간 정도까지 오는 미디 스커트 기장.",
                    "9부": "종아리 아래에서 발목 부근까지 내려오는 긴 스커트 기장.",
                },
            },
            "skirt_silhouette": {
                "name": "스커트 실루엣",
                "cardinality": "single",
                "values": ["A라인", "H라인", "머메이드"],
                "value_descriptions": {
                    "A라인": "허리에서 밑단으로 갈수록 자연스럽게 넓어져 A자 형태로 퍼지는 실루엣.",
                    "H라인": "허리부터 밑단까지 폭 변화가 적고 일자로 떨어지는 직선형 실루엣.",
                    "머메이드": "허리와 힙 또는 허벅지 부근은 몸에 맞고, 무릎 아래나 밑단에서 퍼지는 실루엣.",
                },
            },
            "skirt_waist_fit": {
                "name": "스커트 허리 여밈",
                "cardinality": "multi",
                "values": ["버튼", "지퍼", "밴딩", "없음",],
                "value_descriptions": {
                    "버튼": "허리나 앞중심, 옆선 등에 단추가 여밈 장치로 보이는 경우.",
                    "지퍼": "허리나 옆선, 뒤중심 등에 지퍼 레일이나 지퍼선이 여밈으로 보이는 경우.",
                    "밴딩": "허리 부분이 고무줄처럼 주름지거나 늘어나는 밴드 구조로 보이는 경우.",
                    "없음": "버튼, 지퍼, 밴딩 등 허리 여밈 장치가 겉으로 뚜렷하지 않은 경우.",
                },
            },
            "skirt_detail": {
                "name": "스커트 디테일",
                "cardinality": "multi",
                "values": [
                    "플리츠",
                    "프릴",
                    "슬릿",
                    "포켓",
                    "레이스",
                    "프린지",
                    "랩",
                    "언밸런스",
                    "벨트",
                    "시스루",
                ],
                "value_descriptions": {
                    "플리츠": "접힌 주름이 반복되어 스커트 폭이나 장식으로 뚜렷하게 보이는 디테일.",
                    "프릴": "밑단, 허리, 절개선 등에 물결 모양으로 겹치거나 주름진 장식.",
                    "슬릿": "밑단이나 옆선, 앞뒤 중심이 트여 다리나 안쪽 공간이 보이는 절개 디테일.",
                    "포켓": "스커트 앞판, 옆선, 뒷판 등에 실제 주머니나 패치 포켓이 보이는 경우.",
                    "레이스": "구멍이 있는 망상 또는 꽃무늬 조직의 장식 원단이 사용된 디테일.",
                    "프린지": "밑단이나 절개선에 실, 끈, 술 장식이 늘어져 있는 디테일.",
                    "랩": "앞판 한쪽이 다른 쪽 위로 겹쳐 감싸는 랩 스커트 형태.",
                    "언밸런스": "좌우 또는 앞뒤 밑단 길이가 다르거나 비대칭 절개가 뚜렷한 디자인.",
                    "벨트": "허리 부분에 묶거나 조이는 벨트, 끈, 버클 장식이 보이는 경우.",
                    "시스루": "원단을 통해 피부, 안감, 배경이 비쳐 보이는 반투명 디테일.",
                },
            },
        },
    },
    "OT": {
        "domain": "outer",
        "groups": {
            "outer_collar": {
                "name": "아우터 칼라",
                "cardinality": "single",
                "values": ["칼라없음", "칼라있음"],
            },
            "outer_sleeve": {
                "name": "아우터 소매",
                "cardinality": "single",
                "values": ["긴소매", "반소매"],
                "required": ["name", "confidence"],
            },
            "outer_fit": {
                "name": "아우터 핏·실루엣",
                "cardinality": "single",
                "values": ["슬림핏", "레귤러핏", "오버핏"],
                "required": ["name", "confidence"],
            },
            "outer_length": {
                "name": "아우터 기장",
                "cardinality": "single",
                "values": ["숏기장", "스탠다드 기장", "롱기장"],
                "required": ["name", "confidence"],
            },
            "outer_closure": {
                "name": "아우터 여밈",
                "cardinality": "multi",
                "values": [
                    "싱글버튼여밈",
                    "더블버튼여밈",
                    "지퍼여밈",
                    "벨트여밈",
                    "여밈없음",
                ],
                "value_descriptions": {
                    "싱글버튼여밈": "앞중심에 버튼이 한 줄로 배열된 여밈. 한쪽 앞판이 다른 쪽 위로 겹쳐지고 버튼 줄은 하나만 보임.",
                    "더블버튼여밈": "앞중심에 버튼이 두 줄로 배열된 여밈. 앞판 겹침이 넓고 좌우 또는 평행한 두 줄의 버튼이 뚜렷함.",
                    "지퍼여밈": "앞중심에 지퍼 레일, 지퍼 손잡이, 또는 지퍼선이 주요 여밈으로 보이는 경우.",
                    "벨트여밈": "허리나 몸판을 묶거나 조이는 벨트가 주요 여밈으로 보이는 경우. 트렌치코트의 허리 벨트 포함.",
                    "여밈없음": "버튼, 지퍼, 벨트 등 닫는 장치가 겉으로 뚜렷하지 않거나 오픈형으로 보이는 경우.",
                },
                "exclusive_values": ["여밈없음"],
                "required": ["name", "confidence"],
            },
            "outer_pocket": {
                "name": "아우터 주머니 위치",
                "cardinality": "multi",
                "values": ["주머니없음", "가슴포켓", "허리포켓", "하단포켓"],
                "value_descriptions": {
                    "주머니없음": "겉으로 보이는 주머니가 없는 경우. 선택 시 다른 포켓 위치와 함께 선택하지 않음.",
                    "가슴포켓": "가슴 높이의 앞판 위쪽에 있는 포켓. 셔츠형 자켓, 데님 자켓, 사파리 자켓의 가슴 부분 포켓.",
                    "허리포켓": "허리나 골반 높이에 있어 손을 넣는 위치의 포켓. 일반 자켓, 점퍼, 코트의 좌우 손주머니.",
                    "하단포켓": "긴 코트나 롱 아우터의 앞판 아래쪽에 있는 포켓. 허리보다 낮은 위치의 큰 포켓.",
                },
                "exclusive_values": ["주머니없음"],
            },
        },
    },
    "TO": {
        "domain": "top",
        "exclude_note": (
            "Judge only the target top. Ignore outerwear, bottoms, accessories, "
            "and any inner or layered garment that is not the target item."
        ),
        "prompt_guidance": """TO-specific visual decision rules:
- Treat neckline, sleeve length, fit, length, and closure as independent groups.
- For every single group, return exactly one best-supported value. When the image is ambiguous, choose the closest visible value and lower its confidence instead of inventing a value.
- For the detail group, return every clearly visible applicable detail and return an empty array when none is visible. Do not add a detail merely because it is common for the garment type.
- Determine neckline from the shape of the neck opening. Use 카라넥 only when a distinct folded or standing collar is visibly attached to the neckline.
- Determine sleeve length separately from sleeve shape. 퍼프소매 and 벌룬소매 are details and may coexist with 긴팔, 반팔, or 7부소매.
- Determine fit from the garment silhouette, ease, body width, and shoulder position. Do not infer the wearer's body shape or use pose as evidence.
- Determine top length from the hem position relative to the natural waist and hips, not from image cropping.
- Select 버튼여밈 or 지퍼여밈 only when a functional opening is visibly supported. Decorative hardware alone is not a closure.
- Select 시스루 only when skin, an underlayer, or the background is visibly discernible through a meaningful area of the fabric; sheen or light color alone is insufficient.""",
        "groups": {
            "top_neckline": {
                "name": "상의 넥라인",
                "cardinality": "single",
                "values": [
                    "브이넥",
                    "라운드넥",
                    "스퀘어넥",
                    "보트넥",
                    "홀터넥",
                    "터틀넥",
                    "카라넥",
                ],
                "value_descriptions": {
                    "브이넥": "목 중심 아래로 양쪽 선이 대각선으로 내려가 V자 모양의 파임을 이루는 넥라인.",
                    "라운드넥": "목둘레가 원형이나 완만한 U자 곡선으로 이어지는 기본적인 둥근 넥라인.",
                    "스퀘어넥": "앞 파임의 아래선이 비교적 수평이고 양옆이 각지게 올라가 사각형 윤곽을 이루는 넥라인.",
                    "보트넥": "쇄골을 따라 좌우 어깨 방향으로 넓고 얕게 벌어진 가로형 넥라인.",
                    "홀터넥": "목 뒤나 목둘레로 이어지는 스트랩 또는 원단이 상의를 지지하며 어깨가 크게 드러나는 넥라인.",
                    "터틀넥": "목을 감싸도록 원단이 높게 올라오며 접히거나 세워지는 하이넥 형태.",
                    "카라넥": "셔츠 칼라, 폴로 칼라처럼 목둘레에 접히거나 세워진 별도의 칼라 구조가 뚜렷한 넥라인.",
                },
            },
            "top_sleeve_length": {
                "name": "상의 소매 길이",
                "cardinality": "single",
                "values": ["긴팔", "반팔", "민소매", "7부소매"],
                "value_descriptions": {
                    "긴팔": "소매가 팔꿈치를 지나 손목 부근까지 내려오는 길이.",
                    "반팔": "소매가 어깨에서 시작해 주로 위팔 중간에서 팔꿈치 위 사이에 끝나는 길이.",
                    "민소매": "팔을 덮는 소매통이 없고 몸판이 암홀에서 끝나는 형태. 조끼형 상의도 포함.",
                    "7부소매": "소매가 팔꿈치 아래에서 손목 위, 대체로 아래팔 중간 부근에 끝나는 길이.",
                },
            },
            "top_fit": {
                "name": "상의 핏·실루엣",
                "cardinality": "single",
                "values": ["슬림핏", "레귤러핏", "오버핏"],
                "value_descriptions": {
                    "슬림핏": "몸판과 소매가 신체선에 가깝게 붙고 여유분이 적어 전체 폭이 좁게 보이는 핏.",
                    "레귤러핏": "신체선에 과하게 붙거나 넓지 않고 어깨와 몸판에 일반적인 여유가 있는 기본 핏.",
                    "오버핏": "몸판 폭과 소매가 넉넉하고 드롭 숄더나 큰 품처럼 의도적으로 크게 보이는 핏.",
                },
            },
            "top_length": {
                "name": "상의 기장",
                "cardinality": "single",
                "values": ["숏", "스탠다드", "롱"],
                "value_descriptions": {
                    "숏": "밑단이 자연 허리선 부근이나 그 위에 위치해 허리 또는 복부가 드러날 수 있는 크롭 기장.",
                    "스탠다드": "밑단이 허리 아래에서 골반 위쪽 또는 엉덩이 상단 부근까지 오는 일반적인 상의 기장.",
                    "롱": "밑단이 골반과 엉덩이를 상당 부분 덮거나 그 아래로 내려오는 튜닉형의 긴 기장.",
                },
            },
            "top_closure": {
                "name": "상의 여밈",
                "cardinality": "single",
                "values": ["버튼여밈", "지퍼여밈", "여밈없음"],
                "value_descriptions": {
                    "버튼여밈": "앞중심이나 주요 개방부를 실제로 열고 닫는 버튼과 버튼 줄이 뚜렷한 여밈.",
                    "지퍼여밈": "앞중심이나 주요 개방부에 지퍼 레일과 슬라이더가 보여 지퍼로 열고 닫는 여밈.",
                    "여밈없음": "기능적인 버튼이나 지퍼 개방부가 보이지 않는 풀오버 또는 오픈 장치가 없는 형태.",
                },
            },
            "top_detail": {
                "name": "상의 장식·디테일",
                "cardinality": "multi",
                "values": [
                    "프릴장식",
                    "리본",
                    "셔링",
                    "레이스",
                    "타이",
                    "자수",
                    "퍼프소매",
                    "벌룬소매",
                    "포켓",
                    "후드",
                    "시스루",
                ],
                "value_descriptions": {
                    "프릴장식": "원단 가장자리나 절개선을 따라 물결치거나 층을 이루는 주름 장식.",
                    "리본": "끈이나 원단을 나비 모양으로 묶어 형태가 고정되거나 장식으로 드러난 디테일.",
                    "셔링": "봉제선이나 고무줄로 원단을 모아 잔주름과 볼륨이 반복적으로 생긴 디테일.",
                    "레이스": "실이나 원단에 구멍이 있는 섬세한 망상·꽃무늬 조직이 장식적으로 사용된 디테일.",
                    "타이": "목이나 앞중심에서 길게 늘어뜨리거나 묶도록 설계된 보타이·스카프형 끈 디테일.",
                    "자수": "원단 표면에 실로 무늬, 글자, 로고 등을 수놓아 질감과 윤곽이 보이는 장식.",
                    "퍼프소매": "어깨나 소매산에 주름을 모아 소매 윗부분이 둥글게 부풀어 오른 소매 형태.",
                    "벌룬소매": "소매 몸통 전체가 풍성하게 부풀고 소매 끝단에서 다시 좁아지는 풍선형 소매.",
                    "포켓": "가슴이나 몸판에 입구, 패치, 플랩 등으로 식별되는 실제 주머니가 보이는 디테일.",
                    "후드": "목 뒤에 연결되어 머리를 덮을 수 있는 모자 형태의 원단 구조.",
                    "시스루": "원단의 의미 있는 영역을 통해 피부, 이너웨어, 또는 배경이 비쳐 보이는 반투명 디테일.",
                },
            },
        },
    },
}


def category_type_codes() -> list[str]:
    return list(CATEGORY_ATTRIBUTES)


def has_category(type_code: str) -> bool:
    return type_code in CATEGORY_ATTRIBUTES


def category_domain(type_code: str) -> str:
    return CATEGORY_ATTRIBUTES[type_code]["domain"]


def category_groups(type_code: str) -> dict[str, dict[str, Any]]:
    return CATEGORY_ATTRIBUTES[type_code]["groups"]


def category_schema_name(type_code: str, suffix: str | None = None) -> str:
    name = f"category_attributes_{category_domain(type_code)}"
    return f"{name}_{suffix}" if suffix else name


def category_attribute_id(group_id: str, name: str) -> str:
    return f"{group_id}:{name}"


def category_group_nodes(type_code: str) -> list[dict[str, str]]:
    domain = category_domain(type_code)
    return [
        {
            "id": group_id,
            "name": group["name"],
            "domain": domain,
            "type_code": type_code,
            "cardinality": group["cardinality"],
        }
        for group_id, group in category_groups(type_code).items()
    ]


def category_value_nodes(type_code: str) -> list[dict[str, str]]:
    domain = category_domain(type_code)
    nodes: list[dict[str, str]] = []
    for group_id, group in category_groups(type_code).items():
        for value in group["values"]:
            nodes.append(
                {
                    "id": category_attribute_id(group_id, value),
                    "name": value,
                    "group": group_id,
                    "domain": domain,
                    "type_code": type_code,
                }
            )
    return nodes


def category_group_edges(type_code: str) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    for group_id, group in category_groups(type_code).items():
        for value in group["values"]:
            edges.append(
                {"from": category_attribute_id(group_id, value), "to": group_id}
            )
    return edges


def category_attribute_required_fields(group: dict[str, Any]) -> list[str]:
    if "required" in group and group["required"] is not None:
        required = list(group["required"])
    else:
        required = list(CATEGORY_ATTRIBUTE_DEFAULT_REQUIRED_FIELDS)
    if "name" not in required:
        required.insert(0, "name")
    return list(dict.fromkeys(required))


def _category_attribute_item_properties(
    group: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for field in required:
        if field == "name":
            properties[field] = {"type": "string", "enum": list(group["values"])}
        else:
            properties[field] = {"type": "number"}
    return properties


def _selected_category_groups(
    type_code: str,
    group_ids: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    groups = category_groups(type_code)
    if group_ids is None:
        return groups
    return {group_id: groups[group_id] for group_id in group_ids}


def category_attribute_schema(
    type_code: str,
    group_ids: list[str] | None = None,
) -> dict[str, Any]:
    groups = _selected_category_groups(type_code, group_ids)
    properties = {}
    for group_id, group in groups.items():
        required = category_attribute_required_fields(group)
        group_schema: dict[str, Any] = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": required,
                "properties": _category_attribute_item_properties(group, required),
            },
        }
        if group["cardinality"] == "single":
            group_schema["minItems"] = 1
            group_schema["maxItems"] = 1
        else:
            group_schema["maxItems"] = len(group["values"])
        properties[group_id] = group_schema
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(groups),
        "properties": properties,
    }


def category_groups_json(
    type_code: str,
    group_ids: list[str] | None = None,
) -> str:
    """The category's attribute groups (name, cardinality, values) as readable JSON.

    Embedded in the category user prompt so the model sees the group names and
    cardinality, which the response JSON schema (enum-only) does not convey.
    """
    return json.dumps(
        _selected_category_groups(type_code, group_ids),
        ensure_ascii=False,
        indent=2,
    )


def category_attribute_system_prompt(type_code: str) -> str:
    note = CATEGORY_ATTRIBUTES[type_code].get("exclude_note")
    guidance = CATEGORY_ATTRIBUTES[type_code].get("prompt_guidance")
    prompt_notes = [text for text in (note, guidance) if text]
    exclude_line = "".join(f"\n\n{text}" for text in prompt_notes)
    return CATEGORY_ATTRIBUTE_SYSTEM_PROMPT_TEMPLATE.format(
        category_name=TYPE_NAMES.get(type_code, type_code),
        exclude_line=exclude_line,
    )
