"""Generate Neo4j Cypher import scripts from Fashion-How GraphDB JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


DEFAULT_GRAPH_DIR = Path("jiyoon/output/fashion-how-graphdb-outer-all-260629")
DEFAULT_CYPHER_OUTPUT = DEFAULT_GRAPH_DIR / "neo4j_import.cypher"
DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI")
DEFAULT_NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
DEFAULT_NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
DEFAULT_BATCH_SIZE = 500
DEFAULT_CREATE_CONSTRAINTS = True
DEFAULT_LOAD_TO_NEO4J = True
DEFAULT_WRITE_CYPHER = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import Fashion-How GraphDB JSON into Neo4j or write a Cypher file.",
    )
    parser.add_argument(
        "--graph_dir",
        type=Path,
        default=DEFAULT_GRAPH_DIR,
        help="Directory containing nodes/ and edges/ JSON outputs.",
    )
    parser.add_argument(
        "--cypher_output",
        type=Path,
        default=None,
        help="Path for the generated .cypher file. Defaults to <graph_dir>/neo4j_import.cypher.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of rows to import per Neo4j batch.",
    )
    parser.add_argument(
        "--no_constraints",
        action="store_true",
        help="Skip creating Neo4j uniqueness constraints.",
    )
    parser.add_argument(
        "--write_cypher",
        action="store_true",
        default=DEFAULT_WRITE_CYPHER,
        help="Write a .cypher import file.",
    )
    parser.add_argument(
        "--no_load_to_neo4j",
        action="store_true",
        help="Do not load directly into Neo4j.",
    )
    return parser.parse_args()


NODE_LABELS = {
    "items": "Item",
    "item_types": "ItemType",
    "slots": "Slot",
    "colors": "Color",
    "materials": "Material",
    "styles": "Style",
    "occasions": "Occasion",
    "patterns": "Pattern",
    "seasons": "Season",
    "attribute_groups": "AttributeGroup",
    "attributes": "Attribute",
}

EDGE_SPECS = {
    "is_type": ("Item", "IS_TYPE", "ItemType"),
    "belongs_to_slot": ("ItemType", "BELONGS_TO_SLOT", "Slot"),
    "has_color": ("Item", "HAS_COLOR", "Color"),
    "has_material": ("Item", "HAS_MATERIAL", "Material"),
    "has_style": ("Item", "HAS_STYLE", "Style"),
    "has_occasion": ("Item", "HAS_OCCASION", "Occasion"),
    "has_pattern": ("Item", "HAS_PATTERN", "Pattern"),
    "has_season": ("Item", "HAS_SEASON", "Season"),
    "in_attribute_group": ("Attribute", "IN_ATTRIBUTE_GROUP", "AttributeGroup"),
    "has_attribute": ("Item", "HAS_ATTRIBUTE", "Attribute"),
}


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {path}")
    return [row for row in data if isinstance(row, dict)]


def _cypher_key(key: str) -> str:
    return f"`{key.replace('`', '``')}`"


def _cypher_literal(value: Any) -> str:
    if isinstance(value, dict):
        items = ", ".join(
            f"{_cypher_key(str(key))}: {_cypher_literal(item)}"
            for key, item in value.items()
            if item is not None
        )
        return "{" + items + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_cypher_literal(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if value is None:
        return "null"
    return _cypher_string(str(value))


def _cypher_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f"'{escaped}'"


def _chunked(
    values: list[dict[str, Any]], batch_size: int
) -> list[list[dict[str, Any]]]:
    return [
        values[index : index + batch_size]
        for index in range(0, len(values), batch_size)
    ]


def _node_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "props": row,
        }
        for row in rows
        if row.get("id") is not None
    ]


def _edge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edge_rows = []
    for row in rows:
        if row.get("from") is None or row.get("to") is None:
            continue
        edge_rows.append(
            {
                "from": row["from"],
                "to": row["to"],
                "props": {
                    key: value
                    for key, value in row.items()
                    if key not in {"from", "to"}
                },
            }
        )
    return edge_rows


def _constraint_cypher(label: str) -> str:
    name = f"{label.lower()}_id_unique"
    return (
        f"CREATE CONSTRAINT {_cypher_key(name)} IF NOT EXISTS "
        f"FOR (n:{_cypher_key(label)}) REQUIRE n.id IS UNIQUE;"
    )


def _node_import_cypher(
    label: str, rows: list[dict[str, Any]], batch_size: int
) -> list[str]:
    statements = []
    for batch in _chunked(_node_rows(rows), batch_size):
        if not batch:
            continue
        statements.append(
            "\n".join(
                [
                    f"// Nodes: {label} ({len(batch)})",
                    f"UNWIND {_cypher_literal(batch)} AS row",
                    f"MERGE (n:{_cypher_key(label)} {{id: row.id}})",
                    "SET n += row.props;",
                ]
            )
        )
    return statements


def _edge_import_cypher(
    from_label: str,
    rel_type: str,
    to_label: str,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> list[str]:
    statements = []
    for batch in _chunked(_edge_rows(rows), batch_size):
        if not batch:
            continue
        statements.append(
            "\n".join(
                [
                    f"// Relationships: {rel_type} ({len(batch)})",
                    f"UNWIND {_cypher_literal(batch)} AS row",
                    f"MATCH (from_node:{_cypher_key(from_label)} {{id: row.`from`}})",
                    f"MATCH (to_node:{_cypher_key(to_label)} {{id: row.`to`}})",
                    f"MERGE (from_node)-[r:{_cypher_key(rel_type)}]->(to_node)",
                    "SET r += row.props;",
                ]
            )
        )
    return statements


def _node_query(label: str) -> str:
    return "\n".join(
        [
            "UNWIND $rows AS row",
            f"MERGE (n:{_cypher_key(label)} {{id: row.id}})",
            "SET n += row.props",
        ]
    )


def _edge_query(from_label: str, rel_type: str, to_label: str) -> str:
    return "\n".join(
        [
            "UNWIND $rows AS row",
            f"MATCH (from_node:{_cypher_key(from_label)} {{id: row.from}})",
            f"MATCH (to_node:{_cypher_key(to_label)} {{id: row.to}})",
            f"MERGE (from_node)-[r:{_cypher_key(rel_type)}]->(to_node)",
            "SET r += row.props",
        ]
    )


def _run_batches(
    session: Any,
    query: str,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
) -> int:
    total = 0
    for batch in _chunked(rows, batch_size):
        if not batch:
            continue
        session.run(query, rows=batch)
        total += len(batch)
    return total


def generate_cypher(
    graph_dir: Path,
    *,
    batch_size: int = 500,
    include_constraints: bool = True,
) -> str:
    statements = [
        "// Generated by fashion_how_graphdb.cypher",
        f"// Source: {graph_dir}",
    ]

    if include_constraints:
        statements.extend(_constraint_cypher(label) for label in NODE_LABELS.values())

    for file_name, label in NODE_LABELS.items():
        rows = _load_json(graph_dir / "nodes" / f"{file_name}.json")
        statements.extend(_node_import_cypher(label, rows, batch_size))

    for file_name, (from_label, rel_type, to_label) in EDGE_SPECS.items():
        rows = _load_json(graph_dir / "edges" / f"{file_name}.json")
        statements.extend(
            _edge_import_cypher(from_label, rel_type, to_label, rows, batch_size)
        )

    return "\n\n".join(statements) + "\n"


def write_cypher_file(
    graph_dir: Path = DEFAULT_GRAPH_DIR,
    output: Path = DEFAULT_CYPHER_OUTPUT,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_constraints: bool = DEFAULT_CREATE_CONSTRAINTS,
) -> None:
    cypher = generate_cypher(
        graph_dir,
        batch_size=batch_size,
        include_constraints=include_constraints,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(cypher, encoding="utf-8")
    print(f"Wrote Cypher: {output}")


def load_graph_to_neo4j(
    graph_dir: Path = DEFAULT_GRAPH_DIR,
    *,
    uri: str | None = DEFAULT_NEO4J_URI,
    user: str = DEFAULT_NEO4J_USER,
    password: str | None = DEFAULT_NEO4J_PASSWORD,
    database: str | None = DEFAULT_NEO4J_DATABASE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_constraints: bool = DEFAULT_CREATE_CONSTRAINTS,
) -> None:
    if not uri:
        raise SystemExit("NEO4J_URI is not set. Set it in .env or DEFAULT_NEO4J_URI.")
    if not password:
        raise SystemExit(
            "NEO4J_PASSWORD is not set. Set it in .env or DEFAULT_NEO4J_PASSWORD."
        )

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install neo4j") from exc

    auth = (user, password)
    driver = GraphDatabase.driver(uri, auth=auth)

    try:
        with (
            driver.session(database=database) if database else driver.session()
        ) as session:
            if include_constraints:
                for label in NODE_LABELS.values():
                    session.run(_constraint_cypher(label))
                print(f"constraints: {len(NODE_LABELS)}")

            for file_name, label in NODE_LABELS.items():
                rows = _node_rows(_load_json(graph_dir / "nodes" / f"{file_name}.json"))
                count = _run_batches(
                    session,
                    _node_query(label),
                    rows,
                    batch_size=batch_size,
                )
                print(f"nodes/{file_name}.json -> :{label} ({count})")

            for file_name, (from_label, rel_type, to_label) in EDGE_SPECS.items():
                rows = _edge_rows(_load_json(graph_dir / "edges" / f"{file_name}.json"))
                count = _run_batches(
                    session,
                    _edge_query(from_label, rel_type, to_label),
                    rows,
                    batch_size=batch_size,
                )
                print(f"edges/{file_name}.json -> :{rel_type} ({count})")
    finally:
        driver.close()


def main() -> None:
    args = parse_args()
    graph_dir = args.graph_dir
    cypher_output = args.cypher_output or graph_dir / "neo4j_import.cypher"
    include_constraints = not args.no_constraints
    load_to_neo4j = DEFAULT_LOAD_TO_NEO4J and not args.no_load_to_neo4j

    if not (graph_dir / "nodes").exists() or not (graph_dir / "edges").exists():
        raise SystemExit(f"Graph directory must contain nodes/ and edges/: {graph_dir}")

    if args.write_cypher:
        write_cypher_file(
            graph_dir,
            cypher_output,
            batch_size=args.batch_size,
            include_constraints=include_constraints,
        )
    if load_to_neo4j:
        load_graph_to_neo4j(
            graph_dir,
            uri=DEFAULT_NEO4J_URI,
            user=DEFAULT_NEO4J_USER,
            password=DEFAULT_NEO4J_PASSWORD,
            database=DEFAULT_NEO4J_DATABASE,
            batch_size=args.batch_size,
            include_constraints=include_constraints,
        )


if __name__ == "__main__":
    main()
