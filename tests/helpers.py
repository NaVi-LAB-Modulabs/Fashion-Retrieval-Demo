"""Shared offline retrieval fixtures, independent of individual test cases."""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fashion_how_graphdb import web_service as service


class RetrievalHelper:
    def __init__(self):
        self.extraction = {
            "item_type_codes": ["tops"],
            "common_filters": [{"group": "colors", "value": "blue", "hard": True}],
            "category_filters": [], "excluded_common_filters": [],
            "excluded_category_filters": [], "description_query": "",
            "style_axis_targets": [],
        }

    def run_search(self, rows, **settings):
        driver = MagicMock()
        session = driver.session.return_value.__enter__.return_value
        def run_query(cypher, **params):
            if "AS description_eligible" in cypher:
                extraction = service.search.normalize_extraction(self.extraction)
                eligible = [row for row in rows if service.hybrid_search.passes_hard_filters(
                    row.get("mapped_attributes") or [], extraction, params["min_confidence"],
                )]
                return [{"id": row["id"], "graph_score": service.hybrid_search.graph_score(
                    row.get("mapped_attributes") or [], extraction, params["min_confidence"],
                )[0], "description_eligible": row.get("text_score") is not None} for row in eligible]
            if "ORDER BY text_score DESC, id ASC" in cypher:
                text = sorted([row for row in rows if row["id"] in params["filtered_ids"]],
                              key=lambda row: (-row["text_score"], row["id"]))[:params["branch_limit"]]
                return [{"id": row["id"], "text_score": row["text_score"]} for row in text]
            return [{"result": row} for row in rows if row["id"] in params["candidate_ids"]]
        session.run.side_effect = run_query
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])
        with ExitStack() as stack:
            stack.enter_context(patch.object(service, "configuration", return_value={"ready": True}))
            stack.enter_context(patch.object(service, "neo4j_driver", return_value=driver))
            stack.enter_context(patch.object(service, "openai_client", return_value=client))
            stack.enter_context(patch.object(
                service.search, "extract_search_filters",
                return_value=service.search.normalize_extraction(self.extraction),
            ))
            result = service.retrieve({"query": "a blue blouse", **settings})
        return result, session, client


@contextmanager
def retrieval_fixture():
    """Give each test its own extraction and restore environment settings on exit."""
    with patch.dict("os.environ", {"FASHION_CACHE_ENABLED": "0", "FASHION_VECTOR_SEARCH_MODE": "hnsw"}):
        yield RetrievalHelper()
