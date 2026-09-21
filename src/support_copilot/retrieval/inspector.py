"""Print ranked evidence for command-line retrieval debugging."""


def print_results(results) -> None:
    """Display rank, score, metadata, and text for every result."""

    for rank, item in enumerate(results, start=1):
        print(f"RESULT {rank} score={item.score}")
        print(item.node.metadata)
        print(item.node.get_content()[:800])
