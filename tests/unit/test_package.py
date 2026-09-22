def test_package_exposes_semantic_version() -> None:
    import mini_stack

    assert mini_stack.__version__ == "0.1.0"
