def test_import():
    try:
        import dfttk.config
    except ImportError:
        assert False, "Failed to import dfttk.config module"