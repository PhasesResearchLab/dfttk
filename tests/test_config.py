def test_import():
    try:
        import config
    except ImportError:
        assert False, "Failed to import config module"