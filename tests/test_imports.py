def test_import_mktools():
    import mktools
    assert mktools is not None


def test_import_subpackages():
    import mktools.kstat
    import mktools.kfile
    import mktools.kio
    assert mktools.kstat is not None
    assert mktools.kfile is not None
    assert mktools.kio is not None
