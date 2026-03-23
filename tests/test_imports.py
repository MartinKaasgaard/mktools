def test_import_mktools():
    import mktools
    assert mktools is not None

def test_import_kstat():
    from mktools.kstat import KStatProfiler, DateSeriesValidator
    assert KStatProfiler is not None
    assert DateSeriesValidator is not None
