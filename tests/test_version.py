import spatial_ops


class TestVersion(object):

    def test_version(self):
        v = spatial_ops.__version__
        assert v == '0.1.0'
