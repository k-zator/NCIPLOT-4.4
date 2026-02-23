import spatial.constants as constants


def test_colorid_array_shape_and_range():
    assert constants.COLORID.shape[1] == 4
    assert constants.COLORID.shape[0] >= 30
    assert (constants.COLORID[:, :3] >= 0.0).all()
    assert (constants.COLORID[:, :3] <= 1.0).all()
