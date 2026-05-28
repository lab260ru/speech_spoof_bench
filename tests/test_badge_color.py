import pytest
from speech_spoof_bench import badge


@pytest.mark.parametrize("eer,expected", [
    (0.0, "brightgreen"),
    (1.99, "brightgreen"),
    (2.0, "green"),         # >= 2.0 → green
    (4.99, "green"),
    (5.0, "yellow"),        # >= 5.0 → yellow
    (9.99, "yellow"),
    (10.0, "lightgrey"),    # >= 10.0 → lightgrey
    (50.0, "lightgrey"),
])
def test_color_for_eer(eer, expected):
    assert badge._color_for_eer(eer) == expected
