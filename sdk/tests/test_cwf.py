from atriumdb.windowing.window import CommonWindowFormat


def test_cwf():
    start_time = 1234567890 * (10 ** 9)
    example_cwf = CommonWindowFormat()
