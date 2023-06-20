from atriumdb.windowing.window import CommonWindowFormat
from atriumdb.windowing.window_config import WindowConfig


def test_cwf():
    start_time = 1234567890 * (10 ** 9)
    device_one = "74"
    measure_one = "MDC_RESP"
    windows_size_sec = 0.5
    window_slide_sec = 0.02
    wc = WindowConfig(measures=[measure_one], window_size_sec=windows_size_sec, window_slide_sec=window_slide_sec,
                      allowed_lateness_sec=0)

    example_cwf = CommonWindowFormat(start_time=start_time, device_id=device_one, window_config=wc)
