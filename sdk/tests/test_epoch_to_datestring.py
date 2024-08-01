from atriumdb.transfer.adb.datestring_conversion import nanoseconds_to_date_string_with_tz


def test_nanoseconds_to_date_string_with_tz():
    # Test case 1: Convert a known timestamp to a specific timezone
    nanoseconds = 1638316800000000000  # Corresponds to 2021-12-01T00:00:00 UTC
    timezone_str = 'America/New_York'
    expected = '2021-11-30T19:00:00.000000-0500'  # Eastern Standard Time (EST)
    assert nanoseconds_to_date_string_with_tz(nanoseconds, timezone_str) == expected

    # Test case 2: Default timezone (Etc/GMT)
    nanoseconds = 1638316800000000000  # Corresponds to 2021-12-01T00:00:00 UTC
    expected = '2021-12-01T00:00:00.000000+0000'  # GMT
    assert nanoseconds_to_date_string_with_tz(nanoseconds) == expected

    # Test case 3: Convert to a timezone with a different offset
    nanoseconds = 1638316800000000000  # Corresponds to 2021-12-01T00:00:00 UTC
    timezone_str = 'Asia/Tokyo'
    expected = '2021-12-01T09:00:00.000000+0900'  # Japan Standard Time (JST)
    assert nanoseconds_to_date_string_with_tz(nanoseconds, timezone_str) == expected

    # Test case 4: Convert a timestamp with fractional seconds
    nanoseconds = 1638316800123456789  # Corresponds to 2021-12-01T00:00:00.123456789 UTC
    timezone_str = 'Europe/London'
    expected = '2021-12-01T00:00:00.123457+0000'  # GMT (same as UTC in this case)
    assert nanoseconds_to_date_string_with_tz(nanoseconds, timezone_str) == expected

    # Test case 5: Handle Unix epoch start
    nanoseconds = 0  # Corresponds to 1970-01-01T00:00:00 UTC
    timezone_str = 'UTC'
    expected = '1970-01-01T00:00:00.000000+0000'  # UTC
    assert nanoseconds_to_date_string_with_tz(nanoseconds, timezone_str) == expected

    # Test case 6: Convert a timestamp before daylight saving time starts
    nanoseconds = 1647148400000000000  # Corresponds to 2022-03-13T05:13:20 UTC (before DST start)
    timezone_str = 'America/New_York'
    expected = '2022-03-13T00:13:20.000000-0500'  # Eastern Standard Time (EST)
    assert nanoseconds_to_date_string_with_tz(nanoseconds, timezone_str) == expected

    # Test case 7: Convert a timestamp after daylight saving time starts
    nanoseconds = 1647168400000000000  # Corresponds to Sunday, March 13, 2022 10:46:40 UTC (after DST start)
    timezone_str = 'America/New_York'
    expected = '2022-03-13T06:46:40.000000-0400'  # Eastern Daylight Time (EDT)
    assert nanoseconds_to_date_string_with_tz(nanoseconds, timezone_str) == expected

    # Test case 8: Southern Hemisphere
    nanoseconds = 1647148400000000000  # Corresponds to 2022-03-13T05:13:20 UTC
    timezone_str = 'Australia/Sydney'
    expected = '2022-03-13T16:13:20.000000+1100'  # Eastern Daylight Time (EDT)
    assert nanoseconds_to_date_string_with_tz(nanoseconds, timezone_str) == expected

    # Test case 9: Southern Hemisphere on the other side of daylight savings
    nanoseconds = 1649148400000000000  # Corresponds to Tuesday, April 5, 2022 8:46:40 UTC
    timezone_str = 'Australia/Sydney'
    expected = '2022-04-05T18:46:40.000000+1000'  # Eastern Standard Time (EST)
    assert nanoseconds_to_date_string_with_tz(nanoseconds, timezone_str) == expected
