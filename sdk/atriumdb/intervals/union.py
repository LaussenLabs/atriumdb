def intervals_union(a, b):
    # Combine all intervals
    intervals = a + b
    # Sort intervals by their start
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]
    for current in intervals:
        # if the list of merged intervals is empty or if the current interval does not overlap with the previous, append it
        if not merged or merged[-1][1] < current[0]:
            merged.append(current)
        # otherwise, there is overlap, so we merge the current and previous intervals.
        else:
            merged[-1][1] = max(merged[-1][1], current[1])

    return merged
