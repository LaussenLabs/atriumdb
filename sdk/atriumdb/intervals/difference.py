def list_difference(first, second):
    result = []
    i, j = 0, 0

    while i < len(first) and j < len(second):
        # Check for non-overlapping intervals and add to result
        if first[i][1] <= second[j][0]:
            result.append(first[i])
            i += 1
            continue
        if second[j][1] <= first[i][0]:
            j += 1
            continue

        # Find overlapping intervals and update first list
        if first[i][0] < second[j][0]:
            result.append([first[i][0], second[j][0]])
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            first[i][0] = second[j][1]
            j += 1

    while i < len(first):
        result.append(first[i])
        i += 1

    return result
