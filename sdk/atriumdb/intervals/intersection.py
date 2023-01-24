def list_intersection(first, second):
    result = []
    i, j = 0, 0

    while i < len(first) and j < len(second):
        inter_list = [max(first[i][0], second[j][0]), min(first[i][1], second[j][1])]
        if inter_list[0] < inter_list[1]:
            if result and result[-1][-1] >= inter_list[0]:
                result[-1][-1] = inter_list[1]
            else:
                result.append(inter_list)

        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1

    return result
