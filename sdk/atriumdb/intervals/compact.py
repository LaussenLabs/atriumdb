def reverse_compact_list(lst):
    for index in range(len(lst) - 2, -1, -1):
        # if lst[index][1] + 1 >= lst[index + 1][0]:  # Original Code
        if lst[index][1] >= lst[index + 1][0]:
            lst[index][1] = lst[index + 1][1]
            del lst[index + 1]  # remove compacted entry O(n)*
    return lst
