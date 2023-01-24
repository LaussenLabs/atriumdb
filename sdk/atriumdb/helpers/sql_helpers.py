def gen_complete_dict(key_list, partial_dict):
    for key in key_list:
        if key not in partial_dict:
            partial_dict[key] = None

    return partial_dict
