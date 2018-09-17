# utility functions used by parsers for each module


def check_for_missing_keys(a_dict, a_list_of_keys):
    """checks whether required keys are missing from a dict.
    Items in a_list_of_keys can be str or tuple.
    In the case of a tuple, **one** of the strs in the tuple
    must be in the dict.

    Parameters
    ----------
    a_dict : dict
        dict to check for missing keys
    a_list_of_keys : list
        of keys, either str or a tuple of str

    Returns
    -------
    keys_not_found : str
        Keys from a_list_of_ that were not found in the dict.
        If no keys are missing, returns empty string ('')
        (that evaluates as False so no error is raised).

    Wrote a separate function to do this
    because all the module parsers need to check for some situation
    where *one of* a subset of keys is required (i.e., either/or)
    along with a bunch of other keys that are *all* required
    and this is hard/messy to specify with vanilla set logic.
    """
    keys_not_found = []
    for str_or_tuple_key in a_list_of_keys:
        if type(str_or_tuple_key) == str:
            if str_or_tuple_key not in a_dict:
                keys_not_found.append(str_or_tuple_key)
        elif type(str_or_tuple_key) == tuple:
            if not any([key_from_tuple in a_dict
                       for key_from_tuple in str_or_tuple_key]):
                keys_str_rep = ' or '.join(str_or_tuple_key)
                keys_not_found.append(keys_str_rep)
    # below, if keys_not_found is empty list, returns ''
    # which evaluates as False
    keys_not_found = ', '.join(keys_not_found)
    return keys_not_found


def flatten(a_list_of_keys):
    """flatten a list of keys
    where items in list can be str or tuple of str.
    Used when checking for keys that are not"""
    flattened = []
    for str_or_tuple_key in a_list_of_keys:
        if type(str_or_tuple_key) == str:
            flattened.append(str_or_tuple_key)
        elif type(str_or_tuple_key) == tuple:
            for key_from_tuple in str_or_tuple_key:
                flattened.append(key_from_tuple)
    return flattened
