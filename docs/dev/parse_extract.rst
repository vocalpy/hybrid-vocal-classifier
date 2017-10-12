=============================
parse package, extract module
=============================

The extract module parses `extract.config.yml` files.

Here's a rough outline of how it works.

parse\extract.py contains the following functions:
- `validate_spect_params`
- `validate_segment_params`
- `_validate_feature_group_and_convert_to_list`
- `_validate_todo_list_dict`
- `validate_yaml`

`validate_yaml`
---------------

`validate_yaml` is the main function; it gets called by parse.


`_validate_todo_list_dict`
--------------------------

`_validate_todo_list_dict` is called by `validate_yaml` when it finds a key `todo_list` whose value is a list of dicts.



`_validate_feature_group_and_convert_to_list`
---------------------------------------------
This function validates feature groups, which are validated differently than feature lists.
Feature lists are validated by making sure every element in the list is a string found within a valid features list,
which is a concatenation of all the features listend in `feature_groups.yml` in the parse module.

The parsing of a `feature_group` key is a little more complicated. The first step is to make sure
the group or groups appear in the dictionary of valid feature groups in `hvc/parse/feature_groups.yml'.
The keys of the dictionary of valid feature groups are the valid feature group names,
and the values of the dictionary are the actual lists of features.
If each `str` in `feature_group1 is s a valid feature group, then the list of features is taken from the dictionary
 of valid feature groups. The list is then validated by comparing it to the list of all features in `features.yml`.
This is to make sure the developer didn't make a typo. If `feature_group` is a list of feature group names,
then `feature_list` will consist of all features from all groups in the list. A vector of the same length as
the new feature list has values that indicate which feature group each element in the new feature list belongs to.
This vector is named `feature_list_group_ID`. A dict named `ftr_group_dict` is also returned,
 where each key is a name of a feature group and its
corresponding value is the ID number given to that feature group. Using this dict and the identity array,
`hvc/parse/select` can pull the correct features out of a feature array given a feature group name.

Example:
```Python
>>> ftr_tuple = _validate_feature_group_and_convert_to_list(feature_group=['knn','svm'])

>>> ftr_tuple[0]

['duration group','preceding syllable duration' ... ]  # and so on

>>> ftr_tuple[1]

 np.ndarray([0,0,0,0,0,1,1,1,1,1])  # some array with one of two ID numbers

>>> len(ftr_typle[0]) = ftr_tuple[1].shape[-1]

True

>>> ftr_tuple[2]

{'knn': 0, 'svm': 1}
```

If a `feature_list` was passed to this function along with `feature_group`, the features from the feature groups are
appended to the `feature_list`, and in the `feature_group_ID` vector, the original features from the original feature
list have a value of `None`.