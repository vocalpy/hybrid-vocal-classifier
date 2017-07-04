=============================
parse package, extract module
=============================

The extract module parses `extract.config.yml` files.

Here's a rough outline of how it works.

parse\extract.py contains the following functions:
- `validate_spect_params`
- `validate_segment_params`
- `_validate_todo_list_dict`
- `validate_yaml`

`validate_yaml`
---------------

`validate_yaml` is the main function; it gets called by parse.


`_validate_todo_list_dict`
--------------------------

`_validate_todo_list_dict` is called by `validate_yaml` when it finds a key `todo_list` whose value is a list of dicts.

feature lists v. feature groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feature lists are validated by making sure every element in the list is a string found within a valid features list,
which is a concatenation of all the features listend in `feature_groups.yml` in the parse module.

The parsing of a `feature_group` key is a little more complicated. If the key is simply a string, or a one-element list
that is actually a string, then the string is just validated by checking whether it's a known feature group.
If the value for `feature group` is a list with more than one element, then a new `feature list` is generated that's
the concatenation of features from all groups in the list. A vector of the same length as the new feature list has
values that indicate which feature group each element in the new feature list belongs to.