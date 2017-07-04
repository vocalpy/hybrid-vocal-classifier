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

`_validate_todo_list_dict` is called by `validate_yaml` when it finds a key `todo_list` whose value is a dict