# HVC workflow in detail
This document explains in detail how functions and modules work, mainly
as a reference for developers.

Take the example code from the intro notes page:
```Python
import hvc

hvc.extract('extract_config.yml')
hvc.select('select_config.yml')
hvc.predict('predict_config.yml')
```

Here's a step-by-step outline of what happens under the hood:
- `import hvc`
 + automatically imports `featureextract`, `labelpredict`, and
 `modelselect` modules
    - specifically, the `extract`, `predict`, and `select` functions
    from their respective modules

- `hvc.extract('extract_config.yml')`
 + first `parse.extract` parses the config file
 + for each element in `todo_list` from config
   + for each data directory `datadir` in `todo_list`:
     - change to that directory
     - get all the audio files in that directory with `glob`
     - for each audo file:
       + run `features.extract.from_file`
       + add extracted features to `features_from_all_files`
 + save all features in an output file

- `hvc.select('select_config.yml')`

- `hvc.predict('predict_config.yml')`
