==============
named features
==============

These features are pre-defined and can be referred to by name in the `feature_list` of YAML files for `extract`.

feature group `Tachibana`:
==========================

 - `mean_spectrum`
 - `mean_delta_spectrum` : 5-order delta of spectrum
 - `mean_cepstrum`
 - `mean_delta_cepstrum`  : 5-order delta of cepstrum
 - `dur` : duration
 - `mA` : additional subset of features listed below

 - `SpecCentroid`
 - `SpecSpread`
 - `SpecSkewness`
 - `SpecKurtosis`
 - `SpecFlatness`
 - `SpecSlope`
 - `Pitch`
 - `PitchGoodness`
 - `Amp`

References
----------
.. [1] Tachibana, Ryosuke O., Naoya Oosugi, and Kazuo Okanoya. "Semi-
automatic classification of birdsong elements using a linear support vector
 machine." PloS one 9.3 (2014): e92584.
