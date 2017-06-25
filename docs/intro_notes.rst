# intro notes

`hybrid-vocal-classifier` (or `HVC` for short) is a Python library whose
goal is to make it easy to apply machine learning algorithms that
automatically classify the elements of birdsong, often referred to as
 syllables.

Writing scripts that run `HVC` requires almost no coding.
The user writes configuration files in YAML, a language for specifying
data structures. YAML is meant to be easy for humans to read and write.
Most users will only have to copy the example .yml files and then
change a couple of parameters.

Here's how you'd run an analysis using `HVC`:

```Python
import hvc

hvc.extract('extract_config.yml')
hvc.select('select_config.yml')
hvc.predict('predict_config.yml')
```

## But, why?
Scientists that study birdsong at the level of individual birds often
have to label the syllables of each bird's song by hand in order to get
 results.

Because birds can sing hundreds of songs a day, it requires many
person-hours to label song.

In addition, in many songbird species that have been studied,
each individual learns a song that is similar but not exactly the same
as the song of the bird or birds that tutored it.

Therefore any software that automates the process of labeling syllables
must classify them with very high accuracy and must do so in a way that
is robust across the songs of many different individuals from a species.

As stated, the primary goal of the `HVC` package is to make it easier
for any scientist to apply machine-learning algorithms to birdsong.

The secondary goal of the package is to facilitate comparisons of
different machine learning algorithms. Several groups have published
on various algorithms but little work has been done to compare accuracy.

A final goal is to entice the field of artificial intelligence to study
birdsong. Birdsong presents an ideal test-bed to experiment with machine
learning algorithms that segment time-series data, i.e., that decide at
what time point each segment starts and stops. Unlike in speech,
syllables in birdsong are typically discrete elements separated from
each other by brief silent gaps. Algorithms for speech-to-text have
successfully avoided dealing with segmentation, but there are many cases
where it would be useful to have high accuracy segments (e.g., automated
analysis of speech disorders where duration may be affected).

<sub>* A data serialization language--for the non-computer science
people--is a language that represents data types
like an array in such a way that they can be easily stored and/or
transmitted.</sub>