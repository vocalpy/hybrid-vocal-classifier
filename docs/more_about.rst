======================================================================
        More about hybrid-vocal-classifier and songbird science
======================================================================

more about hybrid-vocal-classifier
----------------------------------

Scientists that study how individual birds birdsong at the level of individual birds often
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

more about songbird science
---------------------------

Songbirds provide a model system to understand how the brain learns and produces
 speech and other sequential motor skills acquired by imitation, like
 playing the piano or shooting a basketball. Like babies learning to talk,
 songbirds learn their song from an adult "tutor" during
 a sensitive period in development . Each individual bird
 has its own unique song, often very similar to the song of the bird that tutored it.
The songbird brains contain a specialized network of areas required for learning and producing song.
Although this network, known as the song system, is found only in songbird brains,
 it has evolved on top of the basic floor plan that appears in all vertebrate brains, including humans.
By understanding the song system, we can understand more about our own brain.
Many other aspects of songbird behavior can tell us more about ourselves and environment,
 and by studying their vocalizations we open a window into their world.
For more information, check out http://songbirdscience.com/

hybrid-vocal-classifier was developed in
`the Sober lab<https://scholarblogs.emory.edu/soberlab/>`_