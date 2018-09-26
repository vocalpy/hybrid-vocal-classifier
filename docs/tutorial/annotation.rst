.. _annotation_how_to:

=====================================================
How do I use song I've annotated with other software?
=====================================================

`hybrid-vocal-classifier` facilitates applying
machine learning algorithms to birdsong. This means the library
does not focus on being a general audio processing toolkit.
So, even though the library includes functions for segmenting song into
syllables, these functions simply make it possible to test the library on songs available
in open data repositories, i.e. to demonstrate that the library works. So
how can you apply `hybrid-vocal-classifier` to your own song that you may
have acquired and annotated with some other set of software?

The most general way to work with `hybrid-vocal-classifier` is through
**annotation files**. An annotation file is simply a comma-separated value file
with a specific set of fields (best practices for data storage and sharing recommend
simple text formats like comma-separated value files, see for example
https://www2.usgs.gov/datamanagement/plan/dataformats.php). Each row of an annotation
file corresponds to one syllable from one song, and will have the following
fields: filename, onset_s, offset_s, onset_Hz, offset_Hz, label. The filename
field makes it possible for one annotation file to contain annotations from many
songs.

To create your own annotation files, you'll need to write a short script to convert
from the format you are using. Here's an example, taken from a couple of functions in
`hybrid-vocal-classifier` that convert the `.not.mat` files (saved by the Evsonganaly.m
GUI) into annotation files. The first function loops over a list of `.not.mat` files, like so:

.. literalinclude:: ../../hvc/utils/annotation.py
   :lines: 248-251

and the second converts each syllable in each .not.mat file into a line in the annotation
file:

.. literalinclude:: ../../hvc/utils/annotation.py
   :lines: 90-130