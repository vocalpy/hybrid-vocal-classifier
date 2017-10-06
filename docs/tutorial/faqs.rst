====
FAQs
====

* **How many files of hand-labeled song do I need to train good models?**
Good question. Our current best estimate is that,
at least for Bengalese finch song,
you can get > 99.5% accuracy using the `flatwindow` convolutional neural net
model with 500 hand-labeled syllables. 
https://github.com/NickleDave/ML-comparison-birdsong/blob/master/figure_code/analysis%20for%20scipy%202017%20talk.ipynb
Assuming that each file contains one
song bout where the bird sings 50 syllables, that would be 10 files of song. Not bad.

The best way to know for sure is to generate an accuracy curve.