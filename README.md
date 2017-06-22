# hybrid-vocal-classifier
Python package that automates segmenting and labeling vocalizations. 
The main application of the package is to birdsong behavioral experiments but long-term goals are to make it easy to segment and label any vocalization.

BSD License.

## More about the package
hybrid-vocal-classifier provides a framework to compare different methods to automate the process of labeling vocalizations. In particular, it facilitates labeling birdsong.

In neuroscience, songbirds provide a model system to understand how the brain learns and produces a complex motor skill with many similarities to speech. 

Often neuroscientists carry out behavioral experiments to investigate specific aspects of how the songbird brain learns and produces song. To analyze results from these experiments, the scientists must "label" the elements of birdsong by hand.

Many methods have been proposed to automate the process of labeling birdsong. Recent papers provide code and datasets, but it is often challenging for researchers to install and work with several code bases, then decide which method works best.

### Advantages of hybrid-vocal-classifier:
+ completely open-source, free like free beer and like free speech
+ makes it easy to compare multiple machine learning algorithms
+ almost no coding required, configurable with text files
+ built on top of Python packages road-tested by the greater data-science community
  - [Numpy](http://www.numpy.org/)
  - [Scipy](https://www.scipy.org/scipylib/index.html)
  - [matplotlib](https://matplotlib.org/)
  - [scikit-learn](http://scikit-learn.org/stable/)
  - [keras](https://keras.io/)

## More about the science
Songbirds learn their song during a critical period in development from an adult "tutor". Each individual bird has its own unique song, often very similar to the song of the bird that tutored it.
The songbird brains contain a specialized network of areas required for learning and producing song.
Although this network, known as the song system, is found only in songbird brains, it has evolved on top of the basic floor plan that appears in all vertebrate brains, including humans.
By understanding the song system, we can understand more about our own brain.
Many other aspects of songbird behavior can tell us more about ourselves and environment, and by studying their vocalizations we open a window into their world.
For more information, check out http://songbirdscience.com/

hybrid-vocal-classifier was coded up under the patronage of [the Sober lab](https://scholarblogs.emory.edu/soberlab/)
