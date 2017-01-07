class Syllable:
    """
    Object that represents a syllable.
    Properties:
        position -- starting sample number ("frame") within .wav file *** relative to start of sequence! ***
        length -- duration given as number of samples
        label -- text representation of syllable as classified by a human or a machine learning algorithm
    """
    def __init__(self, position, length, label):
        self.position = position
        self.length = length
        self.label = label
    
    def __repr__(self):
        rep_str = \
            "Syllable labeled {} at position {} with length {}".format(self.label,
                                                                       self.position,
                                                                       self.length) 
        return rep_str

class Sequence:
    """
    Object that represents a sequence of syllables.
    Properties:
        wavFile -- string, file name of .wav file in which sequence occurs
        position -- starting sample number within .wav file
        length -- duration given as number of samples
        syls -- sequence as a list of syllable objects
        seqSpect -- spectrogram object
        
    """
    
    def __init__(self, wav_file,position, length, syl_list):
        self.wavFile = wav_file
        self.position = position
        self.length = length
        self.numSyls = len(syl_list)
        self.syls = syl_list
        self.seqSpect = None
    
    def __repr__(self):
        rep_str = \
            "Sequence from {} with position {} and length {}".format(self.wavFile,
                                                                     self.position,
                                                                     self.length)
        return rep_str
        
    
