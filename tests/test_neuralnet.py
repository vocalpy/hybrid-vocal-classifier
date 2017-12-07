"""
test neuralnet sub-package
"""

import hvc.neuralnet


class TestModels:

    def test_flatwindow(self):
        input_shape = (112, 256, 1)
        num_label_classes = 11
        fw = hvc.neuralnet.models.flatwindow.flatwindow(input_shape,
                                                        num_label_classes)
        assert fw.output_shape == (None, num_label_classes)
        # when freq bins in original spectrogram is 112,
        # num of freq bins in "local window" layer should be 11
        # to exactly reproduce Koumura
        assert fw.layers[-6].input_shape == (None, 11, 29, 16)

