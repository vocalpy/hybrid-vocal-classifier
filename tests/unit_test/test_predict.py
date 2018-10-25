"""module to test high-level extract function in hvc.predict"""
import os

import hvc


class TestPredict:
    def test_data_dirs_cbins(self, tmp_output_dir, test_data_dir):
        """test that calling predict doesn't fail when we
        pass a data_dirs list that contain cbin audio files"""
        data_dirs = [
            'cbins/gy6or6/032312',
            'cbins/gy6or6/032412']
        data_dirs = [
            os.path.join(test_data_dir,
                         os.path.normpath(data_dir))
            for data_dir in data_dirs
        ]
        file_format = 'cbin'
        model_meta_file = ''
        output_dir = tmp_output_dir
        # explicitly set segment to None because we want to test
        # that default behavior works that happens when
        # we supply argument for data_dirs parameter, **and**
        # segment is set to None (as it should be by default)
        segment = None
        predict_proba = False
        convert_to = 'cbin'  # to check that this works
        return_predictions = True
        predict = hvc.predict(data_dirs=data_dirs,
                              file_format=file_format,
                              model_meta_file=model_meta_file,
                              segment=segment,
                              predict_proba=predict_proba,
                              convert_to=convert_to,
                              return_predictions=return_predictions)
        assert type(predict) == dict
        for key in ['labels', 'pred_labels', 'songfile_IDs', 'onsets_Hz', 'offsets_Hz',
                    'features',]:
            assert key in predict
        # check there are cbin files in output_dir!
