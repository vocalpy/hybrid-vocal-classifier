"""this file tests **just** the high-level extract function in hvc.predict.
More specifically it tests running the function **without** config.yml scripts,
instead using pure Python"""
import os


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
                              return_features=return_features)
        assert type(ftrs) == dict
        assert sorted(ftrs.keys()) == ['features', 'labels']
        # check there are cbin files in output_dir!
