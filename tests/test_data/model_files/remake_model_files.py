import sys
import os
from glob import glob
import shutil

import joblib

import hvc

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(here, '../../utils'))

# have to append utils to sys before importing rewrite_config from utils.config
# utils is utilities used just for testing; considered best practice to not have
# sub-packages that are used during tests with pytest.
# Other alternative would be to have a pytest fixture in conftest.py that returns
# the utility function(s) but this is less readable to me.
from config import rewrite_config

config_feature_file_pairs = {
    'knn': ('test_select_knn_ftr_grp.config.yml',
            'knn.features'),
    'svm': ('test_select_svm.config.yml',
            'svm.features'),
    'flatwindow': ('test_select_flatwindow.config.yml',
                   'flatwindow.features'),
}
feature_files_dir = os.path.join(here, '..', 'feature_files')


def main():
    for model_name, (select_config,
                     feature_filename) in config_feature_file_pairs.items():
        print('running {} to create model files'.format(select_config))
        # have to put tmp_output_dir into yaml file
        select_config = os.path.join(here, '..', 'config.yml', select_config)
        feature_file = glob(os.path.join(feature_files_dir, feature_filename))
        if len(feature_file) != 1:
            raise ValueError('found more than one feature file with search {}:\n{}'
                             .format(feature_filename, feature_file))
        else:
            feature_file = feature_file[0]

        replace_dict = {'feature_file':
                            ('replace with feature_file',
                             feature_file),
                        'output_dir':
                            ('replace with tmp_output_dir',
                             here)}

        select_config_rewritten = rewrite_config(select_config,
                                                 here,
                                                 replace_dict)
        select_output_before = glob(os.path.join(here,
                                                 '*select*output*'))

        hvc.select(select_config_rewritten)

        select_output_after = glob(os.path.join(here,
                                                '*select*output*'))
        select_output_dir = [after for after in select_output_after
                             if after not in select_output_before]

        if len(select_output_dir) != 1:
            raise ValueError('incorrect number of outputs when looking for extract '
                             'ouput dirs:\n{}'. format(extract_output_dir))
        else:
            select_output_dir = select_output_dir[0]

        # arbitrarily grab the last .model and associated .meta file
        model_file = glob(os.path.join(select_output_dir,
                                       '*',
                                       '*.model'))[-1]
        model_file_dst = os.path.join(here, model_name + '.model')
        shutil.move(src=model_file,
                    dst=model_file_dst)
        meta_file = glob(os.path.join(select_output_dir,
                                      '*',
                                      '*.meta'))[-1]
        meta_file_dst = os.path.join(here, model_name + '.meta')
        shutil.move(src=meta_file,
                    dst=meta_file_dst)

        # need to change 'model_filename' in .meta file
        meta_file = joblib.load(os.path.join(here, meta_file_dst))
        meta_file['model_filename'] = os.path.abspath(model_file_dst)
        joblib.dump(meta_file, meta_file_dst)

        # clean up -- delete all the other model files, directory, and config
        shutil.rmtree(select_output_dir)
        os.remove(select_config_rewritten)


if __name__ == '__main__':
    main()
