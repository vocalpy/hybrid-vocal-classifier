import sys
import os
from glob import glob
import shutil

import hvc

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(here, '../../utils'))

# have to append utils to sys before importing rewrite_config from utils.config
# utils is utilities used just for testing; considered best practice to not have
# sub-packages that are used during tests with pytest.
# Other alternative would be to have a pytest fixture in conftest.py that returns
# the utility function(s) but this is less readable to me.
from config import rewrite_config


def main():
    feature_files_to_create = [
        'knn',
        'svm',
        'multiple_feature_groups',
        'flatwindow',
    ]
    for feature_to_create in feature_files_to_create:
        extract_config = os.path.join(here,
                                      '..',
                                      'config.yml',
                                      'test_extract_{}.config.yml'
                                      .format(feature_to_create)
                                      )
        print('running {} to create feature file'.format(extract_config))
        replace_dict = {'output_dir':
                            ('replace with tmp_output_dir',
                             here)}
        # have to put tmp_output_dir into yaml file
        extract_config_rewritten = rewrite_config(extract_config,
                                                  here,
                                                  replace_dict)
        hvc.extract(extract_config_rewritten)
        extract_output_dir = glob(os.path.join(here,
                                               '*extract*output*'))
        if len(extract_output_dir) != 1:
            raise ValueError('incorrect number of outputs when looking for extract '
                             'ouput dirs:\n{}'. format(extract_output_dir))
        else:
            extract_output_dir = extract_output_dir[0]

        features_created = glob(os.path.join(extract_output_dir,
                                             'features_created*'))
        if len(features_created) != 1:
            raise ValueError('incorrect number of outputs when looking for extract '
                             'feature files:\n{}'. format(features_created))
        else:
            features_created = features_created[0]
        movename = feature_to_create + '.' + 'features'
        shutil.move(src=features_created,
                    dst=os.path.join(here, movename))
        os.rmdir(extract_output_dir)
        os.remove(extract_config_rewritten)


if __name__ == '__main__':
    main()
