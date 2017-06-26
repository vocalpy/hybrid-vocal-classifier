"""
test parse module
"""

import pytest
import yaml

import hvc.parse.extract as extract

with open('./test_data/config.yaml/test_parse_extract.config.yml', 'r') as yaml_file:
    test_yaml_extract = yaml.load(yaml_file)

class TestParseExtract:

    def test_extract_parse(self):

        test_yaml = test_yaml_extract['test_parse']
        # test whether valid yaml parses *without* throwing error
        extract.validate_yaml(test_yaml['valid_with_default_spect_and_seg_params'])
        extract.validate_yaml(test_yaml['valid_with_default_spect_params'])
        extract.validate_yaml(test_yaml['valid_with_default_segment_params'])

        with pytest.raises(KeyError):
            extract.validate_yaml(test_yaml['invalid_no_todo'])

        with pytest.raises(KeyError):
            extract.validate_yaml(test_yaml['invalid_missing_spect_params'])

        with pytest.raises(KeyError):
            extract.validate_yaml(test_yaml['invalid_missing_segment_params'])

    def test_validate_segment_params(self):

        test_yaml = test_yaml_extract['test_segment_params']

        # raises key error when key missing
        with pytest.raises(KeyError):
            extract.validate_segment_params(test_yaml['segparams_missing_key'])

        # raises value error when 'threshold' value is not int
        with pytest.raises(ValueError):
            extract.validate_segment_params(test_yaml['segparams_threshold_wrong_type'])

        # raises value error when 'min_syl_dur' value is not float
        with pytest.raises(ValueError):
            extract.validate_segment_params(test_yaml['segparams_min_syl_dur_wrong_type'])

        # raises value error when 'min_silent_dur' value is not float
        with pytest.raises(ValueError):
            extract.validate_segment_params(test_yaml['segparams_min_silent_dur_wrong_type'])

#class TestParseExtract():