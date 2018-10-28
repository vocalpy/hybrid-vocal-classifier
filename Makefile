clean-feature-files :
	rm -rf ./tests/test_data/feature_files/test_extract*features*
	rm -rf ./tests/test_data/feature_files/hide/

make-feature-files :
	python ./tests/test_data/feature_files/remake_feature_files.py
