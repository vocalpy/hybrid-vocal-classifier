.PHONY: clean-features clean-models

clean-features :
	rm -rf ./tests/test_data/feature_files/test_extract*features*
	rm -rf ./tests/test_data/feature_files/hide/

make-features :
	python ./tests/test_data/feature_files/remake_feature_files.py

clean-models :
	rm -rf ./tests/test_data/feature_files/test_extract*features*
	rm -rf ./tests/test_data/feature_files/hide/

make-models :
	python ./tests/test_data/model_files/remake_model_files.py
