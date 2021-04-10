.PHONY: all clean

clean :
	rm -rf ./tests/data_for_tests/feature_files/*features*
	rm -rf ./tests/data_for_tests/model_files/*.model
	rm -rf ./tests/data_for_tests/model_files/*.meta
	rm -rf ./tests/data_for_tests/model_files/select_output*
	rm -rf ./tests/data_for_tests/model_files/test*rewrite.yml

all : models

models : features
	python ./tests/scripts/remake_model_files.py

features :
	python ./tests/scripts/remake_feature_files.py
