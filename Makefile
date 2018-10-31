.PHONY: all clean

clean :
	rm -rf ./tests/test_data/feature_files/*features*
	rm -rf ./tests/test_data/model_files/*.model
	rm -rf ./tests/test_data/model_files/*.meta

all : models

models : features
	python ./tests/test_data/model_files/remake_model_files.py

features :
	python ./tests/test_data/feature_files/remake_feature_files.py

