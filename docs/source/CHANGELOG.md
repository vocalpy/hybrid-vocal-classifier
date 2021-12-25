# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1]
### changed
- remove `evfuncs` module, replace with external `evfuncs` library
  [#114](https://github.com/NickleDave/hybrid-vocal-classifier/pull/114)
- switch from using `poetry` to using `flit` for development, 
  makes it possible to specify metadata according to PEP 621
  [#123](https://github.com/NickleDave/hybrid-vocal-classifier/pull/123).

### fixed
- change 'requires-python' constraints to include Python 3.9
  [#123](https://github.com/NickleDave/hybrid-vocal-classifier/pull/123).

## [0.3.0](https://github.com/NickleDave/hybrid-vocal-classifier/releases/tag/0.3.0) -- 2021-04-10
### fixed
- make various fixes so that `hvc` works with current versions of dependencies
  [#101](https://github.com/NickleDave/hybrid-vocal-classifier/pull/101)

## 0.2.1b1
### added
- unit tests for high-level extract, select, and predict functions (see 'changed')
- Allow supplying annotation as csv; this will get replaced by using `Crowsetta` to
  deal with annotation shortly

### changed
- **change to API**: Provides direct access to hvc.extract, select, and predict 
  through functions, instead of requiring YAML config files. Can still use YAML though.
- add `FeatureExtractor` class to `hvc.features`
  + high-level `hvc.extract` function basically instantiates a `FeatureExtractor` for the
    user

### fixed
- fix bugs in how `koumura` functions parse xml (error in loop logic that failed to parse
  last sequence)

## 0.1.b2
### Added
- can now train scikit-learn models with 'predict probability' set to `True`

### Fixed
- `flatwindow` model in Keras works
- `svm` feature extraction deals appropriately with segments from which it can't 
  extract features by returning zeros instead; 
  this reproduces approach of Tachibana Okinaya paper

##  0.1.b1 
### Fixed
- Fix setup.py issues, stupid syntax errors that are causing errors with conda build

## 0.1.0-beta
### Added
- predict module working
- can convert predictions to at least one file format
- hence going to beta

## 0.1.0-alpha
- First release
