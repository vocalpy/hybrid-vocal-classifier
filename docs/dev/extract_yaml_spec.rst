===================================================
spec for YAML files to configure feature extraction
===================================================

This document specifies the structure of HVC config files written in
YAML. It is a painfully dry document that exists to guide the project
code, not to teach someone how to write the files. For a gentle
introduction to writing the files, please see
[writing config files for feature extraction](writing_extract_yaml.md) .

`extract` config files may define the following keys:

<table>
  <tbody>
    <tr>
      <th>Key</th>
      <th align="center">Value Type and Definition</th>
    </tr>
    <tr>
      <td>spect_params</td>
      <td align="left">
      Dictionary, parameters for spectrogram
        <ul>
            <li>samp_freq : integer, sampling frequency (Hz)</li1>
            <li>window_size : integer, FFT window length in number of samples</li>
            <li>window_step : integer, number of samples to step forward for each window</li>
            <li>freq_cutoffs : list of 2 integers [low,high], range of frequencies to keep</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>feature_list</td>
      <td align="left">list, named features. See the
       <a href="named_features.md">list of named features here</a></td>
    </tr>
    <tr>
      <td>feature_group</td>
      <td align="left">named group of features</td>
    </tr>
    <tr>
      <td>todo_list</td>
      <td align="left">list of dictionaries, 'jobs' to run. Typically data from one subject.
      Each dictionary in the list must define the following fields:
        <ul>
            <li>subject_ID : string, alphanumeric.</li>
            <li>dirs : list of strings, names of directories with
            audio files from which to exract features</li>
            <li>labelset : string, labels of syllables from which
            features should be extracted. Provide the set as a single
            string, e.g., `iabcdefg`. If a label appears in the labeled data but
            does not appear in this string, it will be ignored.</li>
            <li>output_dir : string, directory name in which to save output, the extracted feature files</li>
        </ul>
    </td>
    </tr>
  </tbody>
</table>

## example `extract_config.yml`

```YAML
spect_params :
    samp_freq : 32000 # Hz
    window_size : 512
    window_step : 32
    freq_cutoffs : [1000,8000]
```