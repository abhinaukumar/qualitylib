# QualityLIB
_Seamlessly run quality assessment experiments on subjective datasets_

# Description

QualityLIB is a library that simplifies running quality assessment experiments on video datasets in Python. QualityLIB library interfaces with the [VideoLIB](https://github.com/abhinaukumar/videolib) package to provide an easy API that simplifies quality assessment research tasks such as

1. Specifying and reading datasets of videos, conforming to various ITU standards.
2. Standardizing the implementation of quality models using the `FeatureExtractor` class.
3. Simplifying the execution of feature extraction over datasets using the `Runner` class.
4. Standardizing the results of quality modeling using the `Result` class.
5. Easy interfacing with Scikit-Learn regressor models for routines such as `cross_validation`.

QualityLIB is inspired by the structure of [VMAF's Python library](https://github.com/Netflix/vmaf/blob/master/resource/doc/python.md), but engineered from the ground up to be lightweight and to leverage third-party Python libraries.

# Usage
Refer to the [official documentation](https://qualitylib.readthedocs.io/en/latest/) for examples using QualityLIB and detailed descriptions of the API.

# Installation
To use QualityLIB, you will need Python >= 3.7.0. To install using `pip`, run
```
pip install git+https://github.com/abhinaukumar/qualitylib@main
```
To install using `conda`, install `pip` and `git` in your environment using
```
conda install git pip
```
and use the `pip` command above.

# Issues, Suggestions, and Contributions
The goal of QualityLIB is to share with the community a tool that I build to accelerate my own quality assessment research workflows, and one that I have found great success with. Any feedback that can improve the quality of QualityLIB for the community and myself is greatly appreciated!

Please [file an issue](https://github.com/abhinaukumar/qualitylib/issues) if you would like to suggest a feature, or flag any bugs/issues, and I will respond to them as promptly as I can. Contributions that add features and/or resolve any issues are also welcome! Please create a [pull request](https://github.com/abhinaukumar/qualityilb/pulls) with your contribution and I will review it at the earliest.

# Contact Me
If you would like to contact me personally regarding QualityLIB, please email me at either [abhinaukumar@utexas.edu](mailto:abhinaukumar@utexas.edu) or [ab.kumr98@gmail.com](mailto:ab.kumr98@gmail.com).

# License
QualityLIB is covered under the MIT License, as shown in the [LICENSE](https://github.com/abhinaukumar/qualitylib/blob/main/LICENSE) file.


