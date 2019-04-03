# A fast implementation of SigProfiler [1]

## License and Disclaimer

This is a 'best faith' effort at implementing SigProfiler as described in [1]. To my best knowledge, no one has attempted to reproduce signatures found in [1] using this repository. Note that the 'spherical k-means clustering' algorithm described in [1] is not the clustering algorithm used to find centroids in this implementation.

This repository was produced for the purpose of a class project and is released publically via the CRAPL license.

## Usage

Please refer to the following examples for usagage:
1. An example of training SigProfiler - [train_sigprofiler.py](https://github.com/theJasonFan/Reproducing-NikZainal2016/blob/master/scripts/train_sigprofiler.py)
2. Am example of accessing learned signatures -[plot_err_vs_silhouette.py](https://github.com/theJasonFan/Reproducing-NikZainal2016/blob/master/scripts/plot_err_vs_silhouette.py)

## Installation
    pip install .

## References

1. Alexandrov, L. B., Nik-Zainal, S., Wedge, D. C., Campbell, P. J. & Stratton, M. R. (2013) "Deciphering signatures of mutational processes operative in human cancer." _Cell Rep._  **3**, pages 246–259. [doi:10.1016/j.celrep.2012.12.008](https://doi.org/10.1016/j.celrep.2012.12.008)