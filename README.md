# MineSplines

This is sourcecode for the minespline project. 

The minespline project is a library for grouping potential detections of landmines into curves. 

# Example Usage
<b>MinelineHardCases1_1 geojson file</b>
```bash
python -m minespline.mineliner evaluate MinelineHardCases1_1.geojson
```
Input         |  Output
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/c1AmnAR.jpg" width="700" height="600"/>  |  <img src="https://i.imgur.com/33lCat4.jpg" width="600" height="500"/> 

<b>VeryHardWithoutDistractors_1 geojson file</b>
```bash
python -m minespline.mineliner evaluate VeryHardWithoutDistractors_1.geojson
```
Input         |  Output
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/nj4YvFQ.jpg" width="700" height="600"/>  |  <img src="https://i.imgur.com/oeH2i0L.jpg" width="600" height="500"/> 


# Requirements

- [Anaconda3](https://www.anaconda.com/distribution/), for **python 3.7+** must be installed.

# Setup

1. Obtain the sourcecode. If you have access to the git repo, then you can use
    ```bash
    git clone https://https://gitlab.com/mu-indepth/minespline
    ```
2. Use anaconda to install the project dependancies.
   This will create our own virtual python environment called `ML` (for Mine Lines). 
   
    ```bash
    cd minespline
    conda env create
    ```
     
    * If you already have an `ML` environment for this project, you should instead update it
     using 
     ```bash
     conda env update
     ``` 
    
    
   
# Usage

Before doing anything, make sure the ML environment is active

```bash
conda activate ML
```

If you want to run one of the command below from a script, you can use
```bash
conda run -n ML <Rest of the command goes here>
```
which will run the script in the appropriate environment. 
 

# Experiments
```bash

conda activate ML

# The most recent "very hard" examples
python -m minespline.mineliner evaluate ./data/test/very-hard/*.geojson

# Plot an example at a different threshold and throw away the output
python -m minespline.mineliner --max-spacing 9 --top-k 10 --threshold 0.00001 show data/test/RealCurvilinearData/RealDat_Curvi_1_1.geojson --plot -o /dev/null

```

#### Findine Minelines


From the command line interface (CLU) 
```bash
(ML)$  python -m minespline.mineliner --help 

Usage: mineliner.py [OPTIONS] COMMAND [ARGS]...

Options:
  --min-length INTEGER  The minimum length of a path  [env var: ML_MIN_LENGTH;
                        default: 3]
  --top-k INTEGER       The number of paths to return (in descending order)
                        [env var: ML_TOP_K; default: 3]
  --max-it INTEGER      The max length of a path  [env var: ML_MAX_IT;
                        default: 9223372036854775807]
  --threshold FLOAT     The minimum probability of a path  [env var:
                        ML_THRESHOLD; default: 1e-35]
  --spacing FLOAT       The mine spacing, in meters.  [env var: ML_SPACING;
                        default: 8]
  --std-along FLOAT     The standard deviation in mine spacing along the mine
                        line, in meters.  [env var: ML_STD_ALONG; default:
                        0.2]
  --std-across FLOAT    The STD in mine spacing across the mine line, in
                        meters.  [env var: ML_STD_ACROSS; default: 0.0625]
  --max-spacing FLOAT   The maximum distance between two mines  [env var:
                        ML_MAX_SPACING; default: 17]
  --prob-miss FLOAT     The probability of a false-negative in the input  [env
                        var: ML_PROB_MISS; default: 0.05]
  --no-guess-spacing    Do not guess the spacing -- instead use the value
                        passed in or specified in the file  [env var:
                        ML_NO_GUESS_SPACING; default: False]
  --no-progress         Disable progress bars  [env var: ML_NO_PROGRESS;
                        default: False]
  --help                Show this message and exit.

Commands:
  evaluate  Process many files and summarize results, saving a CSV and
            plots...
  show      Process a single input file and save the results, optionally...

``` 
Will provide help that is most up to date. 
Most of the options allow you to set parameters that control mine processing. 

At the time of this writing the CLI has two main commands:

```bash 
(ML)$  python -m minespline.mineliner  show --help

Usage: mineliner.py show [OPTIONS] INPUT_FILE

  Process a single input file and save the results, optionally showing a
  plot.

Options:
  -o, --output FILENAME  GeoJSON formatted output  [env var: ML_SHOW_OUTPUT]
  --plot                 Display a plot of the results  [env var:
                         ML_SHOW_PLOT]
  --help                 Show this message and exit.

```


```bash
(ML) $ python -m minespline.mineliner  evaluate --help
Usage: mineliner.py evaluate [OPTIONS] [FILES]...

  Process many files and summarize results, saving a CSV and plots of each
  file.

Options:
  -o, --outdir DIRECTORY  Directory to save per-file results  [env var:
                          ML_EVALUATE_OUTDIR]
  --p-sep FLOAT           Set the difference in means (-1 means leave alone)
                          [env var: ML_EVALUATE_P_SEP; default: -1]
  --p-std FLOAT           Set the std. of each class (before clipping to 0/1)
                          (-1 means leave alone)   [env var:
                          ML_EVALUATE_P_STD; default: -1]
  --drop FLOAT            Fraction of positive examples to drop  [env var:
                          ML_EVALUATE_DROP; default: 0]
  --rep INTEGER           Number of times to process each file  [env var:
                          ML_EVALUATE_REP; default: 1]
  --help                  Show this message and exit.

```


##### Ingest COBRA data
We operate on GEOJSON files and GEOTIFF files. Other formats used e.g. by COBRA are converted. 

```python
python -m minespline.ingest_cobra --help
```

##### Sample a Markov-Chain of Mines

Given inital points with confidence values that can be interpreted 
as probabilities, we randomly sample a sequence of mines that form 
a markov chain.  We assign a probability to the entire chain.  The output 
is a single GEOJSON file with a linestring for the entire path, point features for each mine, and 
line features for each edge. 

```
python -m minespline.sample --help
```


