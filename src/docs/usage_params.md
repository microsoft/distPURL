## Usage Parameters

### 1. Input Parameter options (these options can be edited in the json template)

The user may choose to modify as many of the input parameters as they wish, any unchanged parameters keep their default value, see [input_params.md](input_params.md), for more details.
If you want to edit the parameters in place in the JSON file, please edit the sample template: [pipeline_driver_input_template.json](./src/json/pipeline_driver_input_template.json) and use the
option --ifile or -i to provide the json file input.


*Note: Do NOT use any spaces in any list input to avoid errors in reading the values.* 
> --dataset:

    $ python3 ilp_pipeline_driver.py --dataset <str>

> --root-path:

    $ python3 ilp_pipeline_driver.py –-root-path <str>

> --metadata-path:

    $ python3 ilp_pipeline_driver.py --metadata-path <str>

> --output-path: 

    $ python3 ilp_pipeline_driver.py –-output-path <str>

> --num-vois:

    $ python3 ilp_pipeline_driver.py –-num-vois <int>

> --training-sets-sizes: 


    $ python3 ilp_pipeline_driver.py –-training-sets-sizes <list[int]>

> --minimum-ranking-thresholds: 

    $ python3 ilp_pipeline_driver.py –-minimum-ranking-thresholds <list[float]>

> --solvers-apis: 

    $ python3 ilp_pipeline_driver.py -–solvers-apis <list[list[str]]>

>	--weight-approx-resolutions: 

    $ python3 ilp_pipeline_driver.py --weight-approx-resolutions <list[float]>

> --num-cores: 

    $ python3 ilp_pipeline_driver.py --num-cores <int>

> --persist-data: 

    $ python3 ilp_pipeline_driver.py --persist-data <bool>

> --mode: 

    $ python3 ilp_pipeline_driver.py --mode <str>

> --eval-data-mode:

    $ python3 ilp_pipeline_driver.py --eval-data-mode <str>

> --training-data-mode:

    $ python3 ilp_pipeline_driver.py --training-data-mode <str>

> --gurobi-outputflag: 

    $ python3 ilp_pipeline_driver.py --gurobi-outputflag <bool>

> --time-limit: 

    $ python3 ilp_pipeline_driver.py --time-limit <float>

> --num-threads: 

    $ python3 ilp_pipeline_driver.py --num-threads <int>

> --eval-size:

    $ python3 ilp_pipeline_driver.py --eval-size <int>

> --eval-param:
> 
    $ python3 ilp_pipeline_driver.py --eval-param <str>

### 2. JSON file input options

> --create/-c, Create a new file to be saved under [driver_input_temp/](../driver_input_temp/):

*Note: Make sure to change all the parameters you want first!*

    $ python3 ilp_pipeline_driver.py --create <str>.json

    $ python3 ilp_pipeline_driver.py -c <str>.json

> --ifile/-i, Use a pre-written file:

    $ python3 ilp_pipeline_driver.py --ifile <str>.json

    $ python3 ilp_pipeline_driver.py -i <str>.json

### 3. Other Options

> --help/-h, Help, for a description of available options:

    $ python3 ilp_pipeline_driver.py --help

    $ python3 ilp_pipeline_driver.py -h

> --runid, Run ID/label specification, otherwise it will be given a unique numerical ID:

    $ python3 ilp_pipeline_driver.py --runid <str>

> --indices, To specify indices , otherwise they will be chosen randomly:

    $ python3 ilp_pipeline_driver.py --indices <list[int]>

> --manual, Manual inputting, recommended for first-time users unfamiliar with necessary input parameters:

    $ python3 ilp_pipeline_driver.py --manual

> Use the current parameters, no JSON file input (whether default values have been modified or not):

    $ python3 ilp_pipeline_driver.py 
