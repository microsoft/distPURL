## Input Parameters for the ILP SDK

#### Below is a detailed description of all input parameters, attributes to the [ILPParams class](./../python/__pycache__/ilp_common_classes.py):
1.	#### dataset
    >
        type: <str>
    
        description: name of the dataset to be used.

        default value: "test_set_data"

2.	#### path_to_root:
    >
        type: <str>

        description: path to the repo’s root. 

        default value: "../../"
    
3.	#### path_to_metadata:
    >
        type: <str>
    
        description: path to metadata.json.

        default value: "../data/test_set_data/metadata.json"

4.	#### path_to_output:
    >
        type: <str>
    
        description: path to persist output data to.

        default value: "../output_data/output_test_set/"

*Note: You will need to ensure that the above 4 parameters match for the same dataset used. You might need to update all if you update one.*

5.	#### num_vois:
    >
        type: <int> 
    
        description: number of vertices of interest to combine representations for using ILP.

        default value: 5

6.	#### training_sets_sizes:
    >
        type:  <list[int]>
    
        description: list of all training set sizes to experiment with.

        default value: [10,15]

7.	#### minimum_ranking_thresholds:
    >
        type: <list[float]>
    
        description: list of thresholds to experiment with, optionally used to reduce the program’s runtime by creating an edited distance matrix before optimization where all nodes in the training set have a minimum pre-rank of the specified threshold. If you choose to not use this option and leave the distances unedited, you can leave this value at None. 

        default value: [None]

8.	#### solvers_and_apis:
    >
        type: <list[list[str]]>
    
        description: list of lists (or tuples) of API and solver pairs to experiment with. For example (‘pulp’,’coin_cmd’) or (‘gurobi’,’native_sover’).

        default value: [["pulp","coin_cmd"]]

9.	#### weight_approx_resolutions:
    >
        type: <[list[float]>
    
        description: list of resolutions, optionally used to reduce the program’s runtime by having the optimizing API calculate the weight vectors up to a certain accuracy, for example, a value of 0.001 will result in weight vector values with 3 digits after the decimal, 0.0001 with 4 and so on. You can also choose to not use this option and leave the weight vectors as continuous, by setting this parameter to None. 

        default value: [None]

10.	#### num_cores:
    >
        type: int
    
        description: number of cores to parallelize over.

        default value: 3

11.	#### persist_data:
    >
        type: <bool>
    
        description: toggles the option of whether to persist pre-processed data. Run mode has to be set to DEBUG for this option to be available.

        default value: True

12.	#### mode:
    >
        type: <RunMode>
    
        description: an enum object that takes one of 2 values, RunMode("RELEASE") for minimum algorithm and debugging output, or RunMode("DEBUG") for more detailed logging, more output files for individual steps and persisted data.

        default value: RunMode("DEBUG")

1.  #### eval_data_mode: 
    >
        type: <DataInputMode>
        
        description: an enum object that takes one of 2 values, DataInputMode("RANDOM") for an evaluation set randomly-selected from the testing data, or DataInputMode("FIXED") or a fixed evaluation set provided by the user.

        default_value: DataInputMode("RANDOM")

2.  #### training_data_mode: 
    >
        type: <DataInputMode>
        
        description: an enum object that takes one of 2 values, DataInputMode("RANDOM") for a training set randomly-selected from the testing data, or DataInputMode("FIXED") or a fixed training set provided by the user.

        default_value: DataInputMode("RANDOM")   

15.	#### gurobi_outputflag:
    >
        type: <bool>
    
        description: toggles the option of logging to file and terminal specifically for Gurobi’s API.

        default value: 1

16.	#### time_limit:
    >
        type: <float>
    
        description: specifies a time limit in seconds for the API’s solver.

        default value: 120

17.	#### num_threads:
    >
        type: <int>
    
        description: optionally use multi-threading during to solve.

        default value: 1

18.	#### eval_size:
    >
        type: <int>
    
        description: the size of the evaluation set.

        default value: 10

1.  #### eval_param:
    >
        type: <EvalParam>
        
        description: an enum object that takes one of 2 values, EvalParam("training_sets_size") for evaluating the results relative to the training_sets_sizes, or EvalParam("weight_approx_resolution") for evaluating the results relative to the weight_approx_resolution values.

        default_value: EvalParam("training_sets_size")
