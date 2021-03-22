## Metadata

Metadata is necessary to process input data files, this information is organized in a JSON config that's provided under the input parameters. For more details on how to construct the JSON, see [metadata_template.json](../json/metadata_template.json).

### 1. Data Files

There are 3 main data inputs required to run the SDK, described below.

#### 1.1. Vertices of Interest (vois) Data

This data is provided under:

- `"voi_indices_file"`: A mandatory input, such file should include an array of vois, for example a csv file with a specified col_of_interest_indx.

#### 1.2. Distances/Embeddings Data

This data can be provided by one of these 2 options, either option will be processed to provide an array of distance matrices in the shape of **(number of vois x number of nodes x number of embeddings)**. The order of the distance matrices will correspond to the vois obtained from `"voi_indices_file"` in the same order:

- `"embeddings_files"`: A list of file object(s) containing emeddings representing the distances between each voi and every node. A pairwise_function will process these embeddings to get the array of distance matrices described above.
- `"dist_matrices_files"`: A list of file object(s) containing distances between each voi and every node. These files will be concatenated to get the array of distance matrices described above.

#### 1.3. Similar Nodes Data

Each voi requires a set of nodes that are known to be similar to it, these nodes will be used for training and evaluation purposes. This data can be provided in one of 2 options:

- Option 1:
  
  *Both files need to be provided if this option is chosen. Evaluation nodes cannot overlap with training nodes for a specific voi.*

    - `"eval_data_file"`: contains fixed evaluation nodes for each voi, each voi's evaluation array will be used to evaluate the ranking results for that voi. If used, the input parameter eval_data_mode need to be: "FIXED".

    - `"training_data_file"`: contains fixed training nodes for each voi, each voi's training array will be used to train the ILP model to optimize for the best weighted combination of the embeddings. If used, the input parameter training_data_mode need to be: "FIXED".
  
- Option 2:
    - `"testing_data_file"`: contains testing nodes for each voi, which are used to obtain training and evaluation data randomly chosen of the specified size in input parameters, while making sure the 2 sets don't overlap for that voi. If used, the input parameters eval_data_mode and training_data_mode need to be: "RANDOM".
  
### 2. Metadata File Objects

The metadata json config consists of the above componenets, metadata representing each component is organized in a file object that includes the following parameters:

- `"file_pattern"`: for data that takes in multiple files, you can use the `"file_pattern"` together with a directory path for `"path"` to point to all files in that directory that end with that file pattern assuming they share all other parameters. If you're only using one file, then `"file_pattern"` should be `null`.

- `"url"`: an online link to the data file. If you provide data in this parameter, you will get a warning to modify `read_url()` in [ilp_inputdatafile.py](../python/ilp_inputdatafile.py) to handle it according to the online platform you're using.
  
- `"path"`: either a path to a directory that includes multiple files of the same pattern (`"file_pattern"` is not `null`), or simply the path to the data file (`"file_pattern"` is `null`)
  
- `"file_type"`: the file format. The SDK handles csv, pkl, npy and json files, otherwise it will raise an exception.
  
- `"input_format_type"`: specifies the format of the data, usually used for similar nodes data, it can take one of 3 values:

    - `"ONE SET"`: indicates that the data is simply one array of nodes, not separated by vois and they all share the same set.
 
    - `"HEADER VOIS"`: indicates that the data is in a table format where the header includes all vois and each corresponding column has an array of nodes for that voi.
  
    - `"COLUMN VOIS"`: indicates that the data is in a table format where the first column includes all vois and each corresponding row has an array of nodes for that voi.
  
  
- `"col_of_interest_indx"`: in the case of a table like input, such as a csv file, it specifies the index of the column that contains the data, otherwise it can be `null`.
  
- `"header_row"`: if the file includes a header, it contains the index of the header row, otherwise it can be `null`.
  
- `"header"`: if the file includes a header, this lists the header values, otherwise it can be `null`, this parameter will not affect the code.
  
- `"header_type"`: if the file includes a header, this lists the type of values in the column for the corresponding header, otherwise it can be `null`, this parameter will not affect the code.
  
- `"excluded"`: if set to `true`, this file will not be used, otherwise, it will be taken as input.