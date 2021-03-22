#!/bin/bash

python3 ../python/ilp_pipeline_driver.py -i ../json/pipeline_driver_input_template.json --runid test_run
if [[ $? != 0 ]]; then
        printf 'Test failed.'
        exit 1
fi