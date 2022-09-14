# Benchmark with Python

Command
```
benchmark_app -m model/v3-small_224_1.0_float.xml
```

[Step 1/11] Parsing and validating input arguments
[ WARNING ]  -nstreams default value is determined automatically for a device. Although the automatic selection usually provides a reasonable performance, but it still may be non-optimal for some cases, for more information look at README. 
[Step 2/11] Loading OpenVINO
[ WARNING ] PerformanceMode was not explicitly specified in command line. Device CPU performance hint will be set to THROUGHPUT.
[ INFO ] OpenVINO:
         API version............. 2022.1.0-7019-cdb9bec7210-releases/2022/1
[ INFO ] Device info
         CPU
         openvino_intel_cpu_plugin version 2022.1
         Build................... 2022.1.0-7019-cdb9bec7210-releases/2022/1

[Step 3/11] Setting device configuration
[ WARNING ] -nstreams default value is determined automatically for CPU device. Although the automatic selection usually provides a reasonable performance, but it still may be non-optimal for some cases, for more information look at README.
[Step 4/11] Reading network files
[ INFO ] Read model took 28.07 ms
[Step 5/11] Resizing network to match image sizes and given batch
[ INFO ] Network batch size: 1
[Step 6/11] Configuring input of the model
[ INFO ] Model input 'input' precision u8, dimensions ([N,H,W,C]): 1 224 224 3
[ INFO ] Model output 'MobilenetV3/Predictions/Softmax:0' precision f32, dimensions ([...]): 1 1001
[Step 7/11] Loading the model to the device
[ INFO ] Compile model took 249.27 ms
[Step 8/11] Querying optimal runtime parameters
[ INFO ] DEVICE: CPU
[ INFO ]   AVAILABLE_DEVICES  , ['']
[ INFO ]   RANGE_FOR_ASYNC_INFER_REQUESTS  , (1, 1, 1)
[ INFO ]   RANGE_FOR_STREAMS  , (1, 40)
[ INFO ]   FULL_DEVICE_NAME  , Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
[ INFO ]   OPTIMIZATION_CAPABILITIES  , ['FP32', 'FP16', 'INT8', 'BIN', 'EXPORT_IMPORT']
[ INFO ]   CACHE_DIR  , 
[ INFO ]   NUM_STREAMS  , 10
[ INFO ]   INFERENCE_NUM_THREADS  , 0
[ INFO ]   PERF_COUNT  , False
[ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS  , 0
[Step 9/11] Creating infer requests and preparing input data
[ INFO ] Create 10 infer requests took 2.53 ms
[ WARNING ] No input files were given for input 'input'!. This input will be filled with random values!
[ INFO ] Fill input 'input' with random values 
[Step 10/11] Measuring performance (Start inference asynchronously, 10 inference requests using 10 streams for CPU, inference only: True, limits: 60000 ms duration)
[ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
[ INFO ] First inference took 4.75 ms
[Step 11/11] Dumping statistics report
Count:          214260 iterations
Duration:       60003.34 ms
Latency:
    Median:     2.43 ms
    AVG:        2.75 ms
    MIN:        1.61 ms
    MAX:        130.30 ms
Throughput: 3570.80 FPS