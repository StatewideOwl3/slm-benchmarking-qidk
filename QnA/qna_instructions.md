```
./build/bin/llama-cli -m <model> -p "<prompt>" --single-turn --no-cnv --seed 0 --temp 0 --top-k 1 --top-p 1 --n-predict 256 --no-cnv --ctx-size 2048 --threads 6 --threads-batch 6 --batch-size 2048 --ubatch-size 512 --no-warmup 
```

- Write a python script that is stored and ran from the llama.cpp directory. The script should run a squad_50.json script qna benchmark on the model given as cli input. It expects path to model to be given as cli arg, relative to the llama.cpp library. it will be of the form "../model/<model_name.gguf>"

- Generate 3 prompt format chat templates variables. Include a system prompt asking it to keep its answers precise as it is undergoing a qna benchmark and it will be evaluated on f1, em like metrics. Also ask it to demarque the start of its answer using special text like "***ANSWER***", so we know which part to collect from stdout and evaluate with the gold answer. In stdout, when the model is done answering, it will print the following text: "[end of text]". We need to collect the text from its starting till this token from stdout.
Add placeholders in these strings to add data from the 50 squad questions 1 by 1. Choose the format string to use based on the input model - only 3 types - Gemma, Qwen and Llama.

- i want the script to start a subprocess and launch this command in it. I want it to use OS calls to measure memory taken too for each run (this will be running inside termux on Snapdragon 8 gen 3). I want the results to be formatted into a json object.

- The command itself will print to stdout the model's response and to stderr the below profiling lines:
```
llama_perf_sampler_print:    sampling time =     176.40 ms /   726 runs   (    0.24 ms per token,  4115.62 tokens per second)
llama_perf_context_print:        load time =     313.18 ms
llama_perf_context_print: prompt eval time =      17.37 ms /    14 tokens (    1.24 ms per token,   805.89 tokens per second)
llama_perf_context_print:        eval time =    6026.70 ms /   711 runs   (    8.48 ms per token,   117.97 tokens per second)
llama_perf_context_print:       total time =    6451.08 ms /   725 tokens
llama_perf_context_print:    graphs reused =        708
llama_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - Host               |                  764 =   224 +      27 +     513                |
llama_memory_breakdown_print: |   - CPU_REPACK         |                   53 =    53 +       0 +       0                |
```

- I want you to collect all of these metrics from stderr (get these raw lines first, store them in a file called raw_metrics.json). Then i want you to parse these results and extract the data from this.

- The raw_outputs{model_name_from_path}.json should be a list of objects of the below style:

```json
{
    "id": <start from 0 to 49>,
    "question": <string - from "question" field in squad_50.json>,
    "gold_answers": <list[strings] - list of answers from "answers" field in squad_50.json>,
    "prompt": <string - formatted prompt sent to the model with chat correct formats, context and question filled in>,
    "model_answer": <string - RAW stdout fully>,
    "stderr" : <string - last lines of RAW stderr starting with llama_perf_sampler_print or llama_perf_context_print>,
    "memory_usage_mb": <int - peak memory usage in MB during the run>,
    ... whatever memory stats os gives
}
```

- I then want you to parse the raw stdout and stderr field strings. So it is incredibly important you get the raw_output strings correctly from stdout and stderr. I need you to save it as parsed_outputs_{model_name_from_path}.json, which will be a list of json objects of the below kind:

```json
{
  "id": <0 to 49 int>
    "question": <string - from "question" field in squad_50.json>,
    "gold_answers": <list[strings] - list of answers from "answers" field in squad_50.json>,
    "prompt": <string - formatted prompt sent to the model with chat correct formats, context and question filled in>,
    "model_answer": <string - RAW stdout fully>,
    "sampling_time" : [
        "total_sampling_time_ms" : <the sampling time from output line llama_perf_sampler_print>,
        "total_sampling_runs" : <runs (after the '/'),
        "time_per_token_ms": <total_sampling_time_ms/total_runs>,
        "tokens_per_sec" : <total_runs/total_sampling_time_ms> * 1000,   
    ],
    "load_time_ms": <load_time in ms parsed from output>,
    "prompt_eval_time" : [
        "total_prompt_eval_time_ms" : <the prompt eval time from output line llama_perf_context_print>,
        "total_prompt_eval_tokens" : <tokens (after the '/' on the same line as above field)>,
        "time_per_token_ms" : <total_prompt_eval_time_ms/total_prompt_eval_tokens>,
        "tokens_per_sec" : <total_prompt_eval_tokens/total_prompt_eval_time_ms> * 1000,
    ],
    "eval_time": [
        "total_eval_time_ms" : <the eval time from output line llama_perf_context_print>,
        "total_eval_tokens" : <tokens (after the '/' on the same line as above field)>,
        "time_per_token_ms" : <total_eval_time_ms/total_eval_tokens>,
        "tokens_per_sec" : <total_eval_tokens/total_eval_time_ms> * 1000,
    ],
    "total_time" : <parsed from total line>,
    "total_tokens": <parsed from total line>,
    .. same memory stats form OS as in raw_outputs.json
}
```

- Here is the llama prompt template:
```text
main: chat template example:
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there<|eot_id|><|start_header_id|>user<|end_header_id|>

How are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

- Here is the qwen prompt template:
```text
main: chat template example:
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant

```

- Here is the gemma prompt template:
```text
main: chat template example:
<start_of_turn>user
You are a helpful assistant

Hello<end_of_turn>
<start_of_turn>model
Hi there<end_of_turn>
<start_of_turn>user
How are you?<end_of_turn>
<start_of_turn>model

```

- Ensure to add comments thoroughly.
