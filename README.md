# 🧪 Meta AI Scientist

This is an early proof-of-concept of an automated scientist agent, with a focus on running ML experiments.

In the current framework, an LLM-based scientist agent (or team of such agents) repeatedly generates hypotheses, implements these hypotheses in code, executes these experiments as Slurm jobs, and summarizes the results of these jobs for the next iteration.

## 📋 Todos
- [x] Create a single high-level `launch_scientist.py` integrated with `hydra` for config. 
- [x] Refactor existing scientist scripts (`climb_nanogpt.py` and `climb_collatz.py`) to be configurations of a single structured `ScienceRunner` class
	- [x] Break out hypothesis generation and hypothesis implementation logic into simple instances of `Ideator` and `Coder`.
- [x] Enable `ScienceRunner` to run multiple experiments per iteration in parallel
- [x] Add support for diff-based editors (e.g. via Aider-based Coders)
- [x] Support third-party LLM APIs in Azure in `core.llm_client` and `core.coder.aider` (o1-preview support added.
- [x] Add options to configure `BoNScienceRunner` to mimic the selection logic of `AIDE`.
- [x] Add ability to condition on prior knowledge when formulating hypotheses.
- [x] Integrate MLE-Bench tasks via a script to automatically import individual tasks via their competition IDs.
- [x] Add ability to run jobs locally on the same machine that is running the scientist (set `slurm_config_args.use_local_runs=True`.
- [x] Gracefully handle preemption and re-entry.
- [ ] Add a basic web interface to explore a running or previous scientist run.

## Run examples

#### r1
First, spin up an instance of `r1-32b`:

```bash
python serve_vllm.py
```

Find the node id for this vllm job on Slurm and run one of the scientist scripts:
```
python launch_scientist.py node_id=<vllm node id> model=r1_32b task=collatz
```

#### o1-preview
To use `o1-preview`, you do not need to spin up a separate server, as requests go to Meta's Azure instance:
```bash
python launch_scientist.py model=o1_preview task=collatz
```

#### AIDE
To run with AIDE-style search, explicitly set the science_runner type to `aide`:
```bash
python launch_scientist.py \
	model=o1_preview \
	science_runner=aide \
	task=collatz
```

#### Knowledge sources
To pass in external knowledge sources that then inform the idea generation stage, pass in a list of file paths or glob strings to the source files (note the quotations around the argument and value here):
```bash
python launch_scientist.py \
	model=o1_preview \
	task=collatz \
	'knowledge_src_paths=["data/knowledge_nanogpt/*.md"]'
```

See the available models and tasks under `config/model` and `config/task` respectively. You can pass the name of any of these yaml files (without the extension) as the value for `launch_scientist.py`'s' model and task command-line arguments.

### Adding a new task

Adding a new task requires only a few steps:

1. Create a new _workspace template_ as its own folder under `workspace_templates/`. This is the set of starting files available to the scientist. Whatever is in the task's workspace template is copied into the `v_1` workspace when running the scientist.

2. Create a new task config under `config/task/`. Remember to set the header `# @package _global_`. See the existing configs for examples. In particular `collatz.yaml` provides an example for experiments requiring only CPUs, `picogpt.yaml`, requiring GPUs, and `nanogpt.yaml` requiring GPUs and the use of `torchrun`.

3. Change the value of task in `config/default.yaml` to the name of your task's yaml file created above (or create your own top-level config with `task` configured appropriately).

#### Automatically import an MLE-Bench task

Before adding any Kaggle, first make sure you have stored your kaggle login credentials at `~/.kaggle/kaggle.json`.

You can use the script `make_mlebench_task.py` to automatically generate a scientist task config based on an MLE-Bench task's competition ID. The below command will generate the necessary files in `task/config/mlebench` and `workspace_templates/mlebench` for MLE-Bench task `random_acts_of_pizza`:
```bash
python make_mlebench_task.py \
--task_id=random-acts-of-pizza \
--cache_dir_path=/path/for/storing/kaggle/datasets \
--lower_is_better
```

The first time you run this command for a new MLE-Bench task, you will be asked to visit the competition page in your browser to accept the competition rules.

You can then run the scientist with this task as follows by specifying the task in the launch command as follows: `python launch_scientist.py task=mlebench/random_acts_of_pizza`. Note that we convert the task ID (also referred to as "competition ID") from kebab to snake case. 

## Design
The automated "science loop" consists of a few common stages:

**Ideation:** Generating new ideas for hypotheses to test and implementation changes to try.

**Experiment implementation:** Coding the experiments that test the ideas produced in the ideation stage.

**Experiment execution:** Running the code that implements the experiments.

**Results analysis:** Extracting insights from the output of the executed experiments.

This codebase is designed with the following goals in mind:
- Quick iteration speed
- Minimal external dependencies
- Allows easy, reproducible evaluation of each science loop stage, both in isolation and as part of a complete scientist loop.
- Allows mixing and matching different strategies for each science loop stage.
- Clearly tracks all artifacts produced by the scientist at each stage (see workspaces design below)
- Plays nicely with Slurm


### The core science loop
The science loop can be implemented via either of two existing runners (or with any custom logic via your own subclass of `ScienceRunner`): 
- `ScienceRunner` is unstructured in its run logic, allowing for maximum flexibility.
- `BoNScienceRunner` inherits from `ScienceRunner`, and follows a set, structured sequence of steps corresponding to the common stages above, with freedom around how often they are each run and parallelized per iteration. In particular, a batch `n_hypotheses` hypotheses are generated in a single iteration, and a `selection_metric` can be defined via the `ExperimentConfig` passed into the runner in order to select the best hypothesis found so far. This hypothesis is then used as the starting point for the the next iteration of the science loop.

In the `BoNScienceRunner`, ideation and implementation are handled by instances of the `Ideator` and `Coder` classes respectively, which all subclass `Agent` (see the Agent section below). The modules `core.ideators` and `core.coders` serve as central registries for Ideator and Coder subclasses, making it easy to define combinations of ideator and coder strategies, which can all be set with a single line in the top-level hydra config. Moreover, the `BoNScienceRunner` also receives a simple instance of `Agent` (under the `assistant` property), which is used for handling one-off LLM queries. 

#### Why explict modules?
Having explicit implementations for `Ideator` and `Coder` variants is useful, as these strategies will be valuable to run in isolation, either for the purposes of evaluation (per-stage evals) or for handling independent endpoints in downstream integrations (e.g. giving Metamate an "ideation" or "experiment implementation" skill).


### Agents
Each system-prompted LLM instance is abstracted as an agent, with an `act` method, which takes a prompt (e.g. instruction) and optionally a _validator function_ and a value for `max_retries`. The validator function returns a string value, that can be an arbitrary post-processed version of the LLM response to the prompt, and it should return `None` to mark the response as invalid. The method `act` will then retry querying the LLM with the prompt a maximum of `max_retries` times.


### Versioned workspaces
Each scientist run is encapsulated in its own subdirectory inside the `workspaces` directory (auto-generated on first run). 

Each experiment implemented by the scientist during a run corresponds to a _version_ of an initial _workspace template_, a directory of files and potentially nested subdirectories. Workspace templates provides the initial project contents that serve as a starting point for the scientist to begin its experimentation. 

When the workspace is first created, `v_1` (version 1) is initialized by copying the specified workspace template. Each experiment iteration corresponds to branching (i.e. copying) the previous version into a new version (e.g. `v_1 -> v_2`) and making changes in the new version directory. In this way, workspaces track the full history over arbitrary trees of codebases that may be created during the course of a scientist run.


### JobObserver
A core object in this framework is the `JobObserver`, in particular, its shared, singleton instance, `JobObserver.shared`. This object can be used to _observe_ the status of any Slurm job created via `submitit`, as encapsulated by its corresponding `Job` instance. 

The below function call will ask the JobObserver singleton to start watching `job`, executing the `callback` function when the job finishes running (either on successful completion or some other form of termination).

```
slurm_utils.JobObserver.shared.observe(
	job=job,
	metadata={'hypothesis': hypothesis},  # Any serializable dictionary
	callback=lambda res: print(res)
)
```

Importantly, if you want to block thread execution until all observed jobs are finished, you should perform the following call:

`await slurm_utils.JobObserver.shared.wait()`

Under the hood, JobObserver manages a set of `asyncio` tasks that regularly polls for the status of each observed job. This means the main function for each scientist script must be run as `asyncio.run(main())`, so that the asyncio run loop is properly initialized (see `climb_nanogpt.py` or `climb_collatz.py` for examples).





