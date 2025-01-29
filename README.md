# 🧪 Meta AI Scientist

This is an early proof-of-concept of an automated scientist agent, with a focus on running ML experiments.

In the current framework, an LLM-based scientist agent (or team of such agents) repeatedly generates hypotheses, implements these hypotheses in code, executes these experiments as Slurm jobs, and summarizes the results of these jobs for the next iteration.

## 📋 Todos
- [ ] Create a single high-level `launch_scientist.py` integrated with `hydra` for config. 
- [ ] Refactor existing scientist scripts (`climb_nanogpt.py` and `climb_collatz.py`) to be configurations of a single structured `ScienceRunner` class
	- [ ] Break out hypothesis generation and hypothesis implementation logic into simple instances of `Ideator` and `Implementer`.
- [ ] Enable `ScienceRunner` to run multiple experiments per iteration in parallel
- [ ] Add support diff-based editors (e.g. via Aider-based Implementers)
- [ ] Support MetaGen and third-party LLM APIs in `core.llm_client`.

## Run examples

First, spin up an instance of r1 (32B):

```
python serve_vllm.py
```

Find the node id for this vllm job on Slurm and run one of the scientist scripts:
```
python climb_collatz.py <vllm node id>
```


## Design
**This is an early iteration of the codebase and major architectural decisions may be subject to change**

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
To maximize flexibility and speed early on, the scientist experimentation loop for each task, e.g. speedrunning nanoGPT, is implemented in its own script, e.g. `climb_nanogpt.py`.

The core science loop in each script is implemented via a subclass of ScienceRunner (e.g. `NanoGPTClimber`) implements a `run(n_iterations: int)` method, which executes the scientist loop several times.

While the core run logic can currently be free-form, we plan to fork ScienceRunner into two variations: 
- `BaseScienceRunner` will remain unstructured in its run logic
- `ScienceRunner`, which inherits from `BaseScienceRunner`, will follow a structured sequence of steps corresponding to the common stages above, with some freedom around how often they are each run and parallelized per iteration.

In particular, one possible design is to abstract ideation and implementation strategies under subclasses of general `Ideator` and `Implementer` classes. The modules `core.ideators` and `core.implementers` can then act as central registries for Ideator and Implementer subclasses, making it easy to define combinations of ideator and implementer strategies in the hydra config.

#### Why explict modules?
Having explicit implementations for `Ideator` and `Implementer` variants is useful, as these strategies will be valuable to run in isolation, either for the purposes of evaluation (per-stage evals) or for handling independent endpoints in downstream integrations (e.g. giving Metamate an "ideation" or "experiment implementation" skill).


### Agents
Each system-prompted LLM instance is abstracted as an agent, with an `act` method, which takes a prompt (e.g. instruction) and optionally a _validator function_ and a value for `max_retries`. The validator function returns a string value, that can be an arbitrary post-processed version of the LLM response to the prompt, and it should return `None` to mark the response as invalid. The method `act` will then retry querying the LLM with the prompt a maximum of `max_retries` times.

In particular, it would make sense to implement `Ideator` and `Implementer` as subclasses of `Agent`.


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







