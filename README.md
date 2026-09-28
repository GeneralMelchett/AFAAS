# AFAAS

AFAAS (“Alex Ferguson as a Solution”) is a small experimental harness for
studying how language-model responses vary under controlled conditions. The
reusable experimental components live in the YAML catalogues under
`config/definitions/`.

## Run the interactive wizard

Install the locked dependencies with:

```sh
uv sync
```

Then start the normal human interface:

```sh
python run_experiment.py
```

The Questionary wizard loads its choices dynamically from the catalogues. It
selects a persona identity first and then offers only framings belonging to
that identity. The `none` persona is the no-persona control. Situations and
intensities are optional; an empty catalogue is represented by `none` rather
than an unusable empty menu.

The wizard displays a complete summary before asking for confirmation. Until
that confirmation, no result directory is created and no API call is made.
Ctrl-C or cancellation before confirmation has the same side-effect-free
behaviour.

## Reproducible command-line use

The same internal selection and execution path can be driven non-
interactively:

```sh
python run_experiment.py \
  --instruction motivational-advice-v1 \
  --prompt garage-procrastination-v1 \
  --runs 10
```

The command above is the no-persona baseline. A Julia condition is:

```sh
python run_experiment.py \
  --persona julia \
  --framing default-v1 \
  --instruction motivational-advice-v1 \
  --prompt garage-procrastination-v1 \
  --runs 10
```

Optional settings can be supplied with `--situation`, `--intensity`,
`--model`, `--reasoning-effort`, and `--verbosity`. Use the literal value
`none` for an explicit optional-control selection. If only some required
values are supplied from a terminal, the runner asks for the missing values;
unknown labels are rejected rather than guessed.

Discover valid labels without opening the YAML files:

```sh
python run_experiment.py --list-all
python run_experiment.py --list-personas
python run_experiment.py --list-instructions
python run_experiment.py --list-prompts
python run_experiment.py --list-situations
python run_experiment.py --list-intensities
```

## Results and provenance

The runner allocates the next unused identifier from the existing results
directory and never overwrites an existing experiment directory. Results
have this shape:

```text
results/
└── EXP-NNN/
    ├── experiment.yaml
    ├── results.jsonl
    └── results.txt
```

`experiment.yaml` is now generated provenance, not user-authored input. It
contains the selected catalogue labels, generation settings, and the resolved
request specification. `results.jsonl` remains the canonical machine-readable
record set, while `results.txt` is a convenient human-readable rendering.

Each successful run records:

1. the catalogue labels selected;
2. the exact catalogue text resolved at execution time;
3. the exact rendered `instructions` and `input` sent to the Responses API;
4. the exact response text received.

One run is one API request. Run IDs are associated with the experiment, for
example `EXP-003-001` through `EXP-003-010`. Successful records are persisted
immediately. The runner does not silently retry failed API calls or continue
to later runs after an exception; the result directory and all earlier
successful records are preserved so the failure remains visible.
