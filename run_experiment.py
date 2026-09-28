#!/usr/bin/env python3

"""Interactive and reproducible runner for AFAAS experiments.

The YAML catalogues are the source of selectable experimental components.
The runner resolves those selections immediately before execution and records
both the selected labels and the exact rendered request sent to the API.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

import questionary
import yaml
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent
DEFINITIONS_DIR = PROJECT_ROOT / "config" / "definitions"
RESULTS_DIR = PROJECT_ROOT / "results"

LOCAL_TIMEZONE = ZoneInfo("Europe/London")
DEFAULT_MODEL = "gpt-6-luna"
EXPERIMENT_ID_PATTERN = re.compile(r"^EXP-(\d+)$")

CATALOGUE_FILES = {
    "personas": "personas.yaml",
    "instructions": "instructions.yaml",
    "prompts": "prompts.yaml",
    "situations": "situations.yaml",
    "intensities": "intensities.yaml",
    "factors": "factors.yaml",
}

OPTIONAL_CATALOGUES = {"situations", "intensities", "factors"}
REASONING_EFFORTS = ("none", "low", "medium", "high")
VERBOSITIES = ("none", "low", "medium", "high")


class WizardCancelled(Exception):
    """The user cancelled an interactive selection."""


@dataclass(frozen=True)
class ExperimentSpecification:
    """The catalogue labels and generation settings selected by the user."""

    persona: dict[str, str] | None
    instruction: str
    prompt: str
    situation: str | None
    intensity: str | None
    runs: int
    model: str
    reasoning_effort: str | None
    verbosity: str | None

    def as_input(self, experiment_id: str) -> dict[str, Any]:
        """Return the generated-input shape used throughout the runner."""

        return {
            "experiment": {
                "id": experiment_id,
                "runs": self.runs,
                "persona": self.persona,
                "instruction": self.instruction,
                "prompt": self.prompt,
                "situation": self.situation,
                "intensity": self.intensity,
            },
            "generation": {
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "verbosity": self.verbosity,
            },
        }


# ---------------------------------------------------------------------------
# YAML and catalogue loading
# ---------------------------------------------------------------------------


def load_yaml(path: Path, *, optional: bool = False) -> dict[str, Any]:
    """Load a YAML mapping, optionally treating a missing file as empty."""

    if not path.exists():
        if optional:
            return {}
        raise FileNotFoundError(f"Catalogue file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")

    return data


def normalise_catalogue(name: str, data: dict[str, Any]) -> dict[str, Any]:
    """Accept direct catalogues and empty named catalogue mappings."""

    # Empty catalogues currently use ``situations: {}`` and ``factors: {}``,
    # while populated catalogues use their labels at the top level.
    if set(data) == {name} and isinstance(data[name], dict):
        return data[name]
    return data


def load_catalogues(
    definitions_dir: Path = DEFINITIONS_DIR,
) -> dict[str, dict[str, Any]]:
    """Load all definition catalogues from the configured definitions path."""

    catalogues: dict[str, dict[str, Any]] = {}
    for name, filename in CATALOGUE_FILES.items():
        data = load_yaml(
            definitions_dir / filename,
            optional=name in OPTIONAL_CATALOGUES,
        )
        catalogues[name] = normalise_catalogue(name, data)
    return catalogues


def require_entry(
    catalogue: Mapping[str, Any],
    label: str,
    catalogue_name: str,
) -> dict[str, Any]:
    """Return one catalogue entry with a useful validation error."""

    try:
        entry = catalogue[label]
    except KeyError:
        raise ValueError(
            f"Unknown {catalogue_name} label: {label!r}."
        ) from None

    if not isinstance(entry, dict):
        raise ValueError(
            f"{catalogue_name} entry {label!r} must be a mapping."
        )
    return entry


def clean_description(entry: Mapping[str, Any]) -> str | None:
    description = entry.get("description")
    if description is None:
        return None
    return " ".join(str(description).split()) or None


def choice_title(label: str, entry: Mapping[str, Any]) -> str:
    description = clean_description(entry)
    return f"{label} — {description}" if description else label


# ---------------------------------------------------------------------------
# Resolution and request construction
# ---------------------------------------------------------------------------


def text_value(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} text must be a string or null.")
    return value


def resolve_persona(
    experiment: Mapping[str, Any],
    personas: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Resolve an identity and only its own selected framing."""

    selection = experiment.get("persona")
    if selection is None:
        return None
    if not isinstance(selection, dict):
        raise ValueError("experiment.persona must be a mapping or null.")

    identity_label = selection.get("identity")
    framing_label = selection.get("framing")
    if not isinstance(identity_label, str):
        raise ValueError("experiment.persona.identity must be a string.")

    # ``none`` is a control condition rather than a real persona.
    if identity_label == "none":
        return None

    if not isinstance(framing_label, str) or not framing_label:
        raise ValueError(
            f"Persona {identity_label!r} requires a framing belonging to it."
        )

    identity = require_entry(personas, identity_label, "persona identity")
    framings = identity.get("framings", {})
    if not isinstance(framings, dict):
        raise ValueError(
            f"Persona {identity_label!r} framings must be a mapping."
        )
    if framing_label not in framings:
        raise ValueError(
            f"Persona {identity_label!r} has no framing {framing_label!r}."
        )

    framing = framings[framing_label]
    if not isinstance(framing, dict):
        raise ValueError(
            f"Framing {framing_label!r} for persona {identity_label!r} "
            "must be a mapping."
        )

    identity_definition = identity.get("identity", {})
    if not isinstance(identity_definition, dict):
        raise ValueError(
            f"Persona {identity_label!r} identity must be a mapping."
        )

    identity_text = text_value(
        identity_definition.get("text"),
        context=f"Persona {identity_label!r} identity",
    )
    framing_text = text_value(
        framing.get("text"),
        context=f"Framing {framing_label!r}",
    )
    parts = [
        text.strip()
        for text in (identity_text, framing_text)
        if text and text.strip()
    ]

    return {
        "identity": identity_label,
        "framing": framing_label,
        "text": "\n\n".join(parts) if parts else None,
    }


def resolve_component(
    label: str | None,
    catalogue: Mapping[str, Any],
    catalogue_name: str,
) -> dict[str, Any] | None:
    if label is None or label == "none":
        return None

    entry = require_entry(catalogue, label, catalogue_name)
    return {
        "label": label,
        "text": text_value(
            entry.get("text"),
            context=f"{catalogue_name.title()} {label!r}",
        ),
    }


def combine_text(*parts: str | None) -> str:
    return "\n\n".join(
        part.strip()
        for part in parts
        if part and part.strip()
    )


def build_request(
    experiment: Mapping[str, Any],
    catalogues: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve labels and construct the exact API request fields."""

    persona = resolve_persona(experiment, catalogues["personas"])
    instruction = resolve_component(
        experiment.get("instruction"),
        catalogues["instructions"],
        "instruction",
    )
    prompt = resolve_component(
        experiment.get("prompt"),
        catalogues["prompts"],
        "prompt",
    )
    situation = resolve_component(
        experiment.get("situation"),
        catalogues["situations"],
        "situation",
    )
    intensity = resolve_component(
        experiment.get("intensity"),
        catalogues["intensities"],
        "intensity",
    )

    instructions_text = combine_text(
        persona["text"] if persona else None,
        instruction["text"] if instruction else None,
        intensity["text"] if intensity else None,
    )
    input_text = combine_text(
        situation["text"] if situation else None,
        prompt["text"] if prompt else None,
    )

    return {
        "persona": persona,
        "instruction": instruction,
        "prompt": prompt,
        "situation": situation,
        "intensity": intensity,
        "rendered": {
            "instructions": instructions_text,
            "input": input_text,
        },
    }


# ---------------------------------------------------------------------------
# Selection and validation
# ---------------------------------------------------------------------------


def normalise_optional_selection(value: str | None) -> str | None:
    return None if value is None or value == "none" else value


def validate_specification(
    specification: ExperimentSpecification,
    catalogues: Mapping[str, Mapping[str, Any]],
) -> None:
    if not specification.instruction:
        raise ValueError("An instruction must be selected.")
    if not specification.prompt:
        raise ValueError("A prompt must be selected.")
    if specification.runs < 1:
        raise ValueError("Runs must be a positive integer.")
    if not specification.model.strip():
        raise ValueError("Model must not be empty.")

    require_entry(
        catalogues["instructions"],
        specification.instruction,
        "instruction",
    )
    require_entry(
        catalogues["prompts"],
        specification.prompt,
        "prompt",
    )

    if specification.persona is not None:
        identity = specification.persona.get("identity")
        framing = specification.persona.get("framing")
        if not identity or identity == "none":
            raise ValueError("A real persona identity or no persona must be selected.")
        if not framing:
            raise ValueError(f"Persona {identity!r} requires a framing.")
        resolve_persona(
            {"persona": specification.persona},
            catalogues["personas"],
        )

    for label, catalogue_name, display_name in (
        (specification.situation, "situations", "situation"),
        (specification.intensity, "intensities", "intensity"),
    ):
        if label is not None:
            require_entry(catalogues[catalogue_name], label, display_name)


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def questionary_positive_integer(value: str) -> bool | str:
    try:
        if int(value) < 1:
            return "Enter a positive integer."
    except (TypeError, ValueError):
        return "Enter a positive integer."
    return True


def ask_question(ask: Callable[[], Any]) -> Any:
    try:
        value = ask()
    except (KeyboardInterrupt, EOFError) as error:
        raise WizardCancelled from error
    if value is None:
        raise WizardCancelled
    return value


def select_from_catalogue(
    message: str,
    catalogue: Mapping[str, Any],
    *,
    include_none: bool = False,
    none_title: str = "none — no component",
) -> str | None:
    choices: list[Any] = []
    if include_none:
        choices.append(questionary.Choice(none_title, value="none"))
    choices.extend(
        questionary.Choice(choice_title(label, entry), value=label)
        for label, entry in catalogue.items()
        if isinstance(entry, dict)
    )
    if not choices:
        return None

    value = ask_question(
        lambda: questionary.select(message, choices=choices).ask()
    )
    return normalise_optional_selection(value)


def select_framing(
    identity: str,
    catalogues: Mapping[str, Mapping[str, Any]],
) -> str:
    persona = require_entry(
        catalogues["personas"],
        identity,
        "persona identity",
    )
    framings = persona.get("framings", {})
    if not isinstance(framings, dict) or not framings:
        raise ValueError(f"Persona {identity!r} has no usable framings.")

    choices = [
        questionary.Choice(choice_title(label, entry), value=label)
        for label, entry in framings.items()
        if isinstance(entry, dict)
    ]
    if not choices:
        raise ValueError(f"Persona {identity!r} has no usable framings.")

    return ask_question(
        lambda: questionary.select(
            f"Select framing for {identity}",
            choices=choices,
        ).ask()
    )


def ask_runs() -> int:
    value = ask_question(
        lambda: questionary.text(
            "Number of runs",
            default="1",
            validate=questionary_positive_integer,
        ).ask()
    )
    return int(value)


def ask_generation_setting(
    message: str,
    values: tuple[str, ...],
) -> str | None:
    choices = [
        questionary.Choice(
            "none — omit this API setting" if value == "none" else value,
            value=value,
        )
        for value in values
    ]
    value = ask_question(
        lambda: questionary.select(message, choices=choices).ask()
    )
    return None if value == "none" else value


def run_full_wizard(
    catalogues: Mapping[str, Mapping[str, Any]],
) -> ExperimentSpecification:
    persona_identity = select_from_catalogue(
        "Select persona",
        catalogues["personas"],
        include_none=True,
        none_title="none — no persona (control)",
    )

    persona: dict[str, str] | None = None
    if persona_identity is not None:
        framing = select_framing(persona_identity, catalogues)
        persona = {"identity": persona_identity, "framing": framing}

    instruction = select_from_catalogue(
        "Select instruction",
        catalogues["instructions"],
    )
    prompt = select_from_catalogue(
        "Select prompt",
        catalogues["prompts"],
    )
    situation = select_from_catalogue(
        "Select situation",
        catalogues["situations"],
        include_none=True,
    )
    intensity = select_from_catalogue(
        "Select intensity",
        catalogues["intensities"],
        include_none=True,
    )

    if instruction is None or prompt is None:
        raise ValueError("Instruction and prompt catalogues must not be empty.")

    model = ask_question(
        lambda: questionary.text("Model", default=DEFAULT_MODEL).ask()
    )

    return ExperimentSpecification(
        persona=persona,
        instruction=instruction,
        prompt=prompt,
        situation=situation,
        intensity=intensity,
        runs=ask_runs(),
        model=model,
        reasoning_effort=ask_generation_setting(
            "Reasoning effort",
            REASONING_EFFORTS,
        ),
        verbosity=ask_generation_setting("Verbosity", VERBOSITIES),
    )


def collect_specification(
    args: argparse.Namespace,
    catalogues: Mapping[str, Mapping[str, Any]],
) -> tuple[ExperimentSpecification, bool]:
    """Combine CLI values with either the full or partial wizard."""

    supplied_values = (
        args.persona,
        args.framing,
        args.instruction,
        args.prompt,
        args.situation,
        args.intensity,
        args.runs,
        args.model,
        args.reasoning_effort,
        args.verbosity,
    )
    full_wizard = not any(value is not None for value in supplied_values)

    if full_wizard:
        if not sys.stdin.isatty():
            raise ValueError(
                "The interactive wizard requires a terminal. Supply the "
                "required selections on the command line instead."
            )
        return run_full_wizard(catalogues), True

    if args.framing is not None and args.persona is None:
        raise ValueError("--framing requires --persona.")

    if args.persona is None or args.persona == "none":
        if args.framing is not None:
            raise ValueError("--framing cannot be used with --persona none.")
        persona: dict[str, str] | None = None
    else:
        persona = {
            "identity": args.persona,
            "framing": args.framing,
        }

    missing: list[str] = []
    if args.instruction is None:
        missing.append("--instruction")
    if args.prompt is None:
        missing.append("--prompt")
    if args.runs is None:
        missing.append("--runs")
    if persona is not None and args.framing is None:
        missing.append("--framing")

    if missing:
        if not sys.stdin.isatty():
            raise ValueError(
                "Missing required selections: "
                + ", ".join(missing)
                + ". Re-run with those options or from an interactive terminal."
            )

        if persona is not None and args.framing is None:
            persona["framing"] = select_framing(args.persona, catalogues)
        instruction = args.instruction or select_from_catalogue(
            "Select instruction",
            catalogues["instructions"],
        )
        prompt = args.prompt or select_from_catalogue(
            "Select prompt",
            catalogues["prompts"],
        )
        runs = args.runs if args.runs is not None else ask_runs()
        if instruction is None or prompt is None:
            raise ValueError("Instruction and prompt catalogues must not be empty.")

        return ExperimentSpecification(
            persona=persona,
            instruction=instruction,
            prompt=prompt,
            situation=normalise_optional_selection(args.situation),
            intensity=normalise_optional_selection(args.intensity),
            runs=runs,
            model=args.model or DEFAULT_MODEL,
            reasoning_effort=normalise_optional_selection(args.reasoning_effort),
            verbosity=normalise_optional_selection(args.verbosity),
        ), True

    return ExperimentSpecification(
        persona=persona,
        instruction=args.instruction,
        prompt=args.prompt,
        situation=normalise_optional_selection(args.situation),
        intensity=normalise_optional_selection(args.intensity),
        runs=args.runs,
        model=args.model or DEFAULT_MODEL,
        reasoning_effort=normalise_optional_selection(args.reasoning_effort),
        verbosity=normalise_optional_selection(args.verbosity),
    ), False


# ---------------------------------------------------------------------------
# Experiment identifiers and output
# ---------------------------------------------------------------------------


def next_experiment_number(results_dir: Path = RESULTS_DIR) -> int:
    highest = 0
    if not results_dir.exists():
        return 1
    for child in results_dir.iterdir():
        match = EXPERIMENT_ID_PATTERN.fullmatch(child.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def preview_next_experiment_id(results_dir: Path = RESULTS_DIR) -> str:
    return f"EXP-{next_experiment_number(results_dir):03d}"


def allocate_experiment_directory(
    results_dir: Path = RESULTS_DIR,
) -> tuple[str, Path]:
    """Atomically claim a never-used experiment directory where possible."""

    results_dir.mkdir(parents=True, exist_ok=True)
    number = next_experiment_number(results_dir)
    while True:
        experiment_id = f"EXP-{number:03d}"
        output_dir = results_dir / experiment_id
        try:
            output_dir.mkdir()
        except FileExistsError:
            number += 1
            continue
        return experiment_id, output_dir


def now_iso() -> str:
    return datetime.now(LOCAL_TIMEZONE).isoformat(timespec="milliseconds")


def write_experiment_definition(
    path: Path,
    experiment: Mapping[str, Any],
    request: Mapping[str, Any],
) -> None:
    """Write generated provenance, including the resolved request text."""

    payload = {
        "experiment": experiment["experiment"],
        "generation": experiment["generation"],
        "resolved_request": request,
    }
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            payload,
            file,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def display_component(
    heading: str,
    component: Mapping[str, Any] | None,
) -> list[str]:
    lines = [heading, ""]
    if component is None:
        lines.extend(["[not used]", ""])
        return lines

    if "identity" in component:
        lines.append(f"Identity: {component['identity']}")
        lines.append(f"Framing: {component['framing']}")
    else:
        lines.append(f"Label: {component['label']}")
    lines.append("")
    lines.append(component.get("text") or "[no text]")
    lines.append("")
    return lines


def append_text_result(path: Path, record: Mapping[str, Any]) -> None:
    separator = "=" * 68
    lines = [
        separator,
        f"RUN {record['run']['id']}",
        separator,
        "",
        f"Started:  {record['run']['started_at']}",
        f"Finished: {record['run']['finished_at']}",
        "",
        f"Model: {record['model']['name']}",
        f"Reasoning effort: {record['model'].get('reasoning_effort')}",
        f"Verbosity: {record['model'].get('verbosity')}",
        "",
    ]

    request = record["request"]
    lines.extend(display_component("PERSONA", request["persona"]))
    lines.extend(display_component("INSTRUCTION", request["instruction"]))
    lines.extend(display_component("PROMPT", request["prompt"]))
    lines.extend(display_component("SITUATION", request["situation"]))
    lines.extend(display_component("INTENSITY", request["intensity"]))
    lines.extend([
        "EXACT RENDERED INSTRUCTIONS",
        "",
        request["rendered"]["instructions"] or "[empty]",
        "",
        "EXACT RENDERED INPUT",
        "",
        request["rendered"]["input"] or "[empty]",
        "",
        "RESPONSE",
        "",
        record["response"]["text"],
        "",
        "",
    ])

    with path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))


# ---------------------------------------------------------------------------
# OpenAI execution
# ---------------------------------------------------------------------------


def generate(
    client: OpenAI,
    request: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> str:
    """Make exactly one Responses API request for one experimental run."""

    kwargs: dict[str, Any] = {
        "model": generation["model"],
        "instructions": request["rendered"]["instructions"],
        "input": request["rendered"]["input"],
    }
    reasoning_effort = generation.get("reasoning_effort")
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    verbosity = generation.get("verbosity")
    if verbosity is not None:
        kwargs["text"] = {"verbosity": verbosity}

    response = client.responses.create(**kwargs)
    return response.output_text


def execute_experiment(
    specification: ExperimentSpecification,
    catalogues: Mapping[str, Mapping[str, Any]],
    *,
    results_dir: Path = RESULTS_DIR,
    client_factory: Callable[[], OpenAI] = OpenAI,
) -> Path:
    """Allocate output, persist provenance, then execute runs sequentially."""

    validate_specification(specification, catalogues)
    experiment_id, output_dir = allocate_experiment_directory(results_dir)
    experiment = specification.as_input(experiment_id)
    request = build_request(experiment["experiment"], catalogues)

    # This is apparatus-generated provenance, not user-authored input. It is
    # written before the first API call so a failure leaves the attempted
    # condition available for inspection.
    write_experiment_definition(
        output_dir / "experiment.yaml",
        experiment,
        request,
    )

    jsonl_path = output_dir / "results.jsonl"
    text_path = output_dir / "results.txt"
    generation = experiment["generation"]

    print()
    print(f"AFAAS experiment: {experiment_id}")
    print(f"Runs:             {specification.runs}")
    print(f"Model:            {specification.model}")
    print(f"Results:          {output_dir}")
    print()

    client = client_factory()
    for number in range(1, specification.runs + 1):
        run_id = f"{experiment_id}-{number:03d}"
        print(
            f"[{number:02d}/{specification.runs:02d}] {run_id} ... ",
            end="",
            flush=True,
        )
        started_at = now_iso()
        try:
            response_text = generate(client, request, generation)
        except Exception as error:
            print("FAILED", file=sys.stderr)
            print(
                f"Run {run_id} failed; successful earlier runs were preserved.",
                file=sys.stderr,
            )
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
            raise

        finished_at = now_iso()
        record = {
            "run": {
                "id": run_id,
                "started_at": started_at,
                "finished_at": finished_at,
            },
            "request": request,
            "model": {
                "name": generation["model"],
                "reasoning_effort": generation.get("reasoning_effort"),
                "verbosity": generation.get("verbosity"),
            },
            "response": {"text": response_text},
        }

        # Persist every successful generation before proceeding to the next
        # API call. There is intentionally no retry path here.
        with jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        append_text_result(text_path, record)
        print("done")

    print()
    print(f"Experiment {experiment_id} complete.")
    print(f"JSONL: {jsonl_path}")
    print(f"Text:  {text_path}")
    return output_dir


# ---------------------------------------------------------------------------
# CLI and presentation
# ---------------------------------------------------------------------------


def print_summary(
    specification: ExperimentSpecification,
    experiment_id: str,
) -> None:
    persona = specification.persona
    print()
    print("AFAAS Experiment")
    print("----------------------------------------")
    print(f"Experiment     {experiment_id}")
    print(f"Persona        {persona['identity'] if persona else 'none'}")
    print(f"Framing        {persona['framing'] if persona else 'none'}")
    print(f"Instruction    {specification.instruction}")
    print(f"Prompt         {specification.prompt}")
    print(f"Situation      {specification.situation or 'none'}")
    print(f"Intensity      {specification.intensity or 'none'}")
    print(f"Runs           {specification.runs}")
    print(f"Model          {specification.model}")
    print(f"Reasoning      {specification.reasoning_effort or 'none'}")
    print(f"Verbosity      {specification.verbosity or 'none'}")
    print("----------------------------------------")


def print_catalogue(
    title: str,
    catalogue: Mapping[str, Any],
    *,
    show_framings: bool = False,
) -> None:
    print(title)
    if not catalogue:
        print("  (empty — the selectable value is none)")
        return
    for label, entry in catalogue.items():
        if not isinstance(entry, dict):
            continue
        print(f"  {choice_title(label, entry)}")
        if show_framings:
            framings = entry.get("framings", {})
            if isinstance(framings, dict):
                for framing, framing_entry in framings.items():
                    if isinstance(framing_entry, dict):
                        print(f"    {choice_title(framing, framing_entry)}")


def print_requested_catalogues(
    args: argparse.Namespace,
    catalogues: Mapping[str, Mapping[str, Any]],
) -> bool:
    requested = False
    if args.list_all or args.list_personas:
        print_catalogue("Personas", catalogues["personas"], show_framings=True)
        requested = True
    if args.list_all or args.list_instructions:
        print_catalogue("Instructions", catalogues["instructions"])
        requested = True
    if args.list_all or args.list_prompts:
        print_catalogue("Prompts", catalogues["prompts"])
        requested = True
    if args.list_all or args.list_situations:
        print_catalogue("Situations", catalogues["situations"])
        requested = True
    if args.list_all or args.list_intensities:
        print_catalogue("Intensities", catalogues["intensities"])
        requested = True
    return requested


def build_parser() -> argparse.ArgumentParser:
    examples = """Examples:
  python run_experiment.py
  python run_experiment.py --instruction motivational-advice-v1 \\
      --prompt garage-procrastination-v1 --runs 10
  python run_experiment.py --persona julia --framing default-v1 \\
      --instruction motivational-advice-v1 \\
      --prompt garage-procrastination-v1 --runs 10
  python run_experiment.py --list-all"""

    parser = argparse.ArgumentParser(
        description=(
            "Run an AFAAS experiment from the YAML definition catalogues. "
            "With no selections, starts the interactive Questionary wizard."
        ),
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--persona", help="Persona identity label; omit for the no-persona control.")
    parser.add_argument("--framing", help="Framing label belonging to --persona.")
    parser.add_argument("--instruction", help="Instruction catalogue label.")
    parser.add_argument("--prompt", help="Prompt catalogue label.")
    parser.add_argument("--situation", help="Situation label, or none.")
    parser.add_argument("--intensity", help="Intensity label, or none.")
    parser.add_argument("--runs", type=positive_integer, help="Number of API generations.")
    parser.add_argument("--model", help=f"OpenAI model (default: {DEFAULT_MODEL}).")
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        help="Responses API reasoning effort; none omits the setting.",
    )
    parser.add_argument(
        "--verbosity",
        choices=VERBOSITIES,
        help="Responses API text verbosity; none omits the setting.",
    )
    parser.add_argument("--list-all", action="store_true", help="List every catalogue and exit.")
    parser.add_argument("--list-personas", action="store_true", help="List persona identities and their framings.")
    parser.add_argument("--list-instructions", action="store_true", help="List instruction labels and descriptions.")
    parser.add_argument("--list-prompts", action="store_true", help="List prompt labels and descriptions.")
    parser.add_argument("--list-situations", action="store_true", help="List situation labels and descriptions.")
    parser.add_argument("--list-intensities", action="store_true", help="List intensity labels and descriptions.")
    return parser


def main() -> int:
    parser = build_parser()
    try:
        args = parser.parse_args()
        catalogues = load_catalogues()
        if print_requested_catalogues(args, catalogues):
            return 0

        specification, used_wizard = collect_specification(args, catalogues)
        validate_specification(specification, catalogues)
        print_summary(specification, preview_next_experiment_id())

        if used_wizard:
            confirmed = ask_question(
                lambda: questionary.confirm(
                    "Execute this experiment?",
                    default=False,
                ).ask()
            )
            if not confirmed:
                print("Cancelled. No experiment directory was created and no API call was made.")
                return 0

        execute_experiment(specification, catalogues)
        return 0
    except WizardCancelled:
        print("\nCancelled. No experiment directory was created and no API call was made.")
        return 0
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("\nInterrupted. Any successful runs already written were preserved.", file=sys.stderr)
        return 130

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
