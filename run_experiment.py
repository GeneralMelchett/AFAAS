#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent
DEFINITIONS_DIR = PROJECT_ROOT / "definitions"
RESULTS_DIR = PROJECT_ROOT / "results"

LOCAL_TIMEZONE = ZoneInfo("Europe/London")


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")

    return data


def load_catalogues() -> dict[str, dict[str, Any]]:
    return {
        "personas": load_yaml(DEFINITIONS_DIR / "personas.yaml"),
        "instructions": load_yaml(DEFINITIONS_DIR / "instructions.yaml"),
        "prompts": load_yaml(DEFINITIONS_DIR / "prompts.yaml"),
        "situations": load_yaml(DEFINITIONS_DIR / "situations.yaml"),
        "intensities": load_yaml(DEFINITIONS_DIR / "intensities.yaml"),
        "factors": load_yaml(DEFINITIONS_DIR / "factors.yaml"),
    }


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def require_entry(
    catalogue: dict[str, Any],
    label: str,
    catalogue_name: str,
) -> dict[str, Any]:
    try:
        entry = catalogue[label]
    except KeyError:
        raise ValueError(
            f"Unknown {catalogue_name} label: {label!r}"
        ) from None

    if not isinstance(entry, dict):
        raise ValueError(
            f"{catalogue_name} entry {label!r} must be a mapping."
        )

    return entry


def resolve_persona(
    experiment: dict[str, Any],
    personas: dict[str, Any],
) -> dict[str, Any] | None:

    selection = experiment.get("persona")

    if selection is None:
        return None

    identity_label = selection["identity"]
    framing_label = selection["framing"]

    identity = require_entry(
        personas,
        identity_label,
        "persona identity",
    )

    framings = identity.get("framings", {})

    if framing_label not in framings:
        raise ValueError(
            f"Persona {identity_label!r} has no framing "
            f"{framing_label!r}."
        )

    framing = framings[framing_label]

    identity_text = identity.get("identity", {}).get("text")
    framing_text = framing.get("text")

    parts = [
        text.strip()
        for text in (identity_text, framing_text)
        if text
    ]

    resolved_text = "\n\n".join(parts) if parts else None

    return {
        "identity": identity_label,
        "framing": framing_label,
        "text": resolved_text,
    }


def resolve_component(
    label: str | None,
    catalogue: dict[str, Any],
    catalogue_name: str,
) -> dict[str, Any] | None:

    if label is None:
        return None

    entry = require_entry(catalogue, label, catalogue_name)

    return {
        "label": label,
        "text": entry.get("text"),
    }


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------

def combine_text(*parts: str | None) -> str:
    return "\n\n".join(
        part.strip()
        for part in parts
        if part and part.strip()
    )


def build_request(
    experiment: dict[str, Any],
    catalogues: dict[str, dict[str, Any]],
) -> dict[str, Any]:

    persona = resolve_persona(
        experiment,
        catalogues["personas"],
    )

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
# Time
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(LOCAL_TIMEZONE).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def generate(
    client: OpenAI,
    request: dict[str, Any],
    generation: dict[str, Any],
) -> str:

    kwargs: dict[str, Any] = {
        "model": generation["model"],
        "instructions": request["rendered"]["instructions"],
        "input": request["rendered"]["input"],
    }

    reasoning_effort = generation.get("reasoning_effort")

    if reasoning_effort is not None:
        kwargs["reasoning"] = {
            "effort": reasoning_effort,
        }

    verbosity = generation.get("verbosity")

    if verbosity is not None:
        kwargs["text"] = {
            "verbosity": verbosity,
        }

    response = client.responses.create(**kwargs)

    return response.output_text


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------

def display_component(
    heading: str,
    component: dict[str, Any] | None,
) -> list[str]:

    lines = [heading, ""]

    if component is None:
        lines.append("[not used]")
        lines.append("")
        return lines

    if "identity" in component:
        lines.append(f"Identity: {component['identity']}")
        lines.append(f"Framing: {component['framing']}")
    else:
        lines.append(f"Label: {component['label']}")

    lines.append("")

    text = component.get("text")

    if text:
        lines.append(text)
    else:
        lines.append("[no text]")

    lines.append("")

    return lines


def append_text_result(
    path: Path,
    record: dict[str, Any],
) -> None:

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
        f"Reasoning effort: "
        f"{record['model'].get('reasoning_effort')}",
        f"Verbosity: {record['model'].get('verbosity')}",
        "",
    ]

    request = record["request"]

    lines.extend(
        display_component("PERSONA", request["persona"])
    )
    lines.extend(
        display_component("INSTRUCTION", request["instruction"])
    )
    lines.extend(
        display_component("PROMPT", request["prompt"])
    )
    lines.extend(
        display_component("SITUATION", request["situation"])
    )
    lines.extend(
        display_component("INTENSITY", request["intensity"])
    )

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

    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(experiment_path: Path) -> None:

    experiment = load_yaml(experiment_path)
    catalogues = load_catalogues()

    metadata = experiment["experiment"]
    generation = experiment["generation"]

    experiment_id = metadata["id"]
    run_count = metadata["runs"]

    output_dir = RESULTS_DIR / experiment_id

    if output_dir.exists():
        raise FileExistsError(
            f"Results directory already exists: {output_dir}\n"
            "Refusing to overwrite an existing experiment."
        )

    output_dir.mkdir(parents=True)

    # Preserve the exact experiment definition used.
    shutil.copy2(
        experiment_path,
        output_dir / "experiment.yaml",
    )

    jsonl_path = output_dir / "results.jsonl"
    text_path = output_dir / "results.txt"

    request = build_request(experiment, catalogues)

    client = OpenAI()

    print()
    print(f"AFAAS experiment: {experiment_id}")
    print(f"Runs:             {run_count}")
    print(f"Model:            {generation['model']}")
    print(f"Results:          {output_dir}")
    print()

    for number in range(1, run_count + 1):

        run_id = f"{experiment_id}-{number:03d}"

        print(
            f"[{number:02d}/{run_count:02d}] "
            f"{run_id} ... ",
            end="",
            flush=True,
        )

        started_at = now_iso()

        try:
            response_text = generate(
                client,
                request,
                generation,
            )
        except Exception:
            print("FAILED")
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
                "reasoning_effort":
                    generation.get("reasoning_effort"),
                "verbosity":
                    generation.get("verbosity"),
            },

            "response": {
                "text": response_text,
            },
        }

        # Write the canonical machine-readable record immediately.
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        append_text_result(
            text_path,
            record,
        )

        print("done")

    print()
    print(f"Experiment {experiment_id} complete.")
    print(f"JSONL: {jsonl_path}")
    print(f"Text:  {text_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Run an AFAAS experiment."
    )

    parser.add_argument(
        "experiment",
        type=Path,
        help="Path to the experiment YAML file.",
    )

    args = parser.parse_args()

    experiment_path = args.experiment.resolve()

    if not experiment_path.exists():
        parser.error(
            f"Experiment file does not exist: {experiment_path}"
        )

    run_experiment(experiment_path)


if __name__ == "__main__":
    main()