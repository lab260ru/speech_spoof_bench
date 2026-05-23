"""Build the dataset's parquet shards under `data/test-*.parquet`.

The output schema MUST match the canonical schema (PLAN.md §1.2):

    Features({
        "path":  Value("string"),
        "audio": Audio(sampling_rate=16000),
        "label": ClassLabel(names=["bonafide", "spoof"]),
        "notes": Value("string"),
    })

`notes` is a JSON string and MUST contain a unique `utterance_id`. Audio MUST
be 16 kHz mono. Resample at build time.

After building, validate with:

    speech-spoof-bench validate-dataset <this-dir> --skip-submissions
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Implement parquet build for this dataset.")


if __name__ == "__main__":
    main()
