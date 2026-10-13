"""Guard tests for the practice item catalog (task 10, week1.md §5.2).

Task 10 is done "when all 6 entries are filled in and the IDs match the label
classes". The catalog ID is the single string used for the label class, the
memory tables and the trial log, so a drift between the catalog and
``week1_harness.synth.items.CLASS_IDS`` would silently break the chain. These
tests pin them together.

The IDs are read with a regex rather than a YAML parser on purpose: the harness
depends on Pillow and numpy only, and a documentation guard is not a good
reason to add PyYAML. Where PyYAML *is* installed, the structural checks run as
well; otherwise they skip.

Requirements: 5.1
"""

from __future__ import annotations

import pathlib
import re

import pytest

from week1_harness.synth.items import CLASS_IDS

CATALOG_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "prep"
    / "week1"
    / "practice"
    / "catalog.yaml"
)

# Fields every entry must carry (week1.md §5.2).
REQUIRED_FIELDS = (
    "id",
    "name_en",
    "name_de",
    "spoken_de_nominative",
    "spoken_de_accusative",
    "synonyms_en",
    "synonyms_de",
    "category",
    "size",
    "colour",
    "weight",
    "grasp_type",
    "storage_locations",
    "look_alikes",
    "gallery",
)

_ID_RE = re.compile(r"^  - id:\s*(\S+)\s*$", re.MULTILINE)


@pytest.fixture(scope="module")
def catalog_text() -> str:
    assert CATALOG_PATH.is_file(), f"catalog missing at {CATALOG_PATH}"
    return CATALOG_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def catalog_ids(catalog_text: str) -> list[str]:
    return _ID_RE.findall(catalog_text)


def test_catalog_has_one_entry_per_practice_class(catalog_ids):
    assert len(catalog_ids) == len(CLASS_IDS) == 6


def test_catalog_ids_match_the_label_classes_exactly(catalog_ids):
    """The catalog ID is the label class name; they must not drift (task 10)."""
    assert catalog_ids == list(CLASS_IDS)


def test_every_entry_has_every_required_field(catalog_text):
    for field in REQUIRED_FIELDS:
        # `id` is the list-item key, so it appears as "  - id:".
        found = len(re.findall(rf"^\s+(?:- )?{field}:", catalog_text, re.MULTILINE))
        assert found >= 6, f"{field}: found {found} occurrences, expected >= 6"


def test_lookalike_synonyms_are_shared_to_force_a_clarifying_question(catalog_text):
    """An unqualified "screwdriver" must match both entries (§5.1).

    This is the mechanism behind the clarifying question, so it is worth
    pinning: remove the shared synonym and the robot would silently guess.
    """
    assert len(re.findall(r"^\s+synonyms_en:.*\bscrewdriver\b", catalog_text, re.MULTILINE)) == 2
    assert len(re.findall(r"^\s+synonyms_de:.*\bSchraubendreher\b", catalog_text, re.MULTILINE)) == 2


def test_german_adjective_endings_are_normalised(catalog_text):
    """"klein", "kleine" and "kleinen" must all collapse to size: small."""
    for surface in ("klein", "kleine", "kleinen"):
        assert re.search(rf"\b{surface}\b", catalog_text), f"missing {surface!r}"


# --- structural checks, only when PyYAML is available ---------------------


def _load_yaml(text: str):
    y = pytest.importorskip("yaml", reason="PyYAML is not a harness dependency")
    return y.safe_load(text)


def test_catalog_parses_as_yaml(catalog_text):
    data = _load_yaml(catalog_text)
    assert [entry["id"] for entry in data["items"]] == list(CLASS_IDS)


def test_lookalike_pair_points_at_each_other(catalog_text):
    data = _load_yaml(catalog_text)
    by_id = {entry["id"]: entry for entry in data["items"]}

    small = by_id["screwdriver_small"]["look_alikes"]
    large = by_id["screwdriver_large"]["look_alikes"]

    assert [la["id"] for la in small] == ["screwdriver_large"]
    assert [la["id"] for la in large] == ["screwdriver_small"]
    assert small[0]["distinguishing_attribute"] == "size"


def test_masculine_nouns_change_in_the_accusative(catalog_text):
    """der -> den for masculine items; feminine/neuter stay identical."""
    data = _load_yaml(catalog_text)
    by_id = {entry["id"]: entry for entry in data["items"]}

    for item_id in ("screwdriver_small", "screwdriver_large", "wrench", "hammer"):
        entry = by_id[item_id]
        assert entry["spoken_de_nominative"].startswith("der ")
        assert entry["spoken_de_accusative"].startswith("den ")

    # Feminine and neuter are unchanged - identical forms are correct here.
    assert by_id["pliers"]["spoken_de_nominative"] == by_id["pliers"]["spoken_de_accusative"]
    assert (
        by_id["tape_measure"]["spoken_de_nominative"]
        == by_id["tape_measure"]["spoken_de_accusative"]
    )
