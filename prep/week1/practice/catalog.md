# Practice item catalog · task 10

Fills in the template in [week1.md §5.2](../week1.md) for the six practice items.

**[catalog.yaml](catalog.yaml) is the source of truth.** All 15 fields per item live there, and Week 2's request parser will load it directly. This page explains the decisions behind it and gives an at-a-glance index; it deliberately does **not** repeat the field values, because two copies of the same table drift apart and then nobody knows which is right.

## The six entries

| ID | Name (EN) | Spoken (DE), nominative | Spoken (DE), accusative | Bins |
|---|---|---|---|---|
| `screwdriver_small` | flat-head screwdriver, small | der kleine Schraubendreher | den kleinen Schraubendreher | B1–B3 |
| `screwdriver_large` | flat-head screwdriver, large | der große Schraubendreher | den großen Schraubendreher | B1–B3 |
| `wrench` | open-end wrench | der Maulschlüssel | den Maulschlüssel | B1–B3 |
| `hammer` | claw hammer | der Hammer | den Hammer | B1–B2 |
| `pliers` | combination pliers | die Kombizange | die Kombizange | B1–B3 |
| `tape_measure` | retractable tape measure | das Maßband | das Maßband | B2–B3 |

## Three decisions worth knowing

**The ID is a contract.** The same string is the label class, the memory table key and the trial-log column. The six IDs are identical to `week1_harness.synth.items.CLASS_IDS`, and [tests/test_catalog.py](../../../tests/test_catalog.py) fails if they ever drift. Rename an ID and you must rename it in the dataset too.

**Shared synonyms are intentional, not sloppy.** `screwdriver` and `Schraubendreher` appear in *both* screwdriver entries. An unqualified "bring me the screwdriver" therefore matches two candidates, which is what forces the clarifying question described in §5.1. If you "fix" this by making the synonyms unique, the robot will silently pick one and look confident while being wrong half the time — the worst possible failure in front of a jury.

**Two items have identical spoken forms, and that is correct German.** Only masculine nouns change in the accusative (*der* → *den*), so the four masculine items differ between the two forms while `pliers` (*die* Kombizange, feminine) and `tape_measure` (*das* Maßband, neuter) do not. This looks like a copy-paste error and is not; the test asserts it deliberately so nobody "corrects" it later.

## Still to do on real footage

`gallery:` points at a folder per item, for 5–10 clean reference crops from several angles. Those crops come out of the task 6 filming and cannot be produced here — see [blocked-tasks.md](../blocked-tasks.md). Everything else in the catalog is complete and testable now.

Sizes, colours and weights describe the household tools you will actually film. Measure yours and correct them; they are plausible values for a 15 cm screwdriver and a 450 g claw hammer, not measurements of your drawer.
