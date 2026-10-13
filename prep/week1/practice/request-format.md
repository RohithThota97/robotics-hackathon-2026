# Item request format · task 9

Fills in the template in [week1.md §5.1](../week1.md) and tries it on all six practice items, in German and English.

The language model never drives the motors. It fills in this form and picks from five fixed skills — `find`, `pick`, `deliver`, `scan`, `cancel` — and every field is checked against [catalog.yaml](catalog.yaml) before anything moves.

## The form

| Field | Type | Rule |
|---|---|---|
| `text` | string | Exact phrasing as heard. Goes into the trial log verbatim; never cleaned up. |
| `language` | `de` \| `en` | Detected, not assumed. |
| `item` | list of catalog IDs | One ID when clear, all candidates when ambiguous, empty when unknown. |
| `attributes` | map | Normalised to catalog words, e.g. `size: small`. |
| `quantity` | int | Default `1`. |
| `destination` | `tray` \| `hand-over` | Default `tray`. |
| `status` | `clear` \| `ambiguous` \| `unknown` | Decides what happens next. |

### What each status does

- **`clear`** — exactly one catalog ID survived matching. Proceed.
- **`ambiguous`** — two or more candidates. Ask **one** short question naming the distinguishing attribute, then stop and wait. Never guess.
- **`unknown`** — no candidate. Say so plainly and offer to enroll the item.

A request is only `clear` when the item resolves to one ID **and** every attribute is a catalog word. An unrecognised attribute makes it `ambiguous`, not `clear` — "the metric screwdriver" must not silently become "the screwdriver".

## Attribute normalisation

German adjective endings change with case and gender, so each surface form collapses to one catalog word before matching. The table lives in `attribute_normalisation` in [catalog.yaml](catalog.yaml).

| Catalog word | Surface forms accepted |
|---|---|
| `size: small` | small, little, klein, kleine, kleinen, kleiner, kleines |
| `size: large` | large, big, groß, große, großen, großer, großes |

"Bring mir den **kleinen** Schraubendreher" and "bring me the **small** screwdriver" must produce the identical `size: small`.

## Test sentences

Each row is a sentence tried against the form. **Expected** is the parse that counts as correct.

### Clear — one item resolves

| # | Lang | Sentence | Expected `item` | `attributes` | `destination` | `status` |
|---|---|---|---|---|---|---|
| 1 | de | Bring mir den kleinen Schraubendreher | `screwdriver_small` | `size: small` | hand-over | clear |
| 2 | de | Leg den großen Schraubendreher in die Ablage | `screwdriver_large` | `size: large` | tray | clear |
| 3 | en | Bring me the large screwdriver | `screwdriver_large` | `size: large` | hand-over | clear |
| 4 | de | Ich brauche den Hammer | `hammer` | none | tray | clear |
| 5 | en | Hand me the hammer | `hammer` | none | hand-over | clear |
| 6 | de | Wo ist die Kombizange? | `pliers` | none | — (`find`) | clear |
| 7 | en | Pass me the pliers | `pliers` | none | hand-over | clear |
| 8 | de | Gib mir das Maßband | `tape_measure` | none | hand-over | clear |
| 9 | en | I need the tape measure | `tape_measure` | none | tray | clear |
| 10 | de | Bring mir den Maulschlüssel | `wrench` | none | hand-over | clear |
| 11 | en | Bring the spanner | `wrench` | none | tray | clear |
| 12 | de | Bring mir zwei Schraubendreher | `screwdriver_small`, `screwdriver_large` | none | tray | clear *(quantity 2)* |

Row 12 is the edge case: both screwdrivers are wanted, so two candidates is the *answer*, not an ambiguity. `quantity: 2` with two matching IDs is `clear`.

### Ambiguous — ask one question

| # | Lang | Sentence | Candidates | Question to ask | `status` |
|---|---|---|---|---|---|
| 13 | en | Bring me the screwdriver | `screwdriver_small`, `screwdriver_large` | "Two screwdrivers here, small or large?" | ambiguous |
| 14 | de | Bring mir den Schraubendreher | `screwdriver_small`, `screwdriver_large` | "Ich sehe zwei Schraubendreher. Den kleinen oder den großen?" | ambiguous |
| 15 | de | Gib mir den Schraubenzieher | `screwdriver_small`, `screwdriver_large` | "Den kleinen oder den großen?" | ambiguous |
| 16 | en | Bring me the metric screwdriver | `screwdriver_small`, `screwdriver_large` | "I don't know 'metric'. Small or large?" | ambiguous |

Row 15 checks the colloquial synonym *Schraubenzieher* still matches. Row 16 is the important one: an unknown attribute must **not** be silently dropped, or the robot would answer a question nobody asked.

### Unknown — say so and offer to enroll

| # | Lang | Sentence | Expected `item` | Response | `status` |
|---|---|---|---|---|---|
| 17 | en | Bring me the soldering iron | empty | "I don't know a soldering iron. Show me one and I'll learn it." | unknown |
| 18 | de | Bring mir die Säge | empty | "Ich kenne keine Säge. Zeig sie mir, dann lerne ich sie." | unknown |

## How to run this check

These sentences are the acceptance test for task 9. Read each one aloud into Whisper once the ML image exists (task 3), confirm the transcript, then check the parse against the **Expected** columns. Done when every sentence maps to a filled-in form with the right `status`.

Two failure modes to watch for, both worse than a wrong answer:

- A `clear` status on an ambiguous sentence — the robot acts confidently on a guess.
- A dropped attribute — "metric" vanishing turns row 16 into row 13 without anyone noticing.
