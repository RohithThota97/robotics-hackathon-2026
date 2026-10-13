# Memory schema · task 11

Fills in the template in [week1.md §5.3](../week1.md) for the practice bins, and walks a reshuffle through the search rules.

Spatial memory is the robot remembering where it last saw each item and searching the likeliest places first. The plan names it as a first-prize factor most teams will skip.

## Places

Three trays on a table, standing in for the shelves and bins on site. At home there are no AprilTags and no robot, so those columns are the shape the Day 1 rows will take, not measurements.

| place ID | type | part of | AprilTag ID | observation pose | usual categories | travel cost |
|---|---|---|---|---|---|---|
| `table` | zone | — | — | `look_table` | — | 0 s |
| `B1` | bin | `table` | *(Day 1)* | `look_B1` | hand tool | about 3 s |
| `B2` | bin | `table` | *(Day 1)* | `look_B2` | hand tool | about 5 s |
| `B3` | bin | `table` | *(Day 1)* | `look_B3` | measuring tool | about 8 s |

`B3` carries `measuring tool` so the category prior sends `tape_measure` there first. That is the prior doing real work: without it, a tape measure missing from its last-seen bin would be searched for in bin order rather than in likelihood order.

## Sightings

One row per observation. Rows below are a worked example from a practice scan at 10:40 followed by a reshuffle — not real detections. `position` stays blank at home because there is no robot base frame yet.

| item | place | found | position | time | confidence | seen by |
|---|---|---|---|---|---|---|
| `screwdriver_small` | `B2` | yes | — | 10:42:13 | 0.91 | scan |
| `screwdriver_large` | `B1` | yes | — | 10:42:15 | 0.88 | scan |
| `wrench` | `B1` | yes | — | 10:42:16 | 0.93 | scan |
| `hammer` | `B2` | yes | — | 10:42:18 | 0.95 | scan |
| `pliers` | `B3` | yes | — | 10:42:20 | 0.84 | scan |
| `tape_measure` | `B3` | yes | — | 10:42:21 | 0.90 | scan |
| `screwdriver_small` | `B2` | yes | — | 10:51:04 | 0.89 | pick |
| `screwdriver_small` | `tray` | yes | — | 10:51:12 | 0.97 | delivery |

## The three rules

1. **Latest `found = yes` wins.** An item's last-seen place is the most recent row where it was actually found. For `screwdriver_small` above that is `tray` at 10:51:12, not `B2`.
2. **A later miss means it moved.** A row with `found = no` for a place does not erase history; it demotes that place. The item is still "last seen" where it was found, but that place is now known to be wrong.
3. **Not found anywhere → full scan, said honestly.** Run a scan over every place, report that the item is missing, and offer the closest alternative ("No tape measure. There is a wrench in B1.").

Memory updates on **every** observation, including while looking for something else. Spotting the hammer in B3 during a search for pliers is free information and must be recorded.

## Reshuffle walk-through

The required acceptance check for task 11. Starting state is the 10:42 scan above. Someone then reshuffles the trays without telling the robot, moving `wrench` from `B1` to `B3`.

**Request:** "Bring mir den Maulschlüssel" → `item: wrench`, `status: clear`.

**Step 1 — build the search order.**

| rank | place | why |
|---|---|---|
| 1 | `B1` | last seen here (10:42:16, `found = yes`) |
| 2 | `B2` | other place, ranked by recency and travel cost (5 s) |
| 3 | `B3` | other place, higher travel cost (8 s); no `hand tool` prior |

**Step 2 — look in `B1`.** Not there. Write the miss:

| item | place | found | time | seen by |
|---|---|---|---|---|
| `wrench` | `B1` | no | 11:03:40 | scan |

By rule 2 the wrench has moved. `B1` is now demoted. Say it out loud: *"Nicht mehr in B1. Ich schaue in B2."*

**Step 3 — look in `B2`.** Not there either. Another miss row; skip `B2` for the rest of this search (rule: never re-check a place already checked in this search).

**Step 4 — look in `B3`.** Found.

| item | place | found | position | time | confidence | seen by |
|---|---|---|---|---|---|---|
| `wrench` | `B3` | yes | — | 11:04:12 | 0.90 | scan |

**Resulting state.** The wrench's last-seen place is now `B3`. The next request for it searches `B3` first — one look instead of three. That improvement is the whole point of the feature, and it is what the demo should show a jury member doing: reshuffle the trays, ask twice, watch the second search go straight to the right place.

### The case that must not be skipped

Ask for the tape measure after someone has pocketed it. Searches `B3` (last seen, and the `measuring tool` prior agrees), then `B1`, then `B2`, finds nothing, and by rule 3 must say so rather than returning silently or picking up a wrench instead:

> "Ich finde das Maßband nicht. Ich habe B3, B1 und B2 durchsucht. In B1 liegt ein Maulschlüssel — soll ich den bringen?"

A robot that quietly hands over the wrong tool looks far worse to a jury than one that says it cannot find something.

## What the dashboard and voice read from this

- Each item's place and the **age** of that sighting — "seen 2 min ago" — so a stale entry is visibly stale.
- The voice says what *changed*, not just the result: "Not in bin B2 any more. Checking B4."
