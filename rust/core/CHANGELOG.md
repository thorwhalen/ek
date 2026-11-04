# Changelog

## 0.1.0 (unreleased)

- `align(spine, hyp)`: unit-cost Levenshtein alignment over any `T: PartialEq`,
  returning `Vec<AlignOp>` (`Match`, `Delete`, `Insert`) with the tie order of ek's
  Python ROVER (diagonal, then deletion, then insertion, backtracking from the end).
- `AlignError::TableTooLarge` when the backtrace table cannot be allocated.
