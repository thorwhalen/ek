# rover-align

The alignment step of ROVER (Recognizer Output Voting Error Reduction, Fiscus 1997):
a unit-cost Levenshtein alignment between two token sequences, returned as an
edit script, for any token type with `==`.

What sets it apart from a plain edit-distance crate is that the script is
**deterministic under ties**. Among the many minimal scripts, it returns the one a
backtrace from the end of both sequences takes when it prefers, at each cell, the
diagonal (a match or a substitution), then a deletion, then an insertion. That is
the order of the pure-Python ROVER in [ek](https://github.com/thorwhalen/ek), whose
differential tests check this crate against it, so a consensus built on it is
reproducible across languages. Libraries that break these ties differently (such
as rapidfuzz) return scripts that are just as optimal but different.

```rust
use rover_align::{AlignOp, align};

fn main() -> Result<(), rover_align::AlignError> {
    let ops = align(&["the", "cat", "sat"], &["the", "bat", "sat", "down"])?;
    assert_eq!(
        ops,
        [
            AlignOp::Match { spine: 0, hyp: 0 },
            AlignOp::Match { spine: 1, hyp: 1 }, // a substitution: cat -> bat
            AlignOp::Match { spine: 2, hyp: 2 },
            AlignOp::Insert { hyp: 3 },
        ]
    );
    Ok(())
}
```

Time is `O(n * m)`. Memory is one byte per table cell plus two rows of costs, so two
5 000-token sequences need 25 MB. An impossible table returns
`AlignError::TableTooLarge` instead of aborting.

The voting stage of ROVER (scoring the candidates of each slot by frequency and
confidence) lives in ek's Python code for now.

MSRV: Rust 1.85. Licence: MIT.
