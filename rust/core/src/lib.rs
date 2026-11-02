//! `rover-align`: the alignment stage of ROVER, as a token-level edit-distance
//! alignment with a fixed, documented tie order.
//!
//! ROVER (Fiscus, 1997) fuses several recognisers' outputs by aligning each new
//! hypothesis to a growing network of slots, then voting per slot. This crate is the
//! alignment step: a Levenshtein dynamic program (unit costs for substitution,
//! deletion and insertion) over any token type with `==`, returning the edit script.
//!
//! The script is **deterministic under ties**, which is what makes it reusable as a
//! reference implementation. The backtrace starts at the bottom-right corner of the
//! table and, among equal-cost moves, prefers the diagonal (a match or substitution),
//! then a deletion, then an insertion. This is the exact order of the pure-Python
//! ROVER in [ek](https://github.com/thorwhalen/ek), whose differential tests check this
//! crate against it. Other libraries break these ties differently (rapidfuzz, for
//! one), so their scripts are equally optimal but not interchangeable.
//!
//! Time is `O(n * m)`. Memory is one byte per table cell plus two rows of costs.
//!
//! ```
//! use rover_align::{AlignOp, align};
//!
//! let ops = align(&["the", "cat", "sat"], &["the", "bat", "sat", "down"])?;
//! assert_eq!(
//!     ops,
//!     [
//!         AlignOp::Match { spine: 0, hyp: 0 },
//!         AlignOp::Match { spine: 1, hyp: 1 }, // a substitution: cat -> bat
//!         AlignOp::Match { spine: 2, hyp: 2 },
//!         AlignOp::Insert { hyp: 3 },
//!     ]
//! );
//! # Ok::<(), rover_align::AlignError>(())
//! ```

/// One step of an alignment script, in left-to-right order.
///
/// The set is closed by definition (the three edit operations), so the enum is
/// exhaustive. Indices refer to the `spine` and `hyp` slices given to [`align`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlignOp {
    /// `spine[spine]` and `hyp[hyp]` share a slot: equal tokens, or a substitution.
    Match {
        /// Index into the spine.
        spine: usize,
        /// Index into the hypothesis.
        hyp: usize,
    },
    /// `spine[spine]` has no counterpart in the hypothesis.
    Delete {
        /// Index into the spine.
        spine: usize,
    },
    /// `hyp[hyp]` has no counterpart in the spine.
    Insert {
        /// Index into the hypothesis.
        hyp: usize,
    },
}

/// Why [`align`] could not run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AlignError {
    /// The `(spine + 1) x (hyp + 1)` backtrace table does not fit in memory.
    #[error("cannot allocate the alignment table for {spine} x {hyp} tokens")]
    TableTooLarge {
        /// Tokens in the spine.
        spine: usize,
        /// Tokens in the hypothesis.
        hyp: usize,
    },
}

/// The move that reached a table cell, chosen with the documented preference.
#[derive(Debug, Clone, Copy)]
enum Step {
    /// From `(i - 1, j - 1)`: a match or substitution.
    Diagonal,
    /// From `(i - 1, j)`: a deletion.
    Up,
    /// From `(i, j - 1)`: an insertion.
    Left,
}

/// Aligns `hyp` to `spine` with unit-cost edits and returns the edit script.
///
/// The script has minimal cost (the number of substitutions, deletions and
/// insertions) and, among minimal scripts, is the one the backtrace takes when it
/// prefers the diagonal, then a deletion, then an insertion at each cell, starting
/// from the end of both sequences. See the crate documentation.
///
/// # Errors
///
/// [`AlignError::TableTooLarge`] when the `(spine.len() + 1) * (hyp.len() + 1)`-byte
/// backtrace table cannot be allocated. The function never aborts on allocation
/// failure and never panics.
///
/// # Examples
///
/// ```
/// use rover_align::{AlignOp, align};
///
/// // One extra token: the leading one becomes the insertion.
/// let ops = align(&['a'], &['a', 'a'])?;
/// assert_eq!(ops, [AlignOp::Insert { hyp: 0 }, AlignOp::Match { spine: 0, hyp: 1 }]);
/// # Ok::<(), rover_align::AlignError>(())
/// ```
pub fn align<T: PartialEq>(spine: &[T], hyp: &[T]) -> Result<Vec<AlignOp>, AlignError> {
    let too_large = || AlignError::TableTooLarge {
        spine: spine.len(),
        hyp: hyp.len(),
    };
    let rows = spine.len().checked_add(1).ok_or_else(too_large)?;
    let cols = hyp.len().checked_add(1).ok_or_else(too_large)?;
    let cells = rows.checked_mul(cols).ok_or_else(too_large)?;

    let mut steps: Vec<Step> = Vec::new();
    steps.try_reserve_exact(cells).map_err(|_| too_large())?;
    let mut prev: Vec<usize> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    prev.try_reserve_exact(cols).map_err(|_| too_large())?;
    cur.try_reserve_exact(cols).map_err(|_| too_large())?;

    // Row 0: cost j, reached by j insertions (cell (0, 0) is never read).
    prev.extend(0..cols);
    cur.resize(cols, 0);
    steps.resize(cols, Step::Left);

    for (i, s_tok) in spine.iter().enumerate() {
        // Column 0: cost i + 1, reached by a deletion.
        cur[0] = i + 1;
        steps.push(Step::Up);
        for (j, h_tok) in hyp.iter().enumerate() {
            let diagonal = prev[j] + usize::from(s_tok != h_tok);
            let up = prev[j + 1] + 1;
            let left = cur[j] + 1;
            // The reference's order: diagonal if optimal, else deletion, else insertion.
            let (cost, step) = if diagonal <= up && diagonal <= left {
                (diagonal, Step::Diagonal)
            } else if up <= left {
                (up, Step::Up)
            } else {
                (left, Step::Left)
            };
            cur[j + 1] = cost;
            steps.push(step);
        }
        std::mem::swap(&mut prev, &mut cur);
    }

    let mut ops = Vec::new();
    ops.try_reserve_exact(spine.len() + hyp.len())
        .map_err(|_| too_large())?;
    let (mut i, mut j) = (spine.len(), hyp.len());
    while i > 0 || j > 0 {
        match steps[i * cols + j] {
            Step::Diagonal => {
                i -= 1;
                j -= 1;
                ops.push(AlignOp::Match { spine: i, hyp: j });
            }
            Step::Up => {
                i -= 1;
                ops.push(AlignOp::Delete { spine: i });
            }
            Step::Left => {
                j -= 1;
                ops.push(AlignOp::Insert { hyp: j });
            }
        }
    }
    ops.reverse();
    Ok(ops)
}

#[cfg(test)]
mod tests {
    use super::*;
    use AlignOp::{Delete, Insert, Match};

    #[test]
    fn empty_inputs() {
        assert_eq!(align::<u8>(&[], &[]), Ok(vec![]));
        assert_eq!(align(&[1], &[]), Ok(vec![Delete { spine: 0 }]));
        assert_eq!(align(&[], &[1]), Ok(vec![Insert { hyp: 0 }]));
    }

    #[test]
    fn substitution_beats_delete_plus_insert() {
        assert_eq!(align(&['a'], &['b']), Ok(vec![Match { spine: 0, hyp: 0 }]));
    }

    #[test]
    fn ties_prefer_diagonal_then_deletion_then_insertion() {
        // diagonal (two substitutions) over del + ins, all three of cost 2
        assert_eq!(
            align(&['a', 'b'], &['b', 'a']),
            Ok(vec![Match { spine: 0, hyp: 0 }, Match { spine: 1, hyp: 1 }])
        );
        // at the corner, deletion and insertion tie and the diagonal does not: deletion
        assert_eq!(
            align(&['a', 'b', 'a'], &['b', 'a', 'b']),
            Ok(vec![
                Insert { hyp: 0 },
                Match { spine: 0, hyp: 1 },
                Match { spine: 1, hyp: 2 },
                Delete { spine: 2 },
            ])
        );
        assert_eq!(
            align(&['a', 'a'], &['a']),
            Ok(vec![Delete { spine: 0 }, Match { spine: 1, hyp: 0 }])
        );
    }

    #[test]
    fn works_on_optional_tokens() {
        // A NULL spine representative equals nothing in the hypothesis.
        assert_eq!(
            align(&[None, Some("a")], &[Some("a")]),
            Ok(vec![Delete { spine: 0 }, Match { spine: 1, hyp: 0 }])
        );
    }

    #[test]
    fn an_impossible_table_is_an_error_not_an_abort() {
        let huge = vec![(); usize::MAX]; // zero-sized tokens: no memory behind them
        assert_eq!(
            align(&huge, &huge),
            Err(AlignError::TableTooLarge {
                spine: usize::MAX,
                hyp: usize::MAX
            })
        );
        assert!(matches!(
            align(&huge, &[]),
            Err(AlignError::TableTooLarge { .. })
        ));
        assert!(matches!(
            align(&[(); 2], &huge[..usize::MAX / 2]),
            Err(AlignError::TableTooLarge { .. })
        ));
    }
}
