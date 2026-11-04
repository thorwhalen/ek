//! Golden cases written by `tests/golden/make_golden.py` from the Python reference.

use rover_align::{AlignOp, align};

/// `[spine, hyp, [[op, spine_idx, hyp_idx], ...]]`, as `ek._reference._align` returns it.
type Case = (
    Vec<Option<String>>,
    Vec<Option<String>>,
    Vec<(String, i64, i64)>,
);

/// An op as `(name, spine index, hyp index)`, `None` where Python writes `-1`.
fn as_python(op: AlignOp) -> (&'static str, Option<usize>, Option<usize>) {
    match op {
        AlignOp::Match { spine, hyp } => ("match", Some(spine), Some(hyp)),
        AlignOp::Delete { spine } => ("del", Some(spine), None),
        AlignOp::Insert { hyp } => ("ins", None, Some(hyp)),
    }
}

#[test]
fn align_matches_python_golden() -> Result<(), serde_json::Error> {
    let cases: Vec<Case> = serde_json::from_str(include_str!("golden/align.json"))?;
    assert!(cases.len() >= 600, "golden file looks truncated");
    for (spine, hyp, expected) in cases {
        let ops = align(&spine, &hyp).expect("small inputs always fit");
        let got: Vec<_> = ops.into_iter().map(as_python).collect();
        let want: Vec<_> = expected
            .iter()
            .map(|(op, s, h)| {
                (
                    op.as_str(),
                    usize::try_from(*s).ok(),
                    usize::try_from(*h).ok(),
                )
            })
            .collect();
        assert_eq!(got, want, "spine={spine:?} hyp={hyp:?}");
    }
    Ok(())
}
