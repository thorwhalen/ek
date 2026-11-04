//! Properties of `align` on every pair of short sequences over a small alphabet.

use rover_align::{AlignOp, align};

/// Plain Levenshtein distance, computed independently of `align`.
fn distance(a: &[u8], b: &[u8]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(x != y))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

fn words(n: usize) -> Vec<Vec<u8>> {
    let mut out = vec![vec![]];
    let mut last = vec![vec![]];
    for _ in 0..n {
        last = last
            .iter()
            .flat_map(|w: &Vec<u8>| {
                (b'a'..=b'c').map(move |c| {
                    let mut w = w.clone();
                    w.push(c);
                    w
                })
            })
            .collect();
        out.extend(last.iter().cloned());
    }
    out
}

/// The script visits every index of both sides once, in order, and costs the distance.
#[test]
fn scripts_are_complete_ordered_and_minimal() {
    let all = words(5);
    for a in &all {
        for b in &all {
            let ops = align(a, b).expect("small inputs always fit");
            let (mut next_a, mut next_b, mut cost) = (0, 0, 0);
            for op in ops {
                match op {
                    AlignOp::Match { spine, hyp } => {
                        assert_eq!((spine, hyp), (next_a, next_b));
                        cost += usize::from(a[spine] != b[hyp]);
                        next_a += 1;
                        next_b += 1;
                    }
                    AlignOp::Delete { spine } => {
                        assert_eq!(spine, next_a);
                        cost += 1;
                        next_a += 1;
                    }
                    AlignOp::Insert { hyp } => {
                        assert_eq!(hyp, next_b);
                        cost += 1;
                        next_b += 1;
                    }
                }
            }
            assert_eq!((next_a, next_b), (a.len(), b.len()), "a={a:?} b={b:?}");
            assert_eq!(cost, distance(a, b), "a={a:?} b={b:?}");
        }
    }
}
