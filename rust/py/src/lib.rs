//! Python bindings for ek's optional native kernels: a thin layer over `rover_align`.
//!
//! The Python facade (`ek.qe.rover._align`) owns the public signature, defaults,
//! docstring, validation and routing; functions here take every argument explicitly.

use pyo3::prelude::*;

/// The `ek_rs` extension module.
#[pymodule]
mod ek_rs {
    use pyo3::exceptions::{PyMemoryError, PyOverflowError, PyRuntimeError};
    use pyo3::intern;
    use pyo3::prelude::*;
    use pyo3::types::PyString;
    use rover_align::{AlignError, AlignOp};

    /// True in a debug build: the benchmark refuses to time one.
    #[pymodule_export]
    const DEBUG_BUILD: bool = cfg!(debug_assertions);

    /// One `_align` operation as Python sees it: `(op, spine index, hyp index)`.
    type PyOp<'py> = (Bound<'py, PyString>, isize, isize);

    /// Maps each core error to the exception the pure-Python version raised.
    ///
    /// A function, not `impl From<AlignError> for PyErr`: both types are foreign to
    /// this crate, so that impl breaks the orphan rule (E0117).
    fn to_pyerr(err: AlignError) -> PyErr {
        match err {
            // Python's list-of-lists table fails the same way: MemoryError.
            e @ AlignError::TableTooLarge { .. } => PyMemoryError::new_err(e.to_string()),
            // AlignError is #[non_exhaustive]: a variant added later must still map somewhere.
            e => PyRuntimeError::new_err(e.to_string()),
        }
    }

    /// A slice index as a Python int (indices of a `Vec` always fit in `isize`).
    fn index(i: usize) -> PyResult<isize> {
        isize::try_from(i).map_err(|_| PyOverflowError::new_err("index exceeds isize"))
    }

    /// Aligns two sequences of interned token ids, `_align`'s operations out.
    ///
    /// Equal ids are equal tokens. Returns `[(op, spine_idx, hyp_idx), ...]` in
    /// left-to-right order with `op` in `{"match", "del", "ins"}` and `-1` for the
    /// missing side, exactly as `ek._reference._align` does on the original tokens.
    #[pyfunction]
    #[pyo3(signature = (spine, hyp, /))]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts a list argument by value; the Vec is moved into detach"
    )]
    fn align(py: Python<'_>, spine: Vec<usize>, hyp: Vec<usize>) -> PyResult<Vec<PyOp<'_>>> {
        let ops = py
            .detach(|| rover_align::align(&spine, &hyp))
            .map_err(to_pyerr)?;
        let (matched, deleted, inserted) =
            (intern!(py, "match"), intern!(py, "del"), intern!(py, "ins"));
        ops.into_iter()
            .map(|op| {
                Ok(match op {
                    AlignOp::Match { spine, hyp } => (matched.clone(), index(spine)?, index(hyp)?),
                    AlignOp::Delete { spine } => (deleted.clone(), index(spine)?, -1),
                    AlignOp::Insert { hyp } => (inserted.clone(), -1, index(hyp)?),
                })
            })
            .collect()
    }
}
