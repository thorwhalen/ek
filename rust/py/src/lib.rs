//! Python bindings for ek's optional native kernels: a thin layer over `rover_align`.
//!
//! The Python facade (`ek.qe.rover._align`) owns the public signature, defaults,
//! docstring, validation and routing; functions here take every argument explicitly.

use pyo3::prelude::*;

/// The `ek_rs` extension module.
#[pymodule]
mod ek_rs {
    use pyo3::exceptions::PyNotImplementedError;
    use pyo3::prelude::*;

    /// True in a debug build: the benchmark refuses to time one.
    #[pymodule_export]
    const DEBUG_BUILD: bool = cfg!(debug_assertions);

    /// Aligns two sequences of interned token ids (skeleton: not ported yet).
    #[pyfunction]
    #[pyo3(signature = (spine, hyp, /))]
    fn align(spine: Vec<usize>, hyp: Vec<usize>) -> PyResult<Vec<(String, isize, isize)>> {
        let _ = (spine, hyp);
        Err(PyNotImplementedError::new_err("ek_rs.align is not ported yet"))
    }
}
