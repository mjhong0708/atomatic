pub mod rmsd;
pub mod wall;

use pyo3::prelude::*;
use rmsd::compute_rmsd;
use wall::log_fermi_spherical_potential;

/// Module definition
#[pymodule]
fn _ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_rmsd, m)?)?;
    m.add_function(wrap_pyfunction!(log_fermi_spherical_potential, m)?)?;
    Ok(())
}
