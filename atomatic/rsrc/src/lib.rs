extern crate openblas_src;
pub mod rmsd;
pub mod shake;
pub mod wall;

use pyo3::prelude::*;

#[pymodule]
mod _ext {
    use super::*;
    #[pymodule_export]
    use super::rmsd::compute_rmsd;
    #[pymodule_export]
    use super::wall::log_fermi_spherical_potential;

    #[pymodule]
    mod shake_utils {
        #[pymodule_export]
        use super::super::shake::adjust_positions;
        #[pymodule_export]
        use super::super::shake::adjust_momenta;
    }
}
