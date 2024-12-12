use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

const K_B: f64 = 8.617_330_337_217_213e-5;

#[pyfunction]
pub fn log_fermi_spherical_potential<'py>(
    _py: Python<'py>,
    positions: PyReadonlyArray2<f64>,
    radius: f64,
    temperature: f64,
    beta: f64,
) -> (f64, Bound<'py, PyArray2<f64>>) {
    let eps = 1e-9;
    let dists = positions
        .as_array()
        .mapv(|x| x.powi(2))
        .sum_axis(Axis(1))
        .mapv(f64::sqrt);
    let exp_term = (beta * (&dists - radius)).mapv(f64::exp);
    let k_t = K_B * temperature;
    let e_i = k_t * (1.0 + &exp_term).mapv(f64::ln);
    let e = e_i.sum();
    let grad_multiplier = k_t * beta * &exp_term / (&dists * (1.0 + &exp_term) + eps);
    // Multiply each element of grad_multiplier by the corresponding row of positions
    let mut e_grad = (positions.as_array()).into_owned();
    for (i, mut row) in e_grad.axis_iter_mut(Axis(0)).enumerate() {
        row *= grad_multiplier[i];
    }
    (e, e_grad.into_pyarray(_py))
}
