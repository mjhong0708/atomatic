use ndarray::prelude::*;
use nshare::{IntoNalgebra, IntoNdarray2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use nalgebra_lapack::LU;

#[inline]
fn _inverse(m: ArrayView2<f64>) -> Array2<f64> {
    let m = m.into_nalgebra().clone_owned();
    // let lu = m.to_owned().lu();
    let lu = LU::new(m.to_owned());
    let inv = lu.inverse().expect("Inversion failed");
    inv.into_ndarray2()
}

pub fn find_mic(
    dr: ArrayView1<f64>,
    cell: ArrayView2<f64>,
    inv_cell: ArrayView2<f64>,
    pbc: ArrayView1<bool>,
) -> Array1<f64> {
    let dr = dr.to_shape((1, 3)).unwrap();
    let rec_masked = &inv_cell * &pbc.to_shape((3, 1)).unwrap().mapv(|x| x as i64 as f64);
    let dri = dr.dot(&rec_masked).mapv(|x| x.round());
    (dr - dri.dot(&cell)).to_shape((3,)).unwrap().to_owned()
}

#[pyfunction]
pub fn adjust_positions<'py>(
    py: Python<'py>,
    old_pos: PyReadonlyArray2<f64>,
    new_pos: PyReadonlyArray2<f64>,
    cell: PyReadonlyArray2<f64>,
    pbc: PyReadonlyArray1<bool>,
    masses: PyReadonlyArray1<f64>,
    pairs: Vec<(usize, usize)>,
    bondlengths: PyReadonlyArray1<f64>,
    maxiter: usize,
    tolerance: f64,
    parallel: bool,
) -> (usize, Bound<'py, PyArray2<f64>>) {
    // Convert all Python arrays to owned Rust arrays before releasing the GIL
    let old_pos = old_pos.as_array().to_owned();
    let cell = cell.as_array().to_owned();
    let inv_cell = _inverse(cell.view());
    let pbc = pbc.as_array().to_owned();
    let masses = masses.as_array().to_owned();
    let bondlengths = bondlengths.as_array().to_owned();
    let new_pos = Arc::new(Mutex::new(new_pos.as_array().to_owned()));

    // Release the GIL and perform parallel computation
    let n_iter = py.allow_threads(|| {
        let mut iter_count = 0;
        'outer: for _ in 0..maxiter {
            let converged = AtomicBool::new(true);

            let closure = |(i, &(a, b))| {
                let d_fix = bondlengths[i];

                let mut new_pos_guard = new_pos.lock().unwrap();

                let r0 = &old_pos.row(a) - &old_pos.row(b);
                let d0 = find_mic(r0.view(), cell.view(), inv_cell.view(), pbc.view());

                let d1 = &new_pos_guard.row(a) - &new_pos_guard.row(b) - r0 + &d0;
                let m = 1.0 / (1.0 / masses[a] + 1.0 / masses[b]);

                let x = 0.5 * (d_fix.powf(2.0) - d1.dot(&d1)) / d0.dot(&d1);
                if x.abs() > tolerance {
                    let new_pos_a = &new_pos_guard.row(a) + &d0 * (x * m / masses[a]);
                    let new_pos_b = &new_pos_guard.row(b) - &d0 * (x * m / masses[b]);
                    new_pos_guard.row_mut(a).assign(&new_pos_a);
                    new_pos_guard.row_mut(b).assign(&new_pos_b);
                    converged.store(false, Ordering::Relaxed);
                }
            };
            // Use enumerate() to get the correct index
            match parallel {
                true => pairs
                    .par_iter()
                    .with_min_len(100)
                    .enumerate()
                    .for_each(closure),
                false => pairs.iter().enumerate().for_each(closure),
            }

            iter_count += 1;
            if converged.load(Ordering::Relaxed) {
                break 'outer;
            }
        }
        iter_count
    });

    // Get the final positions and convert back to Python array
    let final_positions = Arc::try_unwrap(new_pos).unwrap().into_inner().unwrap();

    (n_iter, final_positions.into_pyarray(py))
}

#[pyfunction]
pub fn adjust_momenta<'py>(
    py: Python<'py>,
    old_pos: PyReadonlyArray2<f64>,
    momenta: PyReadonlyArray2<f64>,
    cell: PyReadonlyArray2<f64>,
    pbc: PyReadonlyArray1<bool>,
    masses: PyReadonlyArray1<f64>,
    pairs: Vec<(usize, usize)>,
    bondlengths: PyReadonlyArray1<f64>,
    maxiter: usize,
    tolerance: f64,
    parallel: bool,
) -> (usize, Bound<'py, PyArray2<f64>>) {
    // Convert all Python arrays to owned Rust arrays
    let old_pos = old_pos.as_array().to_owned();
    let cell = cell.as_array().to_owned();
    let inv_cell = _inverse(cell.view());
    let pbc = pbc.as_array().to_owned();
    let masses = masses.as_array().to_owned();
    let bondlengths = bondlengths.as_array().to_owned();
    let momenta = Arc::new(Mutex::new(momenta.as_array().to_owned()));
    let inv_masses: Vec<f64> = masses.iter().map(|&m| 1.0 / m).collect();
    let bondlengths_sq: Vec<f64> = bondlengths.iter().map(|&b| b * b).collect();
    let reduced_masses: Vec<f64> = pairs
        .iter()
        .map(|&(a, b)| 1.0 / (inv_masses[a] + inv_masses[b]))
        .collect();

    // Release the GIL and perform parallel computation
    let n_iter = py.allow_threads(|| {
        let mut iter_count = 0;
        'outer: for _ in 0..maxiter {
            let converged = AtomicBool::new(true);
            let closure = |(i, &(a, b))| {
                let mut momenta_guard = momenta.lock().unwrap();

                let d = find_mic(
                    (&old_pos.row(a) - &old_pos.row(b)).view(),
                    cell.view(),
                    inv_cell.view(),
                    pbc.view(),
                );

                let dv =
                    &momenta_guard.row(a) * inv_masses[a] - &momenta_guard.row(b) * inv_masses[b];
                let x: f64 = -dv.dot(&d) / bondlengths_sq[i];

                if x.abs() > tolerance {
                    let new_momentum_a = &momenta_guard.row(a) + x * reduced_masses[i] * &d;
                    let new_momentum_b = &momenta_guard.row(b) - x * reduced_masses[i] * &d;
                    momenta_guard.row_mut(a).assign(&new_momentum_a);
                    momenta_guard.row_mut(b).assign(&new_momentum_b);
                    converged.store(false, Ordering::Relaxed);
                }
            };

            match parallel {
                true => pairs
                    .par_iter()
                    .with_min_len(100)
                    .enumerate()
                    .for_each(closure),
                false => pairs.iter().enumerate().for_each(closure),
            }

            if converged.load(Ordering::Relaxed) {
                break 'outer;
            }
            iter_count += 1;
        }
        iter_count
    });

    // Get the final momenta and convert back to Python array
    let final_momenta = Arc::try_unwrap(momenta).unwrap().into_inner().unwrap();

    (n_iter, final_momenta.into_pyarray(py))
}
