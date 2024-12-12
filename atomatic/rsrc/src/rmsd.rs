use nalgebra::Matrix4;
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// FFI function
#[pyfunction]
#[pyo3(signature = (positions_1, positions_2, compute_gradient=false))]
pub fn compute_rmsd<'py>(
    _py: Python<'py>,
    positions_1: PyReadonlyArray2<f64>,
    positions_2: PyReadonlyArray2<f64>,
    compute_gradient: bool,
) -> (f64, Option<Bound<'py, PyArray2<f64>>>) {
    let positions_1 = positions_1.as_array();
    let positions_2 = positions_2.as_array();
    let result = _compute_minimum_rmsd(&positions_1, &positions_2, compute_gradient);
    let rmsd_grad = result.rmsd_grad.map(|x| x.into_pyarray(_py));
    (result.rmsd_val, rmsd_grad)
}

/// Implementation
pub struct RMSDResult {
    pub rmsd_val: f64,
    pub rmsd_grad: Option<Array2<f64>>,
    pub rotation_matrix: Array2<f64>,
    pub translation_vector: Array1<f64>,
}

fn rmsd(x: &ArrayView2<f64>, y: &ArrayView2<f64>) -> f64 {
    let delta_r = x - y;
    (delta_r.mapv(|x| x * x).sum_axis(Axis(1)).mean().unwrap()).sqrt()
}

fn centroid(pos: &ArrayView2<f64>) -> Array1<f64> {
    pos.mean_axis(Axis(0)).unwrap()
}

/// Computes minimum root mean square deviation (RMSD) between two sets of points.
/// The optimal rotation and translation to minimize the RMSD are computed by
/// quaternion approach.
/// Gradient of the RMSD with respect pos_1 is also computed, which is often required
/// for optimization problems.
#[allow(non_snake_case)]
pub fn _compute_minimum_rmsd(
    pos_1: &ArrayView2<f64>,
    pos_2: &ArrayView2<f64>,
    compute_grad: bool,
) -> RMSDResult {
    let mut pos_1 = pos_1.to_owned();
    let mut pos_2 = pos_2.to_owned();
    let centroid_1 = centroid(&pos_1.view());
    let centroid_2 = centroid(&pos_2.view());
    pos_1 -= &centroid_1;
    pos_2 -= &centroid_2;
    let U = find_rotation_matrix(&pos_1.view(), &pos_2.view());
    let pos_2_rotated = pos_2.dot(&U);
    let rmsd_val = rmsd(&pos_1.view(), &pos_2_rotated.view());
    let (rmsd_val, rmsd_grad) = match compute_grad {
        true => {
            let n = pos_1.shape()[0] as f64;
            let mut rmsd_grad = Array2::zeros(pos_1.raw_dim());
            for (i, mut row) in rmsd_grad.axis_iter_mut(Axis(0)).enumerate() {
                let delta_r = &pos_1.row(i) - &pos_2_rotated.row(i);
                row += &(&delta_r / (n * rmsd_val));
            }
            (rmsd_val, Some(rmsd_grad))
        }
        false => (rmsd_val, None),
    };
    let c = centroid_1 - centroid_2.dot(&U);
    RMSDResult {
        rmsd_val,
        rmsd_grad,
        rotation_matrix: U,
        translation_vector: c,
    }
}

/// Find the rotation matrix U that minimizes the RMSD between two sets of
/// positions. Assume that the two sets of positions have the same number of
/// atoms, and centroids are already subtracted.
#[allow(non_snake_case)]
fn find_rotation_matrix(pos_1: &ArrayView2<f64>, pos2: &ArrayView2<f64>) -> Array2<f64> {
    let R = pos_1.t().dot(pos2);
    let R11 = R[[0, 0]];
    let R12 = R[[0, 1]];
    let R13 = R[[0, 2]];
    let R21 = R[[1, 0]];
    let R22 = R[[1, 1]];
    let R23 = R[[1, 2]];
    let R31 = R[[2, 0]];
    let R32 = R[[2, 1]];
    let R33 = R[[2, 2]];

    #[rustfmt::skip]
    let F = Matrix4::new(
        R11 + R22 + R33,       R23 - R32,        R31 - R13,        R12 - R21,
              R23 - R32, R11 - R22 - R33,        R12 + R21,        R13 + R31,
              R31 - R13,       R12 + R21, -R11 + R22 - R33,        R23 + R32,
              R12 - R21,       R13 + R31,        R23 + R32, -R11 - R22 + R33,
    );

    let eig = F.symmetric_eigen();
    let eigvals = eig.eigenvalues;
    let eigvecs = eig.eigenvectors;
    // find the largest eigenvalue and its corresponding eigenvector
    let mut max_eigval = 0.0;
    let mut q = Array1::<f64>::zeros((4,));
    for i in 0..4 {
        if eigvals[i] > max_eigval {
            max_eigval = eigvals[i];
            q[0] = eigvecs[(0, i)];
            q[1] = eigvecs[(1, i)];
            q[2] = eigvecs[(2, i)];
            q[3] = eigvecs[(3, i)];
        }
    }
    let U = quaternion_to_rotation_matrix(&q.view());
    U
}

/// Convert a quaternion to a rotation matrix U.
#[allow(non_snake_case)]
fn quaternion_to_rotation_matrix(q: &ArrayView1<f64>) -> Array2<f64> {
    let q0 = q[0];
    let q1 = q[1];
    let q2 = q[2];
    let q3 = q[3];
    let q0_2 = q0 * q0;
    let q1_2 = q1 * q1;
    let q2_2 = q2 * q2;
    let q3_2 = q3 * q3;
    let q0q1 = q0 * q1;
    let q0q2 = q0 * q2;
    let q0q3 = q0 * q3;
    let q1q2 = q1 * q2;
    let q1q3 = q1 * q3;
    let q2q3 = q2 * q3;

    let mut U = Array2::<f64>::zeros((3, 3));
    U[[0, 0]] = q0_2 + q1_2 - q2_2 - q3_2;
    U[[0, 1]] = 2.0 * (q1q2 - q0q3);
    U[[0, 2]] = 2.0 * (q1q3 + q0q2);
    U[[1, 0]] = 2.0 * (q1q2 + q0q3);
    U[[1, 1]] = q0_2 - q1_2 + q2_2 - q3_2;
    U[[1, 2]] = 2.0 * (q2q3 - q0q1);
    U[[2, 0]] = 2.0 * (q1q3 - q0q2);
    U[[2, 1]] = 2.0 * (q2q3 + q0q1);
    U[[2, 2]] = q0_2 - q1_2 - q2_2 + q3_2;

    U
}
