extern crate hrbf;
extern crate nalgebra;
extern crate num_traits;
extern crate rand;
#[macro_use]
extern crate approx;

use hrbf::*;
use nalgebra::{Matrix3, Point3, Vector3};
use num_traits::Float;
use rand::prelude::*;

fn rel_compare(a: f64, b: f64) {
    assert_relative_eq!(a, b, max_relative = 1e-3, epsilon = 1e-11);
}

fn cube() -> (Vec<Point3<f64>>, Vec<Vector3<f64>>) {
    // Fit an hrbf surface to a unit box
    let pts = vec![
        // Corners of the box
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 1.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(1.0, 1.0, 1.0),
        // Extra vertices on box faces
        Point3::new(0.5, 0.5, 0.0),
        Point3::new(0.5, 0.5, 1.0),
        Point3::new(0.5, 0.0, 0.5),
        Point3::new(0.5, 1.0, 0.5),
        Point3::new(0.0, 0.5, 0.5),
        Point3::new(1.0, 0.5, 0.5),
    ];

    let a = 1.0f64 / 3.0.sqrt();
    let nmls = vec![
        // Corner normals
        Vector3::new(-a, -a, -a),
        Vector3::new(-a, -a, a),
        Vector3::new(-a, a, -a),
        Vector3::new(-a, a, a),
        Vector3::new(a, -a, -a),
        Vector3::new(a, -a, a),
        Vector3::new(a, a, -a),
        Vector3::new(a, a, a),
        // Side normals
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, -1.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(-1.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
    ];

    (pts, nmls)
}

/// Finite difference test
fn test_derivative_fd<F, K: Kernel<f64> + Default>(x: Point3<f64>, compare: F, order: usize)
where
    F: Fn(f64, f64),
{
    let (pts, nmls) = cube();

    let mut hrbf = HRBF::<f64, K>::new(pts.clone());
    assert!(hrbf.fit(&pts, &nmls));

    // We will test hrbf derivatives using central differencing away from zeros since autodiff
    // fails in these scenarios because it can't simplify expressions with division by zero.

    let dx = 1.0 / 8192.0;
    let basis = vec![Vector3::x(), Vector3::y(), Vector3::z()];
    let h: Vec<Vector3<f64>> = basis.into_iter().map(|x| x * (0.5 * dx)).collect();

    // central finite difference function
    let cdf = |x: Point3<f64>| {
        Vector3::new(0, 1, 2).map(|dir| hrbf.eval(x + h[dir]) - hrbf.eval(x - h[dir]))
    };
    let fdf = cdf(x); // finite difference approximation

    // central finite difference for second derivative
    let cddf_dir = |x: Point3<f64>, dir: usize| cdf(x + h[dir]) - cdf(x - h[dir]);
    let cddf =
        |x: Point3<f64>| Matrix3::from_columns(&[cddf_dir(x, 0), cddf_dir(x, 1), cddf_dir(x, 2)]);
    let fddf = cddf(x).transpose();

    if order > 0 {
        let df = hrbf.grad(x);
        for k in 0..3 {
            compare(fdf[k], dx * df[k]);
        }
    }

    if order > 1 {
        let ddf = hrbf.hess(x);
        for k in 0..3 {
            for l in 0..3 {
                compare(fddf[(l, k)], dx * dx * ddf[(l, k)]);
            }
        }
    }
}

fn test_hrbf_derivative_simple<K: Kernel<f64> + Default>(order: usize) {
    let test_pts = [
        Point3::new(0.5, 1.0, 0.5),
        Point3::new(0.1, 0.0, 0.0),
        Point3::new(0.0, 0.1, 0.0),
        Point3::new(0.0, 0.0, 0.1),
    ];

    for &x in test_pts.iter() {
        test_derivative_fd::<_, K>(x, rel_compare, order);
    }
}

fn test_hrbf_derivative_random<K: Kernel<f64> + Default>(order: usize) {
    use self::rand::distributions::Uniform;

    let mut rng: StdRng = SeedableRng::from_seed([3u8; 32]);
    let range = Uniform::new(-1.0, 1.0);
    for _ in 0..99 {
        let x = Point3::new(rng.sample(range), rng.sample(range), rng.sample(range));
        test_derivative_fd::<_, K>(x, rel_compare, order);
    }
}

fn test_hrbf_fit<K: Kernel<f64> + Default>() {
    let (pts, nmls) = cube();

    let mut hrbf = HRBF::<f64, K>::new(pts.clone());
    assert!(hrbf.fit(&pts, &nmls));

    for (p, n) in pts.into_iter().zip(nmls) {
        let p_u = p + 0.001 * n;
        let p_l = p - 0.001 * n;
        rel_compare(hrbf.eval(p), 0.0);
        assert!(hrbf.eval(p_l) < 0.0);
        assert!(0.0 < hrbf.eval(p_u));
        let g = hrbf.grad(p);
        rel_compare(g[0], n[0]);
        rel_compare(g[1], n[1]);
        rel_compare(g[2], n[2]);
    }
}

// NOTE: pow2 and pow4 kernels generate singular fit matrices when data and sites are coincident.
// It is perhaps still possible to use these for least squares fits. The nice thing about x^2 is
// that it's especially fast to compute so we leave it in the code for future testing.
//
// This is why pow2 and pow4 kernels aren't tested below.

/// Derivative tests
#[test]
fn pow3_derivative_test() {
    test_hrbf_derivative_simple::<Pow3<f64>>(1);
    test_hrbf_derivative_random::<Pow3<f64>>(1);
}

#[test]
fn pow5_derivative_test() {
    test_hrbf_derivative_simple::<Pow5<f64>>(2);
    test_hrbf_derivative_random::<Pow5<f64>>(2);
}

#[test]
fn gauss_derivative_test() {
    test_hrbf_derivative_simple::<Gauss<f64>>(2);
    test_hrbf_derivative_random::<Gauss<f64>>(2);
}

#[test]
fn csrbf31_derivative_test() {
    test_hrbf_derivative_simple::<Csrbf31<f64>>(1);
    test_hrbf_derivative_random::<Csrbf31<f64>>(1);
}

#[test]
fn csrbf42_derivative_test() {
    test_hrbf_derivative_simple::<Csrbf42<f64>>(2);
    test_hrbf_derivative_random::<Csrbf42<f64>>(2);
}

/// Fit tests. Check that each of the kernels produce a reasonable fit to the data.
#[test]
fn pow3_fit_test() {
    test_hrbf_fit::<Pow3<f64>>();
}

#[test]
fn pow5_fit_test() {
    test_hrbf_fit::<Pow5<f64>>();
}

#[test]
fn gauss_fit_test() {
    test_hrbf_fit::<Gauss<f64>>();
}

#[test]
fn csrbf31_fit_test() {
    test_hrbf_fit::<Csrbf31<f64>>();
}

#[test]
fn csrbf42_fit_test() {
    test_hrbf_fit::<Csrbf42<f64>>();
}
