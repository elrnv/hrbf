extern crate itertools;
extern crate libc;
extern crate nalgebra as na;
#[macro_use]
extern crate approx;
extern crate chrbf;
use chrbf::*;
use itertools::Itertools;
use libc::{c_double, size_t};
use na::{Point3, Vector3};

#[test]
fn test_hrbf_fit() {
    // Fit an hrbf surface to a unit box
    let pts: &[c_double] = &[
        // Corners of the box
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
        // Extra vertices on box faces
        0.5, 0.5, 0.0, 0.5, 0.5, 1.0, 0.5, 0.0, 0.5, 0.5, 1.0,
        0.5, 0.0, 0.5, 0.5, 1.0, 0.5, 0.5,
    ];

    let a: c_double = 1.0 / c_double::sqrt(3.0);
    let nmls: &[c_double] = &[
        // Corner normals
        -a, -a, -a, -a, -a, a, -a, a, -a, -a, a, a, a, -a, -a, a, -a, a, a,
        a, -a, a, a, a, // Side normals
        0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0,
        1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ];

    unsafe {
        let n: size_t = pts.len() / 3;
        let hrbf = HRBF_create(n, pts.as_ptr(), 0);
        assert!(HRBF_fit(hrbf, n, pts.as_ptr(), nmls.as_ptr()));

        for ((&p0, &p1, &p2), (&n0, &n1, &n2)) in pts.iter().tuples().zip(nmls.iter().tuples()) {
            let p = Point3::new(p0, p1, p2);
            let n = Vector3::new(n0, n1, n2);
            let p_u = p + n * 0.001;
            let p_l = p - n * 0.001;

            assert_relative_eq!(
                HRBF_eval(hrbf, p[0], p[1], p[2]),
                0.0,
                max_relative = 1e-6,
                epsilon = 1e-11
            );
            assert!(HRBF_eval(hrbf, p_l[0], p_l[1], p_l[2]) < 0.0);
            assert!(HRBF_eval(hrbf, p_u[0], p_u[1], p_u[2]) > 0.0);

            let mut g = [0.0f64; 3];
            HRBF_grad(hrbf, p[0], p[1], p[2], g.as_mut_ptr());
            assert_relative_eq!(g[0], n[0], max_relative = 1e-6, epsilon = 1e-11);
            assert_relative_eq!(g[1], n[1], max_relative = 1e-6, epsilon = 1e-11);
            assert_relative_eq!(g[2], n[2], max_relative = 1e-6, epsilon = 1e-11);
        }
    }
}
