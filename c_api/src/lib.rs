extern crate libc;
extern crate hrbf;
extern crate nalgebra as na;
use libc::{size_t, c_int, c_double};
use na::{
    Point3,
    DimName,
    VectorN,
    Point,
    U3,
    DefaultAllocator
};
use na::allocator::Allocator;
use hrbf::{
    Pow3HRBF,
    Pow5HRBF,
    GaussHRBF,
    Csrbf31HRBF,
    Csrbf42HRBF,
    HRBFTrait,
};

/// Create an HRBF field object. At this stage, the field is zero. You need to fit it to some data
/// using `HRBF_fit` to produce a useful field. 
///
///  * `num_sites` is the number of sites needed to represent this HRBF.
///  * `sites_ptr` is the pointer to a contiguous array of 3 dimensional vectors of `double`s. the
///  size of this array must be `3*num_sites`.
///  * `kern_type` is an integer from 0 to 4 representing a particular kernel to use for fitting.
///  Use:
///     `0` for the cubic `x^3` kernel, (Default)
///     `1` for the quintic `x^5` kernel,
///     `2` for Gaussian  `exp(-x^2)` kernel,
///     `3` for CSRBF31 kernel: `(1-x)^4 (4x + 1)` and
///     `4` for CSRBF42 kernel: `(1-x)^6 (35x^2 + 18x + 3)`.
#[no_mangle]
pub unsafe extern "C" fn HRBF_create(num_sites: size_t,
                              sites_ptr: *const c_double,
                              kern_type: c_int) -> *mut HRBFTrait<f64>
{
    let sites = ptr_to_vec_of_points::<f64, U3>(num_sites, sites_ptr);

    let hrbf: Box<HRBFTrait<f64>> =
        match kern_type {
            1 => Box::new(Pow5HRBF::new(sites)),
            2 => Box::new(GaussHRBF::new(sites)),
            3 => Box::new(Csrbf31HRBF::new(sites)),
            4 => Box::new(Csrbf42HRBF::new(sites)),
            _ => Box::new(Pow3HRBF::new(sites)),
        };
    Box::into_raw(hrbf)
}

/// Free memory allocated by HRBF_create.
#[no_mangle]
pub unsafe extern "C" fn HRBF_free(hrbf: *mut HRBFTrait<f64>) {
    let _ = Box::from_raw(hrbf);
}

/// Given an HRBF object, fit it to the given set of points and normals.
/// NOTE: Currently this function only works for the same number of points as there are sites.
#[no_mangle]
pub unsafe extern "C" fn HRBF_fit(hrbf: *mut HRBFTrait<f64>,
                           num_pts: size_t,
                           pts_ptr: *const c_double,
                           nmls_ptr: *const c_double) -> bool {
    let pts = ptr_to_vec_of_points::<f64, U3>(num_pts, pts_ptr);
    let nmls = ptr_to_vec_of_vectors::<f64, U3>(num_pts, nmls_ptr);
    (*hrbf).fit(&pts, &nmls)
}

/// Evaluate the given HRBF field at a particular point.
#[no_mangle]
pub unsafe extern "C" fn HRBF_eval(hrbf: *const HRBFTrait<f64>,
                            x: c_double,
                            y: c_double,
                            z: c_double) -> c_double
{
    (*hrbf).eval(Point3::new(x,y,z))
}

/// Evaluate the gradient of the given HRBF field at a particular point.
/// `out` is expected to be an array of 3 doubles representing a 3D vector.
#[no_mangle]
pub unsafe extern "C" fn HRBF_grad(hrbf: *const HRBFTrait<f64>,
                            x: c_double,
                            y: c_double,
                            z: c_double,
                            out: *mut c_double)
{
    let g = (*hrbf).grad(Point3::new(x,y,z));
    *out.offset(0) = g[0];
    *out.offset(1) = g[1];
    *out.offset(2) = g[2];
}

/// Evaluate the hessian of the given HRBF field at a particular point.
/// `out` is expected to be an array of 9 doubles representing a 3x3 matrix.
#[no_mangle]
pub unsafe extern "C" fn HRBF_hess(hrbf: *const HRBFTrait<f64>,
                            x: c_double,
                            y: c_double,
                            z: c_double,
                            out: *mut c_double)
{
    let h = (*hrbf).hess(Point3::new(x,y,z));
    for i in 0..3 {
        for j in 0..3 {
            *out.offset(3*i + j) = h[(i as usize,j as usize)];
        }
    }
}

/// Advanced. Build the system for fitting an HRBF function. Don't do the actual fitting,
/// simply return the necessary fit matrix A and right-hand-side b, such that $A^{-1}*b$
/// gives the hrbf coefficients.
///
///  * `num_sites` is the number of sites needed to represent this HRBF.
///  * `sites_ptr`` is the pointer to a contiguous array of 3 dimensional vectors of `double`s. The
///  size of this array must be `3*num_sites`.
///  * `num_pts` is the number of data points to fit to.
///  * `pts_ptr`` is the pointer to a contiguous array of 3 dimensional vectors of `double`s. The
///  size of this array must be `3*num_pts`. This array represents the point positions to fit to.
///  * `nmls_ptr`` is the pointer to a contiguous array of 3 dimensional vectors of `double`s. The
///  size of this array must be `3*num_pts`. This array represents the normals to fit to.
///  * `kern_type` is an integer from 0 to 4 representing a particular kernel to use for fitting.
///     Use:
///        `0` for the cubic `x^3` kernel, (Default)
///        `1` for the quintic `x^5` kernel,
///        `2` for Gaussian  `exp(-x^2)` kernel,
///        `3` for CSRBF31 kernel: `(1-x)^4 (4x + 1)` and
///        `4` for CSRBF42 kernel: `(1-x)^6 (35x^2 + 18x + 3)`.
///  * `A` output matrix expected to be a contiguous array of `4*num_sites*4*num_pts` doubles. This
///  array represents a `num_sites x num_pts` sized matrix and is stored in column major order.
///  * `b` output vector expected to be a contiguous array of `4*num_pts` doubles.
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn HRBF_fit_system(num_sites: size_t,
                                  sites_ptr: *const c_double,
                                  num_pts: size_t,
                                  pts_ptr: *const c_double,
                                  nmls_ptr: *const c_double,
                                  kern_type: c_int,
                                  A: *mut c_double,
                                  b: *mut c_double)
{
    let sites = ptr_to_vec_of_points::<f64, U3>(num_sites, sites_ptr);
    let pts = ptr_to_vec_of_points::<f64, U3>(num_pts, pts_ptr);
    let nmls = ptr_to_vec_of_vectors::<f64, U3>(num_pts, nmls_ptr);

    let hrbf: Box<HRBFTrait<f64>> =
        match kern_type {
            1 => Box::new(Pow5HRBF::new(sites)),
            2 => Box::new(GaussHRBF::new(sites)),
            3 => Box::new(Csrbf31HRBF::new(sites)),
            4 => Box::new(Csrbf42HRBF::new(sites)),
            _ => Box::new(Pow3HRBF::new(sites)),
        };

    let mut offsets = Vec::new();
    offsets.resize(num_pts, 0.0);

    let (A_na, b_na) = (*hrbf).fit_system(&pts, &offsets, &nmls);
    for col in 0..4*num_sites {
        for row in 0..4*num_pts {
            *A.offset((num_sites*col + row) as isize) = A_na[(row,col)];
        }
    }
    for i in 0..4*num_pts {
        *b.offset(i as isize) = b_na[i];
    }
}

/// Helper routines for converting C-style data to Rust-style data.
/// `n` is the number of vectors to output, which means that `data_ptr` must point to an array of
/// `n*D::dim()` doubles.
unsafe fn ptr_to_vec_of_vectors<T: na::Real, D: DimName>(n: size_t, data_ptr: *const T) -> Vec<VectorN<T, D>>
    where DefaultAllocator: Allocator<T, D>
{
    let mut data = Vec::with_capacity(n);
        for i in 0..n as isize {
            data.push(VectorN::from_fn(|r,_| *data_ptr.offset(3*i + r as isize)));
        }
    data
}

unsafe fn ptr_to_vec_of_points<T: na::Real, D: DimName>(n: size_t, data_ptr: *const T) -> Vec<Point<T, D>>
    where DefaultAllocator: Allocator<T, D>
{
    ptr_to_vec_of_vectors(n, data_ptr).into_iter().map(|v| Point::from_coordinates(v)).collect()
}
