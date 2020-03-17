use hrbf::{Real, Hrbf,
Csrbf31HrbfBuilder, Csrbf42HrbfBuilder, GaussHrbfBuilder, Pow3HrbfBuilder, Pow5HrbfBuilder, Kernel};
use libc::{c_double, size_t};
use na::allocator::Allocator;
use na::{DefaultAllocator, DimName, Point, Point3, Vector3, Matrix3, U3, VectorN};

/// The type of kernel (real valued function in 3D) to be used for computing the HRBF.
#[repr(C)]
#[allow(non_camel_case_types)]
pub enum KernelType {
    HRBF_POW3,
    HRBF_POW5,
    HRBF_GAUSS,
    HRBF_CSRBF31,
    HRBF_CSRBF42,
}

/// A C-like public interface for evaluating HRBFs.
pub trait HrbfInterface<T: Real> {
    /// Evaluate the HRBF at point `p`.
    fn eval(&self, p: Point3<T>) -> T;

    /// Gradient of the HRBF function at point `p`.
    fn grad(&self, p: Point3<T>) -> Vector3<T>;

    /// Compute the Hessian of the HRBF function.
    fn hess(&self, p: Point3<T>) -> Matrix3<T>;
}

impl<T, K> HrbfInterface<T> for Hrbf<T, K>
where
    T: Real,
    K: Kernel<T> + Clone + Default,
{
    fn eval(&self, p: Point3<T>) -> T {
        Hrbf::eval(self, p)
    }
    fn grad(&self, p: Point3<T>) -> Vector3<T> {
        Hrbf::grad(self, p)
    }
    fn hess(&self, p: Point3<T>) -> Matrix3<T> {
        Hrbf::hess(self, p)
    }
}

macro_rules! raw_hrbf {
    ($build:expr) => {
        $build.map(|x| Box::into_raw(Box::new(x))).unwrap_or(std::ptr::null_mut())
    }
}

/// Create an HRBF field object from a given set of sites and normals.
///
/// If the creation of HRBF fails, this function returns a null pointer.
///
/// This should be the most common constructor for the HRBF.
///
///  * `num_sites` is the number of sites needed to represent this HRBF.
///  * `sites_ptr` is the pointer to a contiguous array of 3 dimensional vectors of `double`s
///  representing HRBF sites defining the HRBF basis and the HRBF zero level set. The size of this
///  array must be `3*num_sites`.
///  * `nmls_ptr` is the pointer to a contiguous array of 3 dimensional vectors of `double`s
///  representing normals at the given sites defining the HRBF gradient at each site. The
///  size of this array must be `3*num_sites`.
///  * `kern_type` is an integer from 0 to 4 representing a particular kernel to use for fitting.
///  Use:
///     `HRBF_POW3` for the cubic `x^3` kernel, (Default)
///     `HRBF_POW5` for the quintic `x^5` kernel,
///     `HRBF_GAUSS` for Gaussian  `exp(-x^2)` kernel,
///     `HRBF_CSRBF31` for CSRBF31 kernel: `(1-x)^4 (4x + 1)` and
///     `HRBF_CSRBF42` for CSRBF42 kernel: `(1-x)^6 (35x^2 + 18x + 3)`.
#[no_mangle]
pub unsafe extern "C" fn HRBF_create_fit(
    num_sites: size_t,
    sites_ptr: *const c_double,
    nmls_ptr: *const c_double,
    kern_type: KernelType,
) -> *mut dyn HrbfInterface<f64> {
    let sites = ptr_to_vec_of_points::<f64, U3>(num_sites, sites_ptr);
    let nmls = ptr_to_vec_of_vectors::<f64, U3>(num_sites, nmls_ptr);

    match kern_type {
       KernelType::HRBF_POW3 => raw_hrbf!(Pow3HrbfBuilder::new(sites).normals(nmls).build()),
       KernelType::HRBF_POW5 => raw_hrbf!(Pow5HrbfBuilder::new(sites).normals(nmls).build()),
       KernelType::HRBF_GAUSS => raw_hrbf!(GaussHrbfBuilder::new(sites).normals(nmls).build()),
       KernelType::HRBF_CSRBF31 => raw_hrbf!(Csrbf31HrbfBuilder::new(sites).normals(nmls).build()),
       KernelType::HRBF_CSRBF42 => raw_hrbf!(Csrbf42HrbfBuilder::new(sites).normals(nmls).build()),
    }
}

/// Create an HRBF field object from a given set of
/// - "sites" defining the positions at which each HRBF kernel is evaluated,
/// - "points" defining the samples at which the HRBF is assumed to be zero (sampling the zero
/// level-set), and
/// - "normals" which define the HRBF gradient at each "point".
///
/// If the creation of HRBF fails, this function returns a null pointer.
///
/// It is often best to have "sites" and "points" close together.
///
///  * `num_sites` is the number of sites needed to represent this HRBF.
///  * `sites_ptr` is the pointer to a contiguous array of 3 dimensional vectors of `double`s
///  representing HRBF sites defining the HRBF basis. The size of this array must be `3*num_sites`.
///  * `pts_ptr` is the pointer to a contiguous array of 3 dimensional vectors of `double`s
///  representing sample where HRBF should be zero. The size of this array must be `3*num_sites`.
///  * `nmls_ptr` is the pointer to a contiguous array of 3 dimensional vectors of `double`s
///  representing normals at the given sites defining the HRBF gradient at each site. The
///  size of this array must be `3*num_sites`.
///  * `kern_type` is an integer from 0 to 4 representing a particular kernel to use for fitting.
///  Use:
///     `HRBF_POW3` for the cubic `x^3` kernel, (Default)
///     `HRBF_POW5` for the quintic `x^5` kernel,
///     `HRBF_GAUSS` for Gaussian  `exp(-x^2)` kernel,
///     `HRBF_CSRBF31` for CSRBF31 kernel: `(1-x)^4 (4x + 1)` and
///     `HRBF_CSRBF42` for CSRBF42 kernel: `(1-x)^6 (35x^2 + 18x + 3)`.
#[no_mangle]
pub unsafe extern "C" fn HRBF_create_fit_to_points(
    num_sites: size_t,
    sites_ptr: *const c_double,
    pts_ptr: *const c_double,
    nmls_ptr: *const c_double,
    kern_type: KernelType,
) -> *mut dyn HrbfInterface<f64> {
    let sites = ptr_to_vec_of_points::<f64, U3>(num_sites, sites_ptr);
    let pts = ptr_to_vec_of_points::<f64, U3>(num_sites, pts_ptr);
    let nmls = ptr_to_vec_of_vectors::<f64, U3>(num_sites, nmls_ptr);

    match kern_type {
       KernelType::HRBF_POW3 => raw_hrbf!(Pow3HrbfBuilder::new(sites).points(pts).normals(nmls).build()),
       KernelType::HRBF_POW5 => raw_hrbf!(Pow5HrbfBuilder::new(sites).points(pts).normals(nmls).build()),
       KernelType::HRBF_GAUSS => raw_hrbf!(GaussHrbfBuilder::new(sites).points(pts).normals(nmls).build()),
       KernelType::HRBF_CSRBF31 => raw_hrbf!(Csrbf31HrbfBuilder::new(sites).points(pts).normals(nmls).build()),
       KernelType::HRBF_CSRBF42 => raw_hrbf!(Csrbf42HrbfBuilder::new(sites).points(pts).normals(nmls).build()),
    }
}


/// Free memory allocated by one of `HRBF_create_fit` or `HRBF_create_fit_to_points`.
#[no_mangle]
pub unsafe extern "C" fn HRBF_free(hrbf: *mut dyn HrbfInterface<f64>) {
    let _ = Box::from_raw(hrbf);
}

/// (Advanced) Build the system for fitting an HRBF function. Don't do the actual fitting,
/// simply return the necessary fit matrix A and right-hand-side b, such that $A^{-1}*b$
/// gives the hrbf coefficients.
///
///  * `num_sites` is the number of sites needed to represent this HRBF.
///  * `sites_ptr`` is the pointer to a contiguous array of 3 dimensional vectors of `double`s. The
///  size of this array must be `3*num_sites`.
///  * `pts_ptr`` is the pointer to a contiguous array of 3 dimensional vectors of `double`s. The
///  size of this array must be `3*num_sites`. This array represents the point positions to fit to.
///  * `nmls_ptr`` is the pointer to a contiguous array of 3 dimensional vectors of `double`s. The
///  size of this array must be `3*num_sites`. This array represents the normals to fit to.
///  * `kern_type` is an integer from 0 to 4 representing a particular kernel to use for fitting.
///     Use:
///        `HRBF_POW3` for the cubic `x^3` kernel, (Default)
///        `HRBF_POW5` for the quintic `x^5` kernel,
///        `HRBF_GAUSS` for Gaussian  `exp(-x^2)` kernel,
///        `HRBF_CSRBF31 for CSRBF31 kernel: `(1-x)^4 (4x + 1)` and
///        `HRBF_CSRBF42 for CSRBF42 kernel: `(1-x)^6 (35x^2 + 18x + 3)`.
///  * `A` output matrix expected to be a contiguous array of `4*num_sites*4*num_sites` doubles. This
///  array represents a `num_sites x num_sites` sized matrix and is stored in column major order.
///  * `b` output vector expected to be a contiguous array of `4*num_sites` doubles.
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn HRBF_fit_system(
    num_sites: size_t,
    sites_ptr: *const c_double,
    pts_ptr: *const c_double,
    nmls_ptr: *const c_double,
    kern_type: KernelType,
    A: *mut c_double,
    b: *mut c_double,
) {
    let sites = ptr_to_vec_of_points::<f64, U3>(num_sites, sites_ptr);
    let pts = ptr_to_vec_of_points::<f64, U3>(num_sites, pts_ptr);
    let nmls = ptr_to_vec_of_vectors::<f64, U3>(num_sites, nmls_ptr);

    let (A_na, b_na) = match kern_type {
        KernelType::HRBF_POW3 => Box::new(Pow3HrbfBuilder::new(sites).points(pts).normals(nmls).build_system()),
        KernelType::HRBF_POW5 => Box::new(Pow5HrbfBuilder::new(sites).points(pts).normals(nmls).build_system()),
        KernelType::HRBF_GAUSS => Box::new(GaussHrbfBuilder::new(sites).points(pts).normals(nmls).build_system()),
        KernelType::HRBF_CSRBF31 => Box::new(Csrbf31HrbfBuilder::new(sites).points(pts).normals(nmls).build_system()),
        KernelType::HRBF_CSRBF42 => Box::new(Csrbf42HrbfBuilder::new(sites).points(pts).normals(nmls).build_system()),
    }.unwrap();

    for col in 0..4 * num_sites {
        for row in 0..4 * num_sites {
            *A.offset((num_sites * col + row) as isize) = A_na[(row, col)];
        }
    }
    for i in 0..4 * num_sites {
        *b.offset(i as isize) = b_na[i];
    }
}

/// Evaluate the given HRBF field at a particular point.
#[no_mangle]
pub unsafe extern "C" fn HRBF_eval(
    hrbf: *const dyn HrbfInterface<f64>,
    x: c_double,
    y: c_double,
    z: c_double,
) -> c_double {
    (*hrbf).eval(Point3::new(x, y, z))
}

/// Evaluate the gradient of the given HRBF field at a particular point.
/// `out` is expected to be an array of 3 doubles representing a 3D vector.
#[no_mangle]
pub unsafe extern "C" fn HRBF_grad(
    hrbf: *const dyn HrbfInterface<f64>,
    x: c_double,
    y: c_double,
    z: c_double,
    out: *mut c_double,
) {
    let g = (*hrbf).grad(Point3::new(x, y, z));
    *out.offset(0) = g[0];
    *out.offset(1) = g[1];
    *out.offset(2) = g[2];
}

/// Evaluate the hessian of the given HRBF field at a particular point.
/// `out` is expected to be an array of 9 doubles representing a 3x3 matrix.
#[no_mangle]
pub unsafe extern "C" fn HRBF_hess(
    hrbf: *const dyn HrbfInterface<f64>,
    x: c_double,
    y: c_double,
    z: c_double,
    out: *mut c_double,
) {
    let h = (*hrbf).hess(Point3::new(x, y, z));
    for i in 0..3 {
        for j in 0..3 {
            *out.offset(3 * i + j) = h[(i as usize, j as usize)];
        }
    }
}

/// Helper routines for converting C-style data to Rust-style data.
/// `n` is the number of vectors to output, which means that `data_ptr` must point to an array of
/// `n*D::dim()` doubles.
unsafe fn ptr_to_vec_of_vectors<T: na::RealField, D: DimName>(
    n: size_t,
    data_ptr: *const T,
) -> Vec<VectorN<T, D>>
where
    DefaultAllocator: Allocator<T, D>,
{
    let mut data = Vec::with_capacity(n);
    for i in 0..n as isize {
        data.push(VectorN::from_fn(|r, _| {
            *data_ptr.offset(3 * i + r as isize)
        }));
    }
    data
}

unsafe fn ptr_to_vec_of_points<T: na::RealField, D: DimName>(
    n: size_t,
    data_ptr: *const T,
) -> Vec<Point<T, D>>
where
    DefaultAllocator: Allocator<T, D>,
{
    ptr_to_vec_of_vectors(n, data_ptr)
        .into_iter()
        .map(|v| Point::from(v))
        .collect()
}
