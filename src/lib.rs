#![warn(missing_docs)]

//!
//! An implementation of Hermite Radial Basis Functions with higher order derivatives.
//!
//! # Overview
//!
//! Let `p: &[Point3<f64>]` be a slice of points in 3D space along with corresponding normal
//! vectors `n: &[Vector3<f64>]`. This library lets us construct a 3D potential field whose zero
//! level-set interpolates the points `p` and whose gradient is aligned with `n` at these points.
//!
//! Additionally `hrbf` implements first and second order derivatives of the resulting potential
//! field with respect to point positions `p` (i.e. the Jacobian and Hessian).
//!
//! This library uses [`nalgebra`](https://nalgebra.org/) for linear algebra computations and in
//! API icalls.
//!
//!
//! ## Examples
//!
//! Suppose that we have a function `fn cube() -> (Vec<Point3<f64>, Vec<Vector3<f64>>)`  that
//! samples a unit cube centered at (0.5, 0.5, 0.5) (see the integration tests for a specific
//! implementation), with points and corresponding normals. Then we can construct and use an HRBF
//! field as follows:
//!
//! ```
//! use hrbf::*;
//! use na::{Point3, Vector3};
//! # fn cube() -> (Vec<Point3<f64>>, Vec<Vector3<f64>>) {
//! #    // Fit an hrbf surface to a unit box
//! #    let pts = vec![
//! #        // Corners of the box
//! #        Point3::new(0.0, 0.0, 0.0),
//! #        Point3::new(0.0, 0.0, 1.0),
//! #        Point3::new(0.0, 1.0, 0.0),
//! #        Point3::new(0.0, 1.0, 1.0),
//! #        Point3::new(1.0, 0.0, 0.0),
//! #        Point3::new(1.0, 0.0, 1.0),
//! #        Point3::new(1.0, 1.0, 0.0),
//! #        Point3::new(1.0, 1.0, 1.0),
//! #        // Extra vertices on box faces
//! #        Point3::new(0.5, 0.5, 0.0),
//! #        Point3::new(0.5, 0.5, 1.0),
//! #        Point3::new(0.5, 0.0, 0.5),
//! #        Point3::new(0.5, 1.0, 0.5),
//! #        Point3::new(0.0, 0.5, 0.5),
//! #        Point3::new(1.0, 0.5, 0.5),
//! #    ];
//! #
//! #    let a = 1.0f64 / 3.0f64.sqrt();
//! #    let nmls = vec![
//! #        // Corner normals
//! #        Vector3::new(-a, -a, -a),
//! #        Vector3::new(-a, -a, a),
//! #        Vector3::new(-a, a, -a),
//! #        Vector3::new(-a, a, a),
//! #        Vector3::new(a, -a, -a),
//! #        Vector3::new(a, -a, a),
//! #        Vector3::new(a, a, -a),
//! #        Vector3::new(a, a, a),
//! #        // Side normals
//! #        Vector3::new(0.0, 0.0, -1.0),
//! #        Vector3::new(0.0, 0.0, 1.0),
//! #        Vector3::new(0.0, -1.0, 0.0),
//! #        Vector3::new(0.0, 1.0, 0.0),
//! #        Vector3::new(-1.0, 0.0, 0.0),
//! #        Vector3::new(1.0, 0.0, 0.0),
//! #    ];
//! #
//! #    (pts, nmls)
//! # }
//!
//! // Construct a sampling of a unit cube centered at (0.5, 0.5, 0.5).
//! let (points, normals) = cube();
//!
//! // Create a new HRBF field using the `x^3` kernel with samples located at `points`.
//! // Normals define the direction of the HRBF gradient field at `points`.
//! let hrbf = Pow3HrbfBuilder::new(points)
//!     .normals(normals)
//!     .build()
//!     .unwrap();
//!
//! // The HRBF potential can then be queried at any 3D location as follows:
//! let mut query_point = Point3::new(0.1_f64, 0.2, 0.3);
//!
//! // Inside the cube the potential is negative.
//! assert!(hrbf.eval(query_point) < 0.0);
//!
//! // Outside it is positive.
//! query_point.z = 1.3;
//! assert!(hrbf.eval(query_point) > 0.0);
//!
//! // The gradient of the HRBF potential can also be queried.
//!
//! // We expect the gradient to point outward from the cube center.
//! let direction = Vector3::new(1.0, 1.0, 1.0);
//! assert!(hrbf.grad(query_point).dot(&direction) > 0.0);
//!
//! query_point.z = 0.3;
//! assert!(hrbf.grad(query_point).dot(&direction) < 0.0);
//! ```
//!
//! ## What is the difference between "points" and "sites" when creating an HRBF field?
//!
//! It may seem surprising that we can specify "points" and "sites" as two distinct sets. So why do
//! we need to ever specify "points" in [`HrbfBuilder`](struct.HrbfBuilder.html) if we have already
//! passed a set of "sites" in [`HrbfBuilder::new`](struct.HrbfBuilder.html#method.new)?
//!
//! The reason is simply that "sites" and "points" have a different purpose.  The set of "sites"
//! (the set of points passed to [`HrbfBuilder::new`](struct.HrbfBuilder.html#method.new)) are used
//! to evaluate the HRBF potential, these are the basis for the potential
//! field.  On the other hand, "points" are the 3D positions that define where the zero level-set
//! (or an offset level-set if "offsets" are specified) of the HRBF potential field goes.  However,
//! currently there is a restriction in the implementation that "sites" and "points" must have the
//! same size. Additionally, the closer "points" are to "sites", the better quality the resulting
//! HRBF potential will be.
//!
//!
//! # Related Publications
//!
//! The following publications introduce and analyse Hermite Radial Basis Functions and describe
//! different uses for approximating scattered point data:
//!
//! [I. Macêdo, J. P. Gois, and L. Velho, "*Hermite Radial Basis Function
//! Implicits*"](https://doi.org/10.1111/j.1467-8659.2010.01785.x)
//!
//! [R. Vaillant, L. Barthe, G. Guennebaud, M.-P. Cani, D. Rhomer, B. Wyvill, O. Gourmel, and M.
//! Paulin, "*Implicit Skinning: Real-Time Skin Deformation with Contact
//! Modeling*"](http://rodolphe-vaillant.fr/pivotx/templates/projects/implicit_skinning/implicit_skinning.pdf)
//!
pub mod kernel;

pub use kernel::*;
use na::storage::Storage;
use na::{
    DMatrix, DVector, Matrix3, Matrix3x4, Matrix4, Point3, RealField, Vector, Vector3, Vector4, U1,
    U3,
};
use num_traits::{Float, Zero};

/// Floating point real trait used throughout this library.
pub trait Real: Float + RealField + std::fmt::Debug {}
impl<T> Real for T where T: Float + RealField + std::fmt::Debug {}

/// Shorthand for an HRBF with a `x^3` kernel.
pub type Pow3Hrbf<T> = Hrbf<T, kernel::Pow3<T>>;
/// Shorthand for an HRBF with a `x^5` kernel.
pub type Pow5Hrbf<T> = Hrbf<T, kernel::Pow5<T>>;
/// Shorthand for an HRBF with a Gaussian `exp(-x*x)` kernel.
pub type GaussHrbf<T> = Hrbf<T, kernel::Gauss<T>>;
/// Shorthand for an HRBF with a CSRBF(3,1) `(1-x)^4 (4x+1)` kernel of type.
pub type Csrbf31Hrbf<T> = Hrbf<T, kernel::Csrbf31<T>>;
/// Shorthand for an HRBF with a CSRBF(4,1) `(1-x)^6 (35x^2 + 18x + 3)` kernel of type.
pub type Csrbf42Hrbf<T> = Hrbf<T, kernel::Csrbf42<T>>;

/// Shorthand for an HRBF builder with a `x^3` kernel.
pub type Pow3HrbfBuilder<T> = HrbfBuilder<T, kernel::Pow3<T>>;
/// Shorthand for an HRBF builder with a `x^5` kernel.
pub type Pow5HrbfBuilder<T> = HrbfBuilder<T, kernel::Pow5<T>>;
/// Shorthand for an HRBF builder with a Gaussian `exp(-x*x)` kernel.
pub type GaussHrbfBuilder<T> = HrbfBuilder<T, kernel::Gauss<T>>;
/// Shorthand for an HRBF builder with a CSRBF(3,1) `(1-x)^4 (4x+1)` kernel of type.
pub type Csrbf31HrbfBuilder<T> = HrbfBuilder<T, kernel::Csrbf31<T>>;
/// Shorthand for an HRBF builder with a CSRBF(4,1) `(1-x)^6 (35x^2 + 18x + 3)` kernel of type.
pub type Csrbf42HrbfBuilder<T> = HrbfBuilder<T, kernel::Csrbf42<T>>;

/// Error indicating that the building the HRBF potential failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The number of points does not match the number of sites.
    NumPointsMismatch,
    /// The number of offsets does not match the number of sites.
    NumOffsetsMismatch,
    /// The number of normalsdoes not match the number of sites.
    NumNormalsMismatch,
    /// The linear system solver responsible for fitting the HRBF to the given data failed.
    LinearSolveFailure,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NumPointsMismatch => "Number of points does not match the number of sites",
            Error::NumOffsetsMismatch => "Number of offsets does not match the number of sites",
            Error::NumNormalsMismatch => "Number of normals does not match the number of sites",
            Error::LinearSolveFailure => "Linear solve failed when building the HRBF potential",
        }
        .fmt(f)
    }
}
impl std::error::Error for Error {}

/// A Result with a custom Error type encapsulating all possible failures in this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// HRBF specific kernel type. In general, we can assign a unique kernel to each HRBF site, or we
/// can use the same kernel for all points. This corresponds to Variable and Constant kernel types
/// respectively.
#[derive(Clone, Debug)]
pub enum KernelType<K> {
    /// Each site has its own kernel, although all kernels must have the same type.
    Variable(Vec<K>),
    /// Same kernel for all sites.
    Constant(K),
}

impl<K> ::std::ops::Index<usize> for KernelType<K> {
    type Output = K;

    fn index(&self, index: usize) -> &K {
        match *self {
            KernelType::Variable(ref ks) => &ks[index],
            KernelType::Constant(ref k) => k,
        }
    }
}

/// A builder for an HRBF potential.
///
/// This struct collects the data needed to build a complete HRBF potential. This includes a set of
/// `sites`, `points`, normals and optionally a custom `kernel`.
#[derive(Clone, Debug)]
pub struct HrbfBuilder<T, K>
where
    T: Real,
    K: Kernel<T>,
{
    sites: Vec<Point3<T>>,
    points: Vec<Point3<T>>,
    normals: Vec<Vector3<T>>,
    offsets: Vec<T>,
    kernel: KernelType<K>,
}

impl<T, K> HrbfBuilder<T, K>
where
    T: Real,
    K: Kernel<T> + Clone + Default,
{
    /// Construct an HRBF builder with a set of `sites`.
    ///
    /// `sites` define the points used by HRBF at which the kernel will be evaluated.  Typically it
    /// is recommended to use the sites colocated with, or closely approximating the `points`
    /// sampling the desired zero level-set of the HRBF field.
    ///
    /// `sites` also initializes the size of the HRBF. Additional data to the HRBF builder must
    /// have the same size as `sites`.
    ///
    /// `sites` also serve as the default sampling `points` used to interpolate the zero level-set
    /// of the HRBF. In other words, if `points` is not specified, `sites` will be used instead.
    pub fn new(sites: Vec<Point3<T>>) -> HrbfBuilder<T, K> {
        HrbfBuilder {
            sites,
            points: Vec::new(),
            normals: Vec::new(),
            offsets: Vec::new(),
            kernel: KernelType::Constant(K::default()),
        }
    }

    /// Specify a set of values indicating the potential to be intepolated at each corresponding
    /// point interpolated by the HRBF.
    ///
    /// If this is not specified, each point is expected to be
    /// interpolating the zero level-set of the HRBF. Effectively, this assigns the offset of the
    /// HRBF field potential from the sample points.
    ///
    /// Negative offsets would result in the HRBF zero level-set to be pushed outwards, while
    /// positive offsets would push it inwards assuming that the interior of the object is
    /// represented by the negative HRBF potential (it is also equally valid to reverse this
    /// convention).
    ///
    /// The number of offsets must be the same as the `sites` specified to `new`.
    pub fn offsets(&mut self, offsets: Vec<T>) -> &mut Self {
        self.offsets = offsets;
        self
    }

    /// A set of points intended to sample the surface, or the zero level-set of the HRBF
    /// potential.
    ///
    /// If `points` is the same as as `sites` specified in `new`, then this setting can be omitted.
    pub fn points(&mut self, points: Vec<Point3<T>>) -> &mut Self {
        self.points = points;
        self
    }

    /// A set of normal vectors setting the direction of the HRBF gradient at each of the `points`
    /// as specified by the `points` function or `sites` passsed into `new` if the `points`
    /// function is not called.
    pub fn normals(&mut self, normals: Vec<Vector3<T>>) -> &mut Self {
        self.normals = normals;
        self
    }

    /// Recall that the HRBF fit is done as
    ///
    /// ```verbatim
    /// ∑ⱼ ⎡  𝜙(𝑥ᵢ - 𝑥ⱼ)  ∇𝜙(𝑥ᵢ - 𝑥ⱼ)'⎤ ⎡ 𝛼ⱼ⎤ = ⎡ 0 ⎤
    ///    ⎣ ∇𝜙(𝑥ᵢ - 𝑥ⱼ) ∇∇𝜙(𝑥ᵢ - 𝑥ⱼ) ⎦ ⎣ 𝛽ⱼ⎦   ⎣ 𝑛ᵢ⎦
    /// ```
    ///
    /// for every HRBF site `i`, where the sum runs over HRBF sites `j`
    /// where `𝜙(𝑥) = 𝜑(||𝑥||)` for one of the basis kernels we define in
    /// [`kernel`](kernel/index.html)
    /// If we rewrite the equation above as
    ///
    /// ```verbatim
    /// ∑ⱼ Aⱼ(𝑥ᵢ)bⱼ = rᵢ
    /// ```
    ///
    /// this function returns the matrix Aⱼ(p).
    ///
    /// This is the symmetric 4x4 matrix block that is used to fit the HRBF coefficients.
    /// This is equivalent to stacking the vector from `eval_block` on top of the
    /// 3x4 matrix returned by `grad_block`. This function is more efficient than
    /// evaluating `eval_block` and `grad_block`.
    /// This is `[g ∇g]' = [𝜙 (∇𝜙)'; ∇𝜙 ∇(∇𝜙)']` in MATLAB notation.
    pub(crate) fn fit_block(
        sites: &[Point3<T>],
        kernel: &KernelType<K>,
        p: Point3<T>,
        j: usize,
    ) -> Matrix4<T> {
        let x = p - sites[j];
        let l = x.norm();
        let w = kernel[j].f(l);
        let g = Hrbf::grad_phi(kernel, x, l, j);
        let h = Hrbf::hess_phi(kernel, x, l, j);
        Matrix4::new(
            w,
            g[0],
            g[1],
            g[2],
            g[0],
            h[(0, 0)],
            h[(0, 1)],
            h[(0, 2)],
            g[1],
            h[(1, 0)],
            h[(1, 1)],
            h[(1, 2)],
            g[2],
            h[(2, 0)],
            h[(2, 1)],
            h[(2, 2)],
        )
    }

    /// Build the linear system given all the needed data.
    #[allow(non_snake_case)]
    pub(crate) fn fit_system(
        sites: &[Point3<T>],
        points: &[Point3<T>],
        offsets: &[T],
        normals: &[Vector3<T>],
        kernel: &KernelType<K>,
    ) -> (DMatrix<T>, DVector<T>) {
        let num_sites = sites.len();
        debug_assert_eq!(points.len(), num_sites);
        debug_assert_eq!(normals.len(), num_sites);
        debug_assert!(offsets.is_empty() || offsets.len() == num_sites);
        let rows = 4 * points.len();
        let cols = 4 * num_sites;
        let mut A = DMatrix::<T>::zeros(rows, cols);
        let mut b = DVector::<T>::zeros(rows);
        for (i, p) in points.iter().enumerate() {
            b[4 * i] = if offsets.is_empty() {
                T::zero()
            } else {
                offsets[i]
            };
            b.fixed_rows_mut::<3>(4 * i + 1).copy_from(&normals[i]);
            for j in 0..num_sites {
                A.fixed_view_mut::<4, 4>(4 * i, 4 * j)
                    .copy_from(&Self::fit_block(sites, &kernel, *p, j));
            }
        }
        (A, b)
    }

    /// (Advanced) Build the linear system that is solved to compute the actual HRBF without evaluating it.
    ///
    /// This function returns the fitting matrix `A` and corresponding right-hand-side `b`.
    /// `b` is a stacked vector of 4D vectors representing the desired HRBF potential
    /// and normal at data point `i`, so `A.inverse()*b` gives the `betas` (or weights)
    /// defining the HRBF potential.
    pub fn build_system(&self) -> Result<(DMatrix<T>, DVector<T>)> {
        let HrbfBuilder {
            sites,
            points,
            normals,
            offsets,
            kernel,
        } = self;

        let num_sites = sites.len();

        let points = if points.is_empty() {
            sites.as_slice()
        } else {
            points.as_slice()
        };

        if points.len() != num_sites {
            return Err(Error::NumPointsMismatch);
        }

        if !offsets.is_empty() || offsets.len() != num_sites {
            return Err(Error::NumOffsetsMismatch);
        }

        if normals.len() != num_sites {
            return Err(Error::NumNormalsMismatch);
        }

        Ok(Self::fit_system(
            &sites, &points, &offsets, &normals, &kernel,
        ))
    }

    /// A non-consuming builder.
    pub fn build(&self) -> Result<Hrbf<T, K>> {
        let HrbfBuilder {
            sites,
            points,
            normals,
            offsets,
            kernel,
        } = self;

        let mut hrbf = Hrbf {
            sites: sites.clone(),
            betas: Vec::new(),
            kernel: kernel.clone(),
        };

        let result = if offsets.is_empty() {
            if points.is_empty() {
                Hrbf::fit(&mut hrbf, &normals)
            } else {
                Hrbf::fit_to_points(&mut hrbf, &points, &normals)
            }
        } else {
            if points.is_empty() {
                Hrbf::offset_fit(&mut hrbf, &offsets, &normals)
            } else {
                Hrbf::offset_fit_to_points(&mut hrbf, &points, &offsets, &normals)
            }
        };

        match result {
            Ok(_) => Ok(hrbf),
            Err(e) => Err(e),
        }
    }
}

impl<T, K> HrbfBuilder<T, K>
where
    T: Real,
    K: Kernel<T> + LocalKernel<T>,
{
    /// Set the kernel radius to be `radius` for all sites.
    ///
    /// Note that this parameter is only valid for local kernel types like `Csrbf31`, `Csrbf42` and
    /// `Gauss`. The radius in the `Gauss` kernel specifies its standard deviation, while in
    /// `Csrbf31` and `Csrbf42`, radius is the support radius beyond which the kernel is zero
    /// valued.
    pub fn radius(mut self, radius: T) -> Self {
        self.kernel = KernelType::Constant(K::new(radius));
        self
    }

    /// Set the kernel radius for each site individually.
    ///
    /// Note that this parameter is only valid for local kernel types like `Csrbf31`, `Csrbf42` and
    /// `Gauss`. The radii for the `Gauss` kernel specify its standard deviation, while in
    /// `Csrbf31` and `Csrbf42`, the radii are the support radii beyond which the kernel is zero.
    pub fn radii(mut self, radii: Vec<T>) -> Self {
        self.kernel = KernelType::Variable(radii.into_iter().map(|r| K::new(r)).collect());
        self
    }
}

/// An HRBF potential field.
///
/// The field and its first and second derivatives can be queried at any 3D position.
/// Additionally this field can be reset with a different set of points and normals using one of
/// the `fit`, `fit_to_points`, `offset_fit`, or `offset_fit_to_points` methods. Some advanced
/// functionality is also exposed (see individual methods for details).
#[derive(Clone, Debug)]
pub struct Hrbf<T, K>
where
    T: Real,
    K: Kernel<T>,
{
    sites: Vec<Point3<T>>,
    betas: Vec<Vector4<T>>,
    kernel: KernelType<K>,
}

impl<T, K> Hrbf<T, K>
where
    T: Real,
    K: Kernel<T> + Clone + Default,
{
    /// Returns a reference to the vector of site locations used by this HRBF.
    pub fn sites(&self) -> &[Point3<T>] {
        &self.sites
    }

    /// (Advanced) Returns a reference to the vector of 4D weight vectors, which determine the
    /// global HRBF potential.
    ///
    /// These are the unknowns computed during fitting.  Each 4D vector has
    /// the structure `[aⱼ; bⱼ]` per site `j` where `a` is a scalar weighing the contribution from
    /// the kernel at site `j` and `b` is a 3D vector weighin the contribution from the kernel
    /// gradient at site `j` to the total HRBF potential.
    pub fn betas(&self) -> &[Vector4<T>] {
        &self.betas
    }

    /// Fit the current HRBF to the `sites`, with which this HRBF was built.
    ///
    /// Normals dictate the direction of the HRBF gradient at the specified `sites`.
    /// Return a mutable reference to `Self` if successful.
    #[allow(non_snake_case)]
    pub fn fit(&mut self, normals: &[Vector3<T>]) -> Result<&mut Self> {
        self.fit_impl(None, None, normals)
    }

    /// Fit the current HRBF to the given data.
    ///
    /// Return a mutable reference to `Self` if successful.
    /// NOTE: Currently, points must be the same size as sites.
    #[allow(non_snake_case)]
    pub fn fit_to_points(
        &mut self,
        points: &[Point3<T>],
        normals: &[Vector3<T>],
    ) -> Result<&mut Self> {
        self.fit_impl(points.into(), None, normals)
    }

    /// Fit the current HRBF to the `sites`, with which this HRBF was built, offset by the given
    /// `offsets`.
    ///
    /// The resulting HRBF field is equal to `offsets` at the `sites`.
    /// and has a gradient equal to `normals`.
    /// Return a mutable reference to `Self` if successful.
    #[allow(non_snake_case)]
    pub fn offset_fit(&mut self, offsets: &[T], normals: &[Vector3<T>]) -> Result<&mut Self> {
        self.fit_impl(None, offsets.into(), normals)
    }

    /// Fit the current HRBF to the given data.
    ///
    /// The resulting HRBF field is equal to `offsets` at the provided `points`
    /// and has a gradient equal to `normals`.
    /// Return a mutable reference to `Self` if successful.
    /// NOTE: Currently, points must be the same size as sites.
    #[allow(non_snake_case)]
    pub fn offset_fit_to_points(
        &mut self,
        points: &[Point3<T>],
        offsets: &[T],
        normals: &[Vector3<T>],
    ) -> Result<&mut Self> {
        self.fit_impl(points.into(), offsets.into(), normals)
    }

    /// Implementation of the fitting algorithm.
    ///
    /// Return a mutable reference to `Self` if the computation is successful.
    #[allow(non_snake_case)]
    fn fit_impl(
        &mut self,
        points: Option<&[Point3<T>]>,
        offsets: Option<&[T]>,
        normals: &[Vector3<T>],
    ) -> Result<&mut Self> {
        let num_sites = self.sites.len();

        let points = points.unwrap_or_else(|| self.sites.as_slice());

        if points.len() != num_sites {
            return Err(Error::NumPointsMismatch);
        }

        if normals.len() != num_sites {
            return Err(Error::NumNormalsMismatch);
        }

        let mut potential = Vec::new();
        let offsets = offsets.unwrap_or_else(|| {
            potential.resize(num_sites, T::zero());
            potential.as_slice()
        });

        if offsets.len() != num_sites {
            return Err(Error::NumOffsetsMismatch);
        }

        let (A, b) = HrbfBuilder::fit_system(
            self.sites.as_slice(),
            points,
            offsets,
            normals,
            &self.kernel,
        );

        self.betas.clear();
        if let Some(x) = A.lu().solve(&b) {
            assert_eq!(x.len(), 4 * num_sites);

            self.betas.resize(num_sites, Vector4::zero());
            for j in 0..num_sites {
                self.betas[j].copy_from(&x.fixed_rows::<4>(4 * j));
            }

            Ok(self)
        } else {
            Err(Error::LinearSolveFailure)
        }
    }

    /// The following are derivatives of the function
    ///
    /// `phi(x) := kernel(|x|)`
    ///
    /// Given a vector `x` and its norm `l`, return the gradient of the kernel evaluated
    /// at `l` wrt `x`. `j` denotes the site at which the kernel is evaluated.
    fn grad_phi(kernel: &KernelType<K>, x: Vector3<T>, l: T, j: usize) -> Vector3<T> {
        x * kernel[j].df_l(l)
    }

    // TODO: The computations below do more than needed. For instance most computed
    // matrices are symmetric, if we can reformulate this formulas below into operations
    // on symmetric matrices, we can optimize a lot of flops out.

    /// Given a vector `x` and its norm `l`, return the hessian of the kernel evaluated
    /// at `l` wrt `x`. `j` denotes the site at which the kernel is evaluated.
    fn hess_phi(kernel: &KernelType<K>, x: Vector3<T>, l: T, j: usize) -> Matrix3<T> {
        let df_l = kernel[j].df_l(l);
        let mut hess = Matrix3::identity();
        if l <= T::zero() {
            debug_assert!({
                let g = kernel[j].ddf(l) - df_l;
                g * g < T::from(1e-12).unwrap()
            });
            return hess * df_l;
        }

        let ddf = kernel[j].ddf(l);
        let x_hat = x / l;
        // df_l*I + x_hat*x_hat.transpose()*(ddf - df_l)
        hess.ger(ddf - df_l, &x_hat, &x_hat, df_l);
        hess
    }

    /// Given a vector `x` and its norm `l`, return the third derivative of the kernel evaluated
    /// at `l` wrt `x` when multiplied by vector b.
    /// `j` denotes the site at which the kernel is evaluated.
    fn third_deriv_prod_phi<S>(
        &self,
        x: Vector3<T>,
        l: T,
        b: &Vector<T, U3, S>,
        j: usize,
    ) -> Matrix3<T>
    where
        S: Storage<T, U3, U1>,
    {
        if l <= T::zero() {
            debug_assert!({
                let g = self.kernel[j].g(l); // ddf(l)/l - df(l)/l^2
                let dddf = self.kernel[j].dddf(l);
                dddf == T::zero() && g == T::zero()
            });
            return Matrix3::zero();
        }

        let g = self.kernel[j].g(l); // ddf(l)/l - df(l)/l^2
        let dddf = self.kernel[j].dddf(l);
        let x_hat = x / l;
        let x_dot_b = b.dot(&x_hat);
        let mut mtx = Matrix3::identity();
        let _3 = T::from(3).unwrap();
        let _1 = T::one();

        // TODO: optimize this expression. we can probably achieve the same thing with less flops
        // (bxT + xTb*I + xbT)*g + xxT*((dddf - T::from(3).unwrap()*g)*xTb)
        mtx.ger(_1, b, &x_hat, x_dot_b);
        mtx.ger(_1, &x_hat, b, _1);
        mtx.ger((dddf - _3 * g) * x_dot_b, &x_hat, &x_hat, g);
        mtx
    }

    /// Given a vector `x` and its norm `l`, return the fourth derivative of the kernel
    /// evaluated at `l` wrt `x` when multiplied by vectors `b` and `c`.
    /// `j` denotes the site at which the kernel is evaluated.
    #[inline]
    fn fourth_deriv_prod_phi<S>(
        &self,
        x: Vector3<T>,
        l: T,
        b: &Vector<T, U3, S>,
        c: &Vector<T, U3, S>,
        j: usize,
    ) -> Matrix3<T>
    where
        S: Storage<T, U3, U1>,
    {
        let g_l = self.kernel[j].g_l(l);
        let bc_tr = Matrix3::new(
            b[0] * c[0],
            b[1] * c[0],
            b[2] * c[0],
            b[0] * c[1],
            b[1] * c[1],
            b[2] * c[1],
            b[0] * c[2],
            b[1] * c[2],
            b[2] * c[2],
        );
        let c_dot_b = bc_tr.trace();
        let bc_tr_plus_cb_tr = bc_tr + bc_tr.transpose();
        let mut res = Matrix3::identity();
        if l <= T::zero() {
            debug_assert!({
                let h3 = self.kernel[j].h(l, T::from(3).unwrap());
                let h52 = self.kernel[j].h(l, T::from(5.0 / 2.0).unwrap());
                let ddddf = self.kernel[j].ddddf(l);
                let a = ddddf - T::from(6.0).unwrap() * h52;
                h3 == T::zero() && a * a < T::from(1e-12).unwrap()
            });
            return res * (g_l * c_dot_b) + (bc_tr_plus_cb_tr) * g_l;
        }

        let h3 = self.kernel[j].h(l, T::from(3).unwrap());
        let h52 = self.kernel[j].h(l, T::from(5.0 / 2.0).unwrap());
        let ddddf = self.kernel[j].ddddf(l);
        let a = ddddf - T::from(6.0).unwrap() * h52;
        let x_hat = x / l;
        let x_dot_b = x_hat.dot(b);
        let x_dot_c = x_hat.dot(c);
        let cb_sum = c * x_dot_b + b * x_dot_c;
        let _1 = T::one();

        // TODO: optimize this expression. we can probably achieve the same thing with less flops
        //xxT*(a*xTb*xTc + h3*cTb)
        //    + I*(h3*xTc*xTb + g_l*cTb)
        //    + ((cxT + xcT)*xTb + (bxT + xbT)*xTc)*h3
        //    + (bcT + cbT)*g_l
        res.ger(
            a * x_dot_b * x_dot_c + h3 * c_dot_b,
            &x_hat,
            &x_hat,
            h3 * x_dot_c * x_dot_b + g_l * c_dot_b,
        );
        res.ger(h3, &cb_sum, &x_hat, _1);
        res.ger(h3, &x_hat, &cb_sum, _1);
        res + (bc_tr_plus_cb_tr) * g_l
    }

    /// Evaluate the HRBF at point `p`.
    pub fn eval(&self, p: Point3<T>) -> T {
        self.betas
            .iter()
            .enumerate()
            .fold(T::zero(), |sum, (j, b)| sum + self.eval_block(p, j).dot(b))
    }

    /// Helper function for `eval`.
    fn eval_block(&self, p: Point3<T>, j: usize) -> Vector4<T> {
        let x = p - self.sites[j];
        let l = x.norm();
        let w = self.kernel[j].f(l);
        let g = Self::grad_phi(&self.kernel, x, l, j);
        Vector4::new(w, g[0], g[1], g[2])
    }

    /// Gradient of the HRBF function at point `p`.
    pub fn grad(&self, p: Point3<T>) -> Vector3<T> {
        self.betas
            .iter()
            .enumerate()
            .fold(Vector3::zero(), |sum, (j, b)| {
                sum + self.grad_block(p, j) * b
            })
    }

    /// Helper function for `grad`. Returns a 3x4 matrix that gives the gradient of the HRBF when
    /// multiplied by the corresponding coefficients.
    fn grad_block(&self, p: Point3<T>, j: usize) -> Matrix3x4<T> {
        let x = p - self.sites[j];
        let l = x.norm();
        let h = Self::hess_phi(&self.kernel, x, l, j);
        let mut grad = Matrix3x4::zero();
        grad.column_mut(0)
            .copy_from(&Self::grad_phi(&self.kernel, x, l, j));
        grad.fixed_columns_mut::<3>(1).copy_from(&h);
        grad
    }

    /// Compute the Hessian of the HRBF function.
    pub fn hess(&self, p: Point3<T>) -> Matrix3<T> {
        self.betas
            .iter()
            .enumerate()
            .fold(Matrix3::zero(), |sum, (j, b)| {
                sum + self.hess_block_prod(p, b, j)
            })
    }

    /// Helper function for computing the hessian
    #[inline]
    fn hess_block_prod(&self, p: Point3<T>, b: &Vector4<T>, j: usize) -> Matrix3<T> {
        let x = p - self.sites[j];
        let l = x.norm();
        let b3 = b.fixed_rows::<3>(1);
        let h = Self::hess_phi(&self.kernel, x, l, j);
        h * b[0] + self.third_deriv_prod_phi(x, l, &b3, j)
    }

    /// (Advanced) Recall that the HRBF fit is done as
    ///
    /// ```verbatim
    /// ∑ⱼ ⎡  𝜙(𝑥ᵢ - 𝑥ⱼ)  ∇𝜙(𝑥ᵢ - 𝑥ⱼ)'⎤ ⎡ 𝛼ⱼ⎤ = ⎡ 0 ⎤
    ///    ⎣ ∇𝜙(𝑥ᵢ - 𝑥ⱼ) ∇∇𝜙(𝑥ᵢ - 𝑥ⱼ) ⎦ ⎣ 𝛽ⱼ⎦   ⎣ 𝑛ᵢ⎦
    /// ```
    ///
    /// for every HRBF site `i`, where the sum runs over HRBF sites `j`
    /// where `𝜙(𝑥) = 𝜑(||𝑥||)` for one of the basis kernels we define in
    /// [`kernel`](kernel/index.html)
    /// If we rewrite the equation above as
    ///
    /// ```verbatim
    /// ∑ⱼ Aⱼ(𝑥ᵢ)bⱼ = rᵢ
    /// ```
    ///
    /// this function returns the matrix Aⱼ(p).
    ///
    /// This is the symmetric 4x4 matrix block that is used to fit the HRBF coefficients.
    /// This is equivalent to stacking the vector from `eval_block` on top of the
    /// 3x4 matrix returned by `grad_block`. This function is more efficient than
    /// evaluating `eval_block` and `grad_block`.
    /// This is `[g ∇g]' = [𝜙 (∇𝜙)'; ∇𝜙 ∇(∇𝜙)']` in MATLAB notation.
    pub fn fit_block(&self, p: Point3<T>, j: usize) -> Matrix4<T> {
        HrbfBuilder::fit_block(self.sites.as_slice(), &self.kernel, p, j)
    }

    /// (Advanced) Using the same notation as above,
    /// this function returns the matrix `∇(Aⱼ(p)b)'`
    pub fn grad_fit_block_prod(&self, p: Point3<T>, b: Vector4<T>, j: usize) -> Matrix3x4<T> {
        let x = p - self.sites[j];
        let l = x.norm();

        let b3 = b.fixed_rows::<3>(1);

        let g = Self::grad_phi(&self.kernel, x, l, j);
        let h = Self::hess_phi(&self.kernel, x, l, j);

        let third = h * b[0] + self.third_deriv_prod_phi(x, l, &b3, j);

        let mut grad = Matrix3x4::zero();
        grad.column_mut(0).copy_from(&(g * b[0] + h * b3));
        grad.fixed_columns_mut::<3>(1).copy_from(&third);
        grad
    }

    /// Using the same notation as above,
    /// given a 4d vector lagrange multiplier `c`, this function returns the matrix
    ///
    /// ```verbatim
    /// ∇(∇(Aⱼ(p)βⱼ)'c)'
    /// ```
    ///
    /// where βⱼ are taken from `self.betas`
    fn hess_fit_prod_block(&self, p: Point3<T>, c: Vector4<T>, j: usize) -> Matrix3<T> {
        let x = p - self.sites[j];
        let l = x.norm();

        let c3 = c.fixed_rows::<3>(1);
        let a = self.betas[j][0];
        let b = self.betas[j].fixed_rows::<3>(1);

        // Compute in blocks
        Self::hess_phi(&self.kernel, x, l, j) * c[0] * a
            + self.third_deriv_prod_phi(x, l, &b, j) * c[0]
            + self.third_deriv_prod_phi(x, l, &c3, j) * a
            + self.fourth_deriv_prod_phi(x, l, &b, &c3, j)
    }

    /// Sum of hess_fit_prod_block evaluated at all sites.
    pub fn hess_fit_prod(&self, p: Point3<T>, c: Vector4<T>) -> Matrix3<T> {
        (0..self.sites.len()).fold(Matrix3::zero(), |sum, j| {
            sum + self.hess_fit_prod_block(p, c, j)
        })
    }
}
