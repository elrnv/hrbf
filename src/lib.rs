extern crate nalgebra as na;
extern crate num_traits;

pub mod kernel;

pub use kernel::*;
use na::storage::Storage;
use na::{
    DMatrix, DVector, Matrix3, Matrix3x4, Matrix4, Point3, U1, U3, U4, Vector, Vector3,
    Vector4,
};
use num_traits::{Float, Zero};

/// Floating point real trait used throughout this library.
pub trait Real: Float + na::Real + ::std::fmt::Debug {}
impl<T> Real for T where T: Float + na::Real + ::std::fmt::Debug {}

/// Shorthand for an HRBF with a constant `x^3` kernel.
pub type Pow3HRBF<T> = HRBF<T, kernel::Pow3<T>>;
/// Shorthand for an HRBF with a constant `x^5` kernel.
pub type Pow5HRBF<T> = HRBF<T, kernel::Pow5<T>>;
/// Shorthand for an HRBF with a constant Gaussian `exp(-x*x)` kernel.
pub type GaussHRBF<T> = HRBF<T, kernel::Gauss<T>>;
/// Shorthand for an HRBF with a constant CSRBF(3,1) `(1-x)^4 (4x+1)` kernel of type.
pub type Csrbf31HRBF<T> = HRBF<T, kernel::Csrbf31<T>>;
/// Shorthand for an HRBF with a constant CSRBF(4,1) `(1-x)^6 (35x^2 + 18x + 3)` kernel of type.
pub type Csrbf42HRBF<T> = HRBF<T, kernel::Csrbf42<T>>;

/// HRBF specific kernel type. In general, we can assign a unique kernel to each HRBF site, or we
/// can use the same kernel for all points. This corresponds to Variable and Constant kernel types
/// respectively.
#[derive(Clone, Debug)]
pub enum KernelType<K> {
    Variable(Vec<K>), // each site has its own kernel
    Constant(K),      // same kernel for all sites
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

#[derive(Clone, Debug)]
pub struct HRBF<T, K>
where
    T: Real,
    K: Kernel<T>,
{
    sites: Vec<Point3<T>>,
    betas: Vec<Vector4<T>>,
    kernel: KernelType<K>,
}

/// HRBF public interface. It is not necessary to use this trait, but it allows using the HRBF as a
/// trait object. For example this is used to pass HRBF objects between functions in the C API.
pub trait HRBFTrait<T: Real> {
    fn fit(&mut self, points: &[Point3<T>], normals: &[Vector3<T>]) -> bool;
    fn fit_offset(
        &mut self,
        points: &[Point3<T>],
        offsets: &[T],
        normals: &[Vector3<T>],
    ) -> bool;
    fn fit_system(
        &self,
        points: &[Point3<T>],
        potential: &[T],
        normals: &[Vector3<T>],
    ) -> (DMatrix<T>, DVector<T>);
    fn eval(&self, p: Point3<T>) -> T;
    fn grad(&self, p: Point3<T>) -> Vector3<T>;
    fn hess(&self, p: Point3<T>) -> Matrix3<T>;
}

impl<T, K> HRBFTrait<T> for HRBF<T, K>
where
    T: Real,
    K: Kernel<T> + Default,
{
    fn fit(&mut self, points: &[Point3<T>], normals: &[Vector3<T>]) -> bool {
        HRBF::fit(self, points, normals)
    }
    fn fit_offset(
        &mut self,
        points: &[Point3<T>],
        offsets: &[T],
        normals: &[Vector3<T>],
    ) -> bool {
        HRBF::fit_offset(self, points, offsets, normals)
    }
    fn fit_system(
        &self,
        points: &[Point3<T>],
        potential: &[T],
        normals: &[Vector3<T>],
    ) -> (DMatrix<T>, DVector<T>) {
        HRBF::fit_system(self, points, potential, normals)
    }
    fn eval(&self, p: Point3<T>) -> T {
        HRBF::eval(self, p)
    }
    fn grad(&self, p: Point3<T>) -> Vector3<T> {
        HRBF::grad(self, p)
    }
    fn hess(&self, p: Point3<T>) -> Matrix3<T> {
        HRBF::hess(self, p)
    }
}

impl<T, K> Default for HRBF<T, K>
where
    T: Real,
    K: Kernel<T> + Default,
{
    fn default() -> Self {
        HRBF {
            sites: Vec::new(),
            betas: Vec::new(),
            kernel: KernelType::Constant(K::default()),
        }
    }
}

impl<T, K> HRBF<T, K>
where
    T: Real,
    K: Kernel<T> + LocalKernel<T>,
{
    pub fn radius(mut self, radius: T) -> Self {
        self.kernel = KernelType::Constant(K::new(radius));
        self
    }

    pub fn radii(mut self, radii: Vec<T>) -> Self {
        self.kernel = KernelType::Variable(radii.into_iter().map(|r| K::new(r)).collect());
        self
    }
}

impl<T, K> HRBF<T, K>
where
    T: Real,
    K: Kernel<T> + Default,
{
    /// Main constructor. Assigns the degrees of freedom used by this HRBF in a form of 3D points
    /// at which the kernel will be evaluated.
    pub fn new(sites: Vec<Point3<T>>) -> Self {
        HRBF {
            sites,
            ..HRBF::default()
        }
    }

    /// Builder that assigns a particular kernel to this HRBF.
    pub fn with_kernel(self, kernel: KernelType<K>) -> Self {
        HRBF { kernel, ..self }
    }

    /// Returns a reference to the vector of site locations used by this HRBF.
    pub fn sites(&self) -> &Vec<Point3<T>> {
        &self.sites
    }

    /// Advanced. Returns a reference to the vector of 4D weight vectors, which determine the
    /// global HRBF potential. These are the unknowns computed during fitting.
    /// Each 4D vector has the structure `[aⱼ; bⱼ]` per site `j` where `a` is a scalar weighing
    /// the contribution from the kernel at site `j` and b is a 3D vector weighin the contribution
    /// from the kernel gradient at site `j` to the total HRBF potential.
    pub fn betas(&self) -> &Vec<Vector4<T>> {
        &self.betas
    }

    /// Fit the current HRBF to the given data. Return true if successful.
    /// NOTE: Currently, points must be the same size as as sites.
    #[allow(non_snake_case)]
    pub fn fit(&mut self, points: &[Point3<T>], normals: &[Vector3<T>]) -> bool {
        assert!(normals.len() == points.len());
        let num_sites = self.sites.len();

        let mut potential = Vec::new();
        potential.resize(num_sites, T::zero());
        let (A, b) = self.fit_system(points, &potential, normals);

        self.betas.clear();
        if let Some(x) = A.lu().solve(&b) {
            assert!(x.len() == 4 * num_sites);

            self.betas.resize(num_sites, Vector4::zero());
            for j in 0..num_sites {
                self.betas[j].copy_from(&x.fixed_rows::<U4>(4 * j));
            }

            true
        } else {
            false
        }
    }

    /// Fit the current HRBF to the given data. The resulting HRBF field is equal to `offsets`
    /// and has a gradient equal to `normals`.
    /// Return true if successful.
    /// NOTE: Currently, points must be the same size as as sites.
    #[allow(non_snake_case)]
    pub fn fit_offset(
        &mut self,
        points: &[Point3<T>],
        offsets: &[T],
        normals: &[Vector3<T>],
    ) -> bool {
        assert!(normals.len() == points.len());
        let num_sites = self.sites.len();

        let (A, b) = self.fit_system(points, offsets, normals);

        self.betas.clear();
        if let Some(x) = A.lu().solve(&b) {
            assert!(x.len() == 4 * num_sites);

            self.betas.resize(num_sites, Vector4::zero());
            for j in 0..num_sites {
                self.betas[j].copy_from(&x.fixed_rows::<U4>(4 * j));
            }

            true
        } else {
            false
        }
    }

    /// Advanced. Returns the fitting matrix `A` and corresponding right-hand-side `b`.
    /// `b` is a stacked vector of 4D vectors representing the desired HRBF potential
    /// and normal at data point `i`, so `A.inverse()*b` gives the `betas` (or weights)
    /// defining the HRBF potential.
    #[allow(non_snake_case)]
    pub fn fit_system(
        &self,
        points: &[Point3<T>],
        potential: &[T],
        normals: &[Vector3<T>],
    ) -> (DMatrix<T>, DVector<T>) {
        assert!(normals.len() == points.len());
        assert!(potential.len() == points.len());
        let rows = 4 * points.len();
        let num_sites = self.sites.len();
        let cols = 4 * num_sites;
        let mut A = DMatrix::<T>::zeros(rows, cols);
        let mut b = DVector::<T>::zeros(rows);
        for (i, p) in points.iter().enumerate() {
            b[4 * i] = potential[i];
            b.fixed_rows_mut::<U3>(4 * i + 1).copy_from(&normals[i]);
            for j in 0..num_sites {
                A.fixed_slice_mut::<U4, U4>(4 * i, 4 * j)
                    .copy_from(&self.fit_block(*p, j));
            }
        }
        (A, b)
    }

    /// The following are derivatives of the function
    ///
    /// `phi(x) := kernel(|x|)`
    ///
    /// Given a vector `x` and its norm `l`, return the gradient of the kernel evaluated
    /// at `l` wrt `x`. `j` denotes the site at which the kernel is evaluated.
    fn grad_phi(&self, x: Vector3<T>, l: T, j: usize) -> Vector3<T> {
        x * self.kernel[j].df_l(l)
    }

    // TODO: The computations below do more than needed. For instance most computed
    // matrices are symmetric, if we can reformulate this formulas below into operations
    // on symmetric matrices, we can optimize a lot of flops out.

    /// Given a vector `x` and its norm `l`, return the hessian of the kernel evaluated
    /// at `l` wrt `x`. `j` denotes the site at which the kernel is evaluated.
    fn hess_phi(&self, x: Vector3<T>, l: T, j: usize) -> Matrix3<T> {
        let df_l = self.kernel[j].df_l(l);
        let mut hess = Matrix3::identity();
        if l <= T::zero() {
            debug_assert!({
                let g = self.kernel[j].ddf(l) - df_l;
                g * g < T::from(1e-12).unwrap()
            });
            return hess * df_l;
        }

        let ddf = self.kernel[j].ddf(l);
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
    /// evaluated at `l` wrt `x` when multiplied by vectors b and c.
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
        let g = self.grad_phi(x, l, j);
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
        let h = self.hess_phi(x, l, j);
        let mut grad = Matrix3x4::zero();
        grad.column_mut(0).copy_from(&self.grad_phi(x, l, j));
        grad.fixed_columns_mut::<U3>(1).copy_from(&h);
        grad
    }

    /// Compute the hessian of the HRBF function.
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
        let b3 = b.fixed_rows::<U3>(1);
        let h = self.hess_phi(x, l, j);
        h * b[0] + self.third_deriv_prod_phi(x, l, &b3, j)
    }

    /// Advanced. Recall that the HRBF fit is done as
    ///
    /// ```verbatim
    /// ∑ⱼ ⎡  𝜙(𝑥ᵢ - 𝑥ⱼ)  ∇𝜙(𝑥ᵢ - 𝑥ⱼ)'⎤ ⎡ 𝛼ⱼ⎤ = ⎡ 0 ⎤
    ///    ⎣ ∇𝜙(𝑥ᵢ - 𝑥ⱼ) ∇∇𝜙(𝑥ᵢ - 𝑥ⱼ) ⎦ ⎣ 𝛽ⱼ⎦   ⎣ 𝑛ᵢ⎦
    /// ```
    ///
    /// for every HRBF site i, where the sum runs over HRBF sites j
    /// where 𝜙(𝑥) = 𝜑(||𝑥||) for one of the basis kernels we define in kernel.rs
    /// If we rewrite the equation above as
    ///
    /// ` ∑ⱼ Aⱼ(𝑥ᵢ)bⱼ = rᵢ `
    ///
    /// this function returns the matrix Aⱼ(p).
    ///
    /// This is the symmetric 4x4 matrix block that is used to fit the HRBF coefficients.
    /// This is equivalent to stacking the vector from `eval_block` on top of the
    /// 3x4 matrix returned by `grad_block`. This function is more efficient than
    /// evaluating `eval_block` and `grad_block`.
    /// This is [g ∇g]' = [𝜙 (∇𝜙)'; ∇𝜙 ∇(∇𝜙)'] in matlab notation.
    pub fn fit_block(&self, p: Point3<T>, j: usize) -> Matrix4<T> {
        let x = p - self.sites[j];
        let l = x.norm();
        let w = self.kernel[j].f(l);
        let g = self.grad_phi(x, l, j);
        let h = self.hess_phi(x, l, j);
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

    /// Advanced. Using the same notation as above,
    /// this function returns the matrix ∇(Aⱼ(p)b)'
    pub fn grad_fit_block_prod(&self, p: Point3<T>, b: Vector4<T>, j: usize) -> Matrix3x4<T> {
        let x = p - self.sites[j];
        let l = x.norm();

        let b3 = b.fixed_rows::<U3>(1);

        let g = self.grad_phi(x, l, j);
        let h = self.hess_phi(x, l, j);

        let third = h * b[0] + self.third_deriv_prod_phi(x, l, &b3, j);

        let mut grad = Matrix3x4::zero();
        grad.column_mut(0).copy_from(&(g * b[0] + h * b3));
        grad.fixed_columns_mut::<U3>(1).copy_from(&third);
        grad
    }

    /// Using the same notation as above,
    /// given a 4d vector lagrange multiplier `c`, this function returns the matrix
    ///
    /// ` ∇(∇(Aⱼ(p)βⱼ)'c)' `
    ///
    /// where βⱼ are taken from `self.betas`
    fn hess_fit_prod_block(&self, p: Point3<T>, c: Vector4<T>, j: usize) -> Matrix3<T> {
        let x = p - self.sites[j];
        let l = x.norm();

        let c3 = c.fixed_rows::<U3>(1);
        let a = self.betas[j][0];
        let b = self.betas[j].fixed_rows::<U3>(1);

        // Compute in blocks
        self.hess_phi(x, l, j) * c[0] * a
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
