extern crate num_traits;
extern crate nalgebra;

pub mod kernel;

pub trait Real: nalgebra::Real + num_traits::Float + ::std::fmt::Debug {}

use nalgebra::{Point3, Vector4, Vector3, DVector, Vector, Matrix3, Matrix4, Matrix3x4, DMatrix, U1, U3, U4, norm};
use nalgebra::storage::Storage;
use num_traits::Zero;
use kernel::{Kernel, LocalKernel};

/// HRBF specific kernel type. In general, we can assign a unique kernel to each hrbf site, or we
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
    where T: Real,
          K: Kernel<T>
{
    sites: Vec<Point3<T>>,
    betas: Vec<Vector4<T>>,
    kernel: KernelType<K>,
}

impl<T,K> Default for HRBF<T,K>
    where T: Real,
          K: Kernel<T>
{
    fn default() -> Self {
        HRBF {
            sites: Vec::new(),
            betas: Vec::new(),
            kernel: KernelType::Constant(K::default()),
        }
    }
}

impl<T,K> HRBF<T,K>
    where T: Real,
          K: Kernel<T> + LocalKernel<T>
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

impl<T,K> HRBF<T,K>
    where T: Real,
          K: Kernel<T>
{
    /// Main constructor. Assigns the degrees of freedom used by this HRBF in a form of 3D points
    /// at which the kernel will be evaluated.
    pub fn new(sites: Vec<Point3<T>>) -> Self {
        HRBF {
            sites,
            ..HRBF::default()
        }
    }

    /// Fit the current HRBF to the given data. Return true if successful.
    /// NOTE: Currently, points must be the same as sites.
    #[allow(non_snake_case)]
    pub fn fit(&mut self, points: &Vec<Point3<T>>, normals: &Vec<Vector3<T>>) -> bool {
        assert!(normals.len() == points.len());
        let rows = 4*points.len();
        let num_sites = self.sites.len();
        let cols = 4*num_sites;
        let mut A = DMatrix::<T>::zeros(rows,cols);
        let mut b = DVector::<T>::zeros(rows);

        for (i, p) in points.iter().enumerate() {
            b.fixed_rows_mut::<U3>(4*i+1).copy_from(&normals[i]);
            for j in 0..num_sites {
                A.fixed_slice_mut::<U4, U4>(4*i,4*j).copy_from(&self.fit_block(*p,j));
            }
        }

        self.betas.clear();
        if let Some(x) = A.lu().solve(&b) {
            assert!(x.len() == num_sites);

            self.betas.resize(num_sites, Vector4::zero());
            for j in 0..num_sites {
                self.betas[j].copy_from(&x.fixed_rows::<U4>(4*j));
            }

            true
        } else {
            false
        }
    }

    /// Returns a reference to the vector of site locations used by this hrbf.
    pub fn sites(&self) -> &Vec<Point3<T>> {
        &self.sites
    }

    /// The following are derivatives of the function
    ///   phi(x) := kernel(|x|)

    /// Given a vector `x` and its norm `l`, return the gradient of the kernel evaluated
    /// at `l` wrt `x`. `j` denotes the site at which the kernel is evaluated.
    fn grad_phi(&self, x: Vector3<T>, l: T, j: usize) -> Vector3<T> {
        x*self.kernel[j].df_l(l)
    }

    /// Given a vector `x` and its norm `l`, return the hessian of the kernel evaluated
    /// at `l` wrt `x`. `j` denotes the site at which the kernel is evaluated.
    fn hess_phi(&self, x: Vector3<T>, l: T, j: usize) -> Matrix3<T> {
        let df_l = self.kernel[j].df_l(l);
        let mut hess = Matrix3::identity();
        if l <= T::zero() {
            debug_assert!({
                let g = self.kernel[j].ddf(l) - df_l;
                g*g < T::from(1e-12).unwrap()
            });
            return hess*df_l;
        }

        let ddf = self.kernel[j].ddf(l);
        let x_hat = x / l;
        hess.ger_symm(ddf - df_l, &x_hat, &x_hat, df_l);
        hess // df_l*I + x_hat*x_hat.transpose()*(ddf - df_l)
    }

    /// Given a vector `x` and its norm `l`, return the third derivative of the kernel evaluated
    /// at `l` wrt `x` when multiplied by vector b.
    /// `j` denotes the site at which the kernel is evaluated.
    fn third_deriv_prod_phi<S>(&self, x: Vector3<T>, l: T, b: &Vector<T,U3,S>, j: usize) -> Matrix3<T>
        where S: Storage<T, U3, U1>
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
        mtx.ger_symm(_1, b, &x_hat, x_dot_b);
        mtx.ger(_1, &x_hat, b, _1);
        mtx.ger_symm((dddf - _3*g)*x_dot_b, &x_hat, &x_hat, g);
        mtx
    }

    /// Given a vector `x` and its norm `l`, return the fourth derivative of the kernel evaluated
    /// at `l` wrt `x` when multiplied by vectors b and c.
    /// `j` denotes the site at which the kernel is evaluated.
    #[inline]
    fn fourth_deriv_prod_phi<S>(&self,
                                x: Vector3<T>,
                                l: T,
                                b: &Vector<T,U3,S>,
                                c: &Vector<T,U3,S>,
                                j: usize) -> Matrix3<T>
        where S: Storage<T, U3, U1>
    {
        let g_l = self.kernel[j].g_l(l);
        let bc_tr = Matrix3::new(
            b[0]*c[0], b[1]*c[0], b[2]*c[0],
            b[0]*c[1], b[1]*c[1], b[2]*c[1],
            b[0]*c[2], b[1]*c[2], b[2]*c[2]);
        let c_dot_b = bc_tr.trace();
        let bc_tr_plus_cb_tr = bc_tr + bc_tr.transpose();
        let mut res = Matrix3::identity();
        if l <= T::zero() {
            debug_assert!({
                let h3 = self.kernel[j].h(l,T::from(3).unwrap());
                let h52 = self.kernel[j].h(l,T::from(5.0/2.0).unwrap());
                let ddddf = self.kernel[j].ddddf(l);
                let a = ddddf - T::from(6.0).unwrap()*h52;
                h3 == T::zero() && a*a < T::from(1e-12).unwrap()
            });
            return res*(g_l*c_dot_b) + (bc_tr_plus_cb_tr)*g_l;
        }

        let h3 = self.kernel[j].h(l,T::from(3).unwrap());
        let h52 = self.kernel[j].h(l,T::from(5.0/2.0).unwrap());
        let ddddf = self.kernel[j].ddddf(l);
        let a = ddddf - T::from(6.0).unwrap()*h52;
        let x_hat = x / l;
        let x_dot_b = x_hat.dot(b);
        let x_dot_c = x_hat.dot(c);
        let cb_sum = c*x_dot_b + b*x_dot_c;
        let _1 = T::one();

        // TODO: optimize this expression. we can probably achieve the same thing with less flops
        //xxT*(a*xTb*xTc + h3*cTb)
        //    + I*(h3*xTc*xTb + g_l*cTb)
        //    + ((cxT + xcT)*xTb + (bxT + xbT)*xTc)*h3
        //    + (bcT + cbT)*g_l
        res.ger_symm(a*x_dot_b*x_dot_c + h3*c_dot_b, &x_hat, &x_hat, h3*x_dot_c*x_dot_b + g_l*c_dot_b);
        res.ger_symm(h3, &cb_sum, &x_hat, _1);
        res.ger(h3, &x_hat, &cb_sum, _1);
        res + (bc_tr_plus_cb_tr)*g_l
    }

    /// Evaluate the HRBF at point `p`.
    pub fn eval(&self, p: Point3<T>) -> T {
        self.betas.iter()
            .enumerate()
            .fold(T::zero(), |sum, (j, b)| sum + self.eval_block(p, j).dot(b))
    }

    /// Helper function for `eval`.
    fn eval_block(&self, p: Point3<T>, j: usize) -> Vector4<T> {
        let x = p - self.sites[j];
        let l = norm(&x);
        let w = self.kernel[j].f(l);
        let g = self.grad_phi(x, l, j);
        Vector4::new(w, g[0], g[1], g[2])
    }

    /// Gradient of the hrbf function at point `p`.
    pub fn grad(&self, p: Point3<T>) -> Vector3<T> {
        self.betas.iter()
            .enumerate()
            .fold(Vector3::zero(), |sum, (j, b)| sum + self.grad_block(p, j)*b)
    }

    /// Helper function for `grad`. Returns a 3x4 matrix that gives the gradient of the hrbf when
    /// multiplied by the corresponding coefficients.
    fn grad_block(&self, p: Point3<T>, j: usize) -> Matrix3x4<T> {
        let x = p - self.sites[j];
        let l = norm(&x);
        let h = self.hess_phi(x, l, j);
        let mut grad = Matrix3x4::zero();
        grad.column_mut(0).copy_from(&self.grad_phi(x,l,j));
        grad.fixed_columns_mut::<U3>(1).copy_from(&h);
        grad
    }

    /// Compute the hessian of the HRBF function.
    pub fn hess(&self, p: Point3<T>) -> Matrix3<T> {
        self.betas.iter()
            .enumerate()
            .fold(Matrix3::zero(), |sum, (j, b)| sum + self.hess_block_prod(p, b, j))
    }

    /// Helper function for computing the hessian
    #[inline]
    fn hess_block_prod(&self, p: Point3<T>, b: &Vector4<T>, j: usize) -> Matrix3<T> {
        let x = p - self.sites[j];
        let l = norm(&x);
        let b3 = b.fixed_rows::<U3>(1);
        let h = self.hess_phi(x, l, j);
        h*b[0] + self.third_deriv_prod_phi(x, l, &b3, j)
    }

    /// Recall that the hrbf fit is done as
    ///
    ///  ∑ⱼ ⎡  𝜙(𝑥ᵢ - 𝑥ⱼ)  ∇𝜙(𝑥ᵢ - 𝑥ⱼ)'⎤ ⎡ 𝛼ⱼ⎤ = ⎡ 0 ⎤
    ///     ⎣ ∇𝜙(𝑥ᵢ - 𝑥ⱼ) ∇∇𝜙(𝑥ᵢ - 𝑥ⱼ) ⎦ ⎣ 𝛽ⱼ⎦   ⎣ 𝑛ᵢ⎦
    ///
    /// for every hrbf site i, where the sum runs over hrbf sites j
    /// where 𝜙(𝑥) = 𝜑(||𝑥||) for one of the basis kernels we define in kernel.rs
    /// If we rewrite the equation above as
    ///
    ///  ∑ⱼ Aⱼ(𝑥ᵢ)bⱼ = rᵢ
    ///
    /// this function returns the matrix Aⱼ(p).
    ///
    /// This is the symmetric 4x4 matrix block that is used to fit the hrbf coefficients.
    /// This is equivalent to stacking the vector from `eval_block` on top of the
    /// 3x4 matrix returned by `grad_block`. This function is more efficient than
    /// evaluating `eval_block` and `grad_block`.
    /// This is [g ∇g]' = [𝜙 (∇𝜙)'; ∇𝜙 ∇(∇𝜙)'] in matlab notation.
    pub fn fit_block(&self, p: Point3<T>, j: usize) -> Matrix4<T> {
        let x = p - self.sites[j];
        let l = norm(&x);
        let w = self.kernel[j].f(l);
        let g = self.grad_phi(x, l, j);
        let h = self.hess_phi(x, l, j);
        Matrix4::new(
            w, g[0], g[1], g[2],
            g[0], h[(0,0)], h[(0,1)], h[(0,2)],
            g[1], h[(1,0)], h[(1,1)], h[(1,2)],
            g[2], h[(2,0)], h[(2,1)], h[(2,2)])
    }

    /// Using the same notation as above,
    /// this function returns the matrix
    ///
    ///  ∇(Aⱼ(p)b)'
    ///
    pub fn grad_fit_block_prod(&self, p: Point3<T>, b: Vector4<T>, j: usize) -> Matrix3x4<T> {
        let x = p - self.sites[j];
        let l = norm(&x);

        let b3 = b.fixed_rows::<U3>(1);

        let g = self.grad_phi(x, l, j);
        let h = self.hess_phi(x, l, j);

        let third = h*b[0] + self.third_deriv_prod_phi(x, l, &b3, j);

        let mut grad = Matrix3x4::zero();
        grad.column_mut(0).copy_from(&(g*b[0] + h*b3));
        grad.fixed_columns_mut::<U3>(1).copy_from(&third);
        grad
    }

    /// Using the same notation as above,
    /// given a 4d vector lagrange multiplier `c`, this function returns the matrix
    ///
    ///  ∇(∇(Aⱼ(p)βⱼ)'c)'
    ///
    /// where βⱼ are taken from `self.betas`
    fn hess_fit_prod_block(&self, p: Point3<T>, c: Vector4<T>, j: usize) -> Matrix3<T> {
        let x = p - self.sites[j];
        let l = norm(&x);

        let c3 = c.fixed_rows::<U3>(1);
        let a = self.betas[j][0];
        let b = self.betas[j].fixed_rows::<U3>(1);

        // Compute in blocks
        self.hess_phi(x, l, j)*c[0] * a
            + self.third_deriv_prod_phi(x, l, &b, j)*c[0]
            + self.third_deriv_prod_phi(x, l, &c3, j)*a
            + self.fourth_deriv_prod_phi(x, l, &b, &c3, j)
    }

    /// Sum of hess_fit_prod_block evaluated at all sites.
    pub fn hess_fit_prod(&self, p: Point3<T>, c: Vector4<T>) -> Matrix3<T> {
        (0..self.sites.len())
            .fold(Matrix3::zero(), |sum, j| sum + self.hess_fit_prod_block(p, c, j))
    }
}
