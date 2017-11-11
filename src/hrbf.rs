use nalgebra::{Point3, Vector4, Vector3, dot, norm};
use kernel::Kernel;
use Real;

/// HRBF specific kernel type. In general, we can assign a unique kernel to each hrbf site, or we
/// can use the same kernel for all points. This corresponds to Local and Global kernel types
/// respectively.
#[derive(Clone, Debug)]
pub enum KernelType<K> {
    Local(Vec<K>), // each site has its own kernel
    Global(K), // same kernel for all sites
}

impl<K> ::std::ops::Index<usize> for KernelType<K> {
    type Output = K;

    fn index(&self, index: usize) -> &K {
        match *self {
            KernelType::Local(ref ks) => &ks[index],
            KernelType::Global(ref k) => k,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HRBF<T, K>
    where T: Real,
          K: Kernel<T>
{
    sites: Vec<Point3<T>>,
    normals: Vec<Vector3<T>>,
    betas: Vec<Vector4<T>>,
    kernel: KernelType<K>,
}

impl<T,K> Default for HRBF<T,K>
    where T: Real,
          K: Kernel<T>
{
    fn default() -> Self {
        HRBF::new()
    }
}

impl<T,K> HRBF<T,K>
    where T: Real,
          K: Kernel<T>
{
    pub fn new() -> Self {
        HRBF {
            sites: Vec::new(),
            normals: Vec::new(),
            betas: Vec::new(),
            kernel: KernelType::Global(K::default()),
        }
    }

    /// Given a vector `x` and its norm `l`, return the gradient of the kernel evaluated
    /// at `l` wrt `x`. `j` denotes the site at which the kernel is evaluated.
    fn grad_phi(&self, x: Vector3<T>, l: T, j: usize) -> Vector3<T> {
        x*self.kernel[j].df_l(l)
    }

    /// Evaluate the HRBF at point `p`.
    pub fn eval(&self, p: Point3<T>) -> T {
        self.betas.iter()
            .enumerate()
            .fold(T::zero(), |sum, (j, b)| sum + dot(&self.eval_block(p, j),b))
    }

    /// Helper function for `eval`.
    fn eval_block(&self, p: Point3<T>, j: usize) -> Vector4<T> {
        let x = p - self.sites[j];
        let l = norm(&x);
        let w = self.kernel[j].f(l);
        let g = self.grad_phi(x, l, j);
        Vector4::new(w, g[0], g[1], g[2])
    }

    ///// Gradient of the hrbf function at point `p`.
    //pub fn grad(&self p: Point3) -> T {
    //    sites.iter()
    //        .zip(betas.iter())
    //        .fold(T::zero(), |sum, (c, b)| sum + grad_block(p - c)*b)
    //}

    ///// Helper function for `grad`. Returns a 3x4 matrix that gives the gradient of the hrbf when
    ///// multiplied by the corresponding coefficients.
    //fn grad_block(&self, x: Vector3) -> Matrix3x4<T> {

    //}

    ///// Hessian block product. To avoid dealing with high order tensors, we expose the product of
    ///// the hessian block with an arbitrary Vector3.
    //pub fn hess_block_prod(&self, p: Point3<T>, lambda: Vector3<T>) -> T {
    //    sites.iter()
    //        .zip(betas.iter())
    //        .fold(T::zero(), |sum, (c, b)| -> sum + dot(grad_block(p - c),b))
    //}

    //pub fn hess_block(&self, x: Vector3<T>) -> T {

    //}
}


