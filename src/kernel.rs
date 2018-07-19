//! Hermite radial basis function kernels (funcions φ)
//! We require that the first derivative of the kernel f go to zero as x -> 0.
//! This ensures that the hrbf fitting matrix is well defined.
//! I.e. in its taylor series representation, f(x) = ∑ᵢ aᵢxⁱ, we require that a₁ = 0.
//!
//! NOTE: Kernels like x^2, x^3, gauss, and (1-x)^4 (4x+1) all satisfy this criterion.
//!
//! To ensure that derivativevs at zero are computed accurately, each kernel must provide a formula
//! for the following additional function:
//!   df_l(x) = φʹ(x)/x
//! where φ is the kernel function. It must be that df(x), ddf(x) - df_l(x) -> 0 as x -> 0 for the HRBF
//! derivaitves to exist.
//!
//!
//! Furthermore, if the HRBF is to be used in the optimization framework,
//! where a 3rd or even 4th derivatives are required, we also need the 3rd derivative to vanish.
//! I.e. in its taylor series representation, f(x) = ∑ᵢ aᵢxⁱ, we require that a₁ = a₃ = 0.
//!
//! NOTE: The gauss and x^2 kernels satisfy this criteria, but x^3, x^2*log(x) and (1-x)^4 (4x+1) do not!
//!
//! To ensure that derivatives at zero are computed accurately, each kernel must provide a formula for
//! the function df_l (as described above) and the following additional functions:
//!   g(x) = φʺ(x)/x - φʹ(x)/x^2
//!   g_l(x) = φʺ(x)/x^2 - φʹ(x)/x^3
//!   h(x,a) = φ‴(x)/x - a(φʺ(x)/x^2 - φʹ(x)/x^3)
//! to see how these are used, see mesh_implicit_surface.cpp
//! It must be that g(x), h(x,3) -> 0 as x -> 0 for the HRBF derivatives to exist.

use num_traits::Float;
use std::marker::PhantomData;

/// Kernel trait declaring all of the necessary derivatives.
pub trait Kernel<T>
where
    T: Float,
{
    /// Main kernel function φ(x)
    fn f(&self, x: T) -> T;
    /// The first derivative φʹ(x)
    fn df(&self, x: T) -> T;
    /// The second derivative φʺ(x)
    fn ddf(&self, x: T) -> T;
    /// The third derivative of φ(x)
    fn dddf(&self, x: T) -> T;
    /// The fourth derivative of φ(x)
    fn ddddf(&self, x: T) -> T;

    /// Additional function to ensure proper derivatives at x = 0.
    /// equivalent to df(x)/x for x != 0.
    /// This function should be well defined at all values of x.
    fn df_l(&self, x: T) -> T;

    /// Need the following functions for third and fourth hrbf derivaitves
    ///
    /// Additional function to ensure proper derivatives at x = 0.
    /// equivalent to ddf(x)/x - df(x)/(x*x) for x != 0.
    /// This function should go to zero as x goes to zero.
    fn g(&self, x: T) -> T;

    /// Additional function to ensure proper derivatives at x = 0.
    /// equivalent to ddf(x)/(x*x) - df(x)/(x*x*x) for x != 0.
    /// This function should be well defined at all values of x.
    fn g_l(&self, x: T) -> T;

    /// Additional function to ensure proper derivatives at x = 0.
    /// equivalent to dddf(x)/x - a*(ddf(x)/(x*x) - df(x)/(x*x*x)) for x != 0.
    /// This function should go to zero as x goes to zero.
    fn h(&self, x: T, a: T) -> T;
}

/// Local kernel trait defines the radial fall-off for appropriate kernels.
pub trait LocalKernel<T>
where
    T: Float,
{
    fn new(r: T) -> Self;
}

/// Global kernel trait defines a constructor for kernels without a radial fallofff.
pub trait GlobalKernel {
    fn new() -> Self;
}

#[derive(Copy, Clone)]
pub struct Pow2<T>(PhantomData<T>);

impl<T: Float> GlobalKernel for Pow2<T> {
    fn new() -> Self {
        Pow2(PhantomData)
    }
}

/// Default constructor for `Pow2` kernel
impl<T: Float> Default for Pow2<T> {
    fn default() -> Self {
        Pow2(PhantomData)
    }
}

impl<T: Float> Kernel<T> for Pow2<T> {
    fn f(&self, x: T) -> T {
        x * x
    }
    fn df(&self, x: T) -> T {
        T::from(2.0).unwrap() * x
    }
    fn ddf(&self, _: T) -> T {
        T::from(2.0).unwrap()
    }
    fn dddf(&self, _: T) -> T {
        T::zero()
    }
    fn ddddf(&self, _: T) -> T {
        T::zero()
    }
    fn df_l(&self, _: T) -> T {
        T::from(2.0).unwrap()
    }
    fn g(&self, _: T) -> T {
        T::zero()
    }
    fn g_l(&self, _: T) -> T {
        T::zero()
    }
    fn h(&self, _: T, _: T) -> T {
        T::zero()
    }
}

#[derive(Copy, Clone)]
pub struct Pow3<T>(::std::marker::PhantomData<T>);

impl<T: Float> GlobalKernel for Pow3<T> {
    fn new() -> Self {
        Pow3(PhantomData)
    }
}

/// Default constructor for `Pow3` kernel
impl<T: Float> Default for Pow3<T> {
    fn default() -> Self {
        Pow3(PhantomData)
    }
}

impl<T: Float> Kernel<T> for Pow3<T> {
    fn f(&self, x: T) -> T {
        x * x * x
    }
    fn df(&self, x: T) -> T {
        T::from(3.0).unwrap() * x * x
    }
    fn ddf(&self, x: T) -> T {
        T::from(6.0).unwrap() * x
    }
    fn dddf(&self, _: T) -> T {
        T::from(6.0).unwrap()
    }
    fn ddddf(&self, _: T) -> T {
        T::zero()
    }
    fn df_l(&self, x: T) -> T {
        T::from(3.0).unwrap() * x
    }

    // WARNING: This function does not go to zero as x -> 0
    fn g(&self, _: T) -> T {
        T::from(3.0).unwrap()
    }

    // WARNING: This function doesn't exist at x = 0.
    fn g_l(&self, x: T) -> T {
        T::from(3.0).unwrap() / x
    }

    // WARNING: This function doesn't exist at x = 0
    fn h(&self, x: T, a: T) -> T {
        T::from(3.0).unwrap() * (T::from(2.0).unwrap() - a) / x
    }
}

#[derive(Copy, Clone)]
pub struct Pow4<T>(::std::marker::PhantomData<T>);

impl<T: Float> GlobalKernel for Pow4<T> {
    fn new() -> Self {
        Pow4(PhantomData)
    }
}

/// Default constructor for `Pow4` kernel
impl<T: Float> Default for Pow4<T> {
    fn default() -> Self {
        Pow4(PhantomData)
    }
}

impl<T: Float> Kernel<T> for Pow4<T> {
    fn f(&self, x: T) -> T {
        x * x * x * x
    }
    fn df(&self, x: T) -> T {
        T::from(4.0).unwrap() * x * x * x
    }
    fn ddf(&self, x: T) -> T {
        T::from(12.0).unwrap() * x * x
    }
    fn dddf(&self, x: T) -> T {
        T::from(24.0).unwrap() * x
    }
    fn ddddf(&self, _: T) -> T {
        T::from(24.0).unwrap()
    }
    fn df_l(&self, x: T) -> T {
        T::from(4.0).unwrap() * x * x
    }
    fn g(&self, x: T) -> T {
        T::from(8.0).unwrap() * x
    }
    fn g_l(&self, _: T) -> T {
        T::from(8.0).unwrap()
    }
    fn h(&self, _: T, a: T) -> T {
        T::from(24.0).unwrap() - a * T::from(8.0).unwrap()
    }
}

/// x^5 kernel.
#[derive(Copy, Clone)]
pub struct Pow5<T>(::std::marker::PhantomData<T>);

impl<T: Float> GlobalKernel for Pow5<T> {
    fn new() -> Self {
        Pow5(PhantomData)
    }
}

/// Default constructor for `Pow5` kernel
impl<T: Float> Default for Pow5<T> {
    fn default() -> Self {
        Pow5(PhantomData)
    }
}

impl<T: Float> Kernel<T> for Pow5<T> {
    fn f(&self, x: T) -> T {
        x * x * x * x * x
    }
    fn df(&self, x: T) -> T {
        T::from(5.0).unwrap() * x * x * x * x
    }
    fn ddf(&self, x: T) -> T {
        T::from(20.0).unwrap() * x * x * x
    }
    fn dddf(&self, x: T) -> T {
        T::from(60.0).unwrap() * x * x
    }
    fn ddddf(&self, x: T) -> T {
        T::from(120.0).unwrap() * x
    }
    fn df_l(&self, x: T) -> T {
        T::from(5.0).unwrap() * x * x * x
    }
    fn g(&self, x: T) -> T {
        T::from(15.0).unwrap() * x * x
    }
    fn g_l(&self, x: T) -> T {
        T::from(15.0).unwrap() * x
    }
    fn h(&self, x: T, a: T) -> T {
        (T::from(60.0).unwrap() - a * T::from(15.0).unwrap()) * x
    }
}

/// Gaussian kernel.
#[derive(Copy, Clone)]
pub struct Gauss<T> {
    r: T,
}

impl<T: Float> LocalKernel<T> for Gauss<T> {
    fn new(r: T) -> Self {
        Gauss { r }
    }
}

/// Default constructor for `Gauss` kernel
impl<T: Float> Default for Gauss<T> {
    fn default() -> Self {
        Gauss { r: T::one() }
    }
}

impl<T: Float> Kernel<T> for Gauss<T> {
    fn f(&self, x: T) -> T {
        T::exp(-x * x / self.r)
    }
    fn df(&self, x: T) -> T {
        -(T::from(2.0).unwrap() / self.r) * x * self.f(x)
    }
    fn ddf(&self, x: T) -> T {
        let _2 = T::from(2.0).unwrap();
        _2 * self.f(x) * (_2 * x * x - self.r) / (self.r * self.r)
    }
    fn dddf(&self, x: T) -> T {
        let r3 = self.r * self.r * self.r;
        let _4 = T::from(4.0).unwrap();
        let _3 = T::from(3.0).unwrap();
        let _2 = T::from(2.0).unwrap();
        _4 * x * self.f(x) * (_3 * self.r - _2 * x * x) / r3
    }
    fn ddddf(&self, x: T) -> T {
        let r2 = self.r * self.r;
        let r4 = r2 * r2;
        let x2 = x * x;
        let x4 = x2 * x2;
        let _4 = T::from(4.0).unwrap();
        let _12 = T::from(12.0).unwrap();
        let _3 = T::from(3.0).unwrap();
        _4 * (_3 * r2 - _12 * self.r * x2 + _4 * x4) * self.f(x) / r4
    }
    fn df_l(&self, x: T) -> T {
        -(T::from(2.0).unwrap() / self.r) * self.f(x)
    }
    fn g(&self, x: T) -> T {
        T::from(4.0).unwrap() * x * self.f(x) / (self.r * self.r)
    }
    fn g_l(&self, x: T) -> T {
        T::from(4.0).unwrap() * self.f(x) / (self.r * self.r)
    }
    fn h(&self, x: T, a: T) -> T {
        let _4 = T::from(4.0).unwrap();
        let _3 = T::from(3.0).unwrap();
        let _2 = T::from(2.0).unwrap();
        // NOTE: at h(x,3) -> 0 as x -> 0 as needed.
        _4 * self.f(x) * ((_3 - a) * self.r - _2 * x * x) / (self.r * self.r * self.r)
    }
}

/// Quintic kernel. Generates a positive definite hrbf fitting matrix.
/// Third and fourth order hrbf derivatives don't exist at x = 0.
#[derive(Copy, Clone)]
pub struct Csrbf31<T> {
    r: T,
}

impl<T: Float> LocalKernel<T> for Csrbf31<T> {
    fn new(r: T) -> Self {
        Csrbf31 { r }
    }
}

/// Default constructor for `Csrbf31` kernel
impl<T: Float> Default for Csrbf31<T> {
    fn default() -> Self {
        Csrbf31 { r: T::one() }
    }
}

impl<T: Float> Kernel<T> for Csrbf31<T> {
    fn f(&self, x: T) -> T {
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let _4 = T::from(4.0).unwrap();
            let t = _1 - x;
            let t4 = t * t * t * t;
            t4 * (_4 * x + _1)
        }
    }
    fn df(&self, x: T) -> T {
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let t = x - _1;
            let t3 = t * t * t;
            T::from(20).unwrap() * t3 * x / self.r
        }
    }
    fn ddf(&self, x: T) -> T {
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let t = _1 - x;
            T::from(20).unwrap() * t * t * (T::from(4).unwrap() * x - _1) / (self.r * self.r)
        }
    }
    fn dddf(&self, x: T) -> T {
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let _2 = T::from(2).unwrap();
            T::from(120).unwrap() * (x - _1) * (_2 * x - _1) / (self.r * self.r * self.r)
        }
    }
    fn ddddf(&self, x: T) -> T {
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            T::from(120).unwrap() * (T::from(4).unwrap() * x - T::from(3).unwrap())
                / (self.r * self.r * self.r * self.r)
        }
    }
    fn df_l(&self, x: T) -> T {
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let t = x - _1;
            let t3 = t * t * t;
            T::from(20).unwrap() * t3 / (self.r * self.r)
        }
    }
    fn g(&self, x: T) -> T {
        // WARNING: does not go to zero as x -> 0.
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let t = _1 - x;
            T::from(60).unwrap() * t * t / (self.r * self.r * self.r)
        }
    }
    fn g_l(&self, x: T) -> T {
        // WARNING: g_l doesn't go to zero as x -> 0.
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let t = _1 - x;
            T::from(60).unwrap() * t * t / (x * self.r * self.r * self.r * self.r)
        }
    }
    fn h(&self, x: T, a: T) -> T {
        // WARNING: h(x,3) doesn't go to zero as x -> 0.
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let t = _1 - x;
            let _2 = T::from(2).unwrap();
            let _4 = T::from(4).unwrap();
            T::from(60).unwrap() * t * (_2 - _4 * x - a * t)
                / (x * self.r * self.r * self.r * self.r)
        }
    }
}

#[derive(Copy, Clone)]
pub struct Csrbf42<T> {
    r: T,
}

impl<T: Float> LocalKernel<T> for Csrbf42<T> {
    fn new(r: T) -> Self {
        Csrbf42 { r }
    }
}

/// Default constructor for `Csrbf42` kernel
impl<T: Float> Default for Csrbf42<T> {
    fn default() -> Self {
        Csrbf42 { r: T::one() }
    }
}

impl<T: Float> Kernel<T> for Csrbf42<T> {
    fn f(&self, x: T) -> T {
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let _3 = T::from(3.0).unwrap();
            let _18 = T::from(18.0).unwrap();
            let _35 = T::from(35.0).unwrap();
            let t = _1 - x;
            let t3 = t * t * t;
            t3 * t3 * (_35 * x * x + _18 * x + _3)
        }
    }
    fn df(&self, x: T) -> T {
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let t = x - _1;
            let t5 = t * t * t * t * t;
            T::from(56.0).unwrap() * t5 * (T::from(5.0).unwrap() * x + _1) * x / self.r
        }
    }
    fn ddf(&self, x: T) -> T {
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let _35 = T::from(35.0).unwrap();
            let _4 = T::from(4.0).unwrap();
            let t = _1 - x;
            let t2 = t * t;
            T::from(56.0).unwrap() * t2 * t2 * (_35 * x * x - _4 * x - _1) / (self.r * self.r)
        }
    }
    fn dddf(&self, x: T) -> T {
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let _7 = T::from(7.0).unwrap();
            let _3 = T::from(3.0).unwrap();
            let t = x - _1;
            T::from(1680).unwrap() * t * t * t * (_7 * x - _3) * x / (self.r * self.r * self.r)
        }
    }
    fn ddddf(&self, x: T) -> T {
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let _7 = T::from(7.0).unwrap();
            let _3 = T::from(3.0).unwrap();
            let _5 = T::from(5.0).unwrap();
            let t = x - _1;
            T::from(1680).unwrap() * t * t * (_5 * x - _3) * (_7 * x - _1)
                / (self.r * self.r * self.r * self.r)
        }
    }
    fn df_l(&self, x: T) -> T {
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let _5 = T::from(5.0).unwrap();
            let t = x - _1;
            T::from(56).unwrap() * t * t * t * t * t * (_5 * x + _1) / (self.r * self.r)
        }
    }
    fn g(&self, x: T) -> T {
        // WARNING: does not go to zero as x -> 0.
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let t = x - _1;
            T::from(1680).unwrap() * x * t * t * t * t / (self.r * self.r * self.r)
        }
    }
    fn g_l(&self, x: T) -> T {
        // WARNING: g_l doesn't go to zero as x -> 0.
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let tr = (x - _1) / self.r;
            T::from(1680).unwrap() * tr * tr * tr * tr
        }
    }
    fn h(&self, x: T, a: T) -> T {
        // WARNING: h(x,3) doesn't go to zero as x -> 0.
        let _1 = T::one();
        let x = x / self.r;
        if x > _1 {
            T::zero()
        } else {
            let t = _1 - x;
            let _3 = T::from(3).unwrap();
            let _7 = T::from(7).unwrap();
            T::from(1680).unwrap() * t * t * t * (_3 - _7 * x - a * t)
                / (self.r * self.r * self.r * self.r)
        }
    }
}
