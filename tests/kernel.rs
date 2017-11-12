extern crate num_traits;
extern crate hrbf;
extern crate rand;
#[macro_use]
extern crate approx;

#[allow(dead_code)]
mod autodiff;

use hrbf::kernel::*;
use autodiff::{Num, cst};

const TEST_VALS: [f64;4] = [0.0, 1.0, 0.5, ::std::f64::consts::PI];
const TEST_RADIUS: f64 = 2.0;

fn test_kernel<F,K: Kernel<Num>>(ker: K, x0: f64, compare: F)
    where F: Fn(f64, f64)
{
    let x = Num { val: x0, eps: 1.0 };

    let f = ker.f(x);
    let df = ker.df(x);
    let ddf = ker.ddf(x);
    let dddf = ker.dddf(x);
    let ddddf = ker.ddddf(x);

    compare(f.eps, df.val);
    compare(df.eps, ddf.val);
    compare(ddf.eps, dddf.val);
    compare(dddf.eps, ddddf.val);

    if x0 != 0.0 {
        let df_l = ker.df_l(x);
        let g = ker.g(x);
        let g_l = ker.g_l(x);
        let h3 = ker.h(x, cst(3.0));
        let h52 = ker.h(x, cst(5.0/2.0));
        compare(x0*df_l.val, df.val);
        compare(x0*x0*g.val, ddf.val*x0 - df.val);
        compare(x0*g_l.val, g.val);
        compare(x0*x0*x0*h3.val, x0*x0*dddf.val - 3.0*(x0*ddf.val - df.val));
        compare(x0*x0*x0*h52.val, x0*x0*dddf.val - 0.5*5.0*(x0*ddf.val - df.val));
    }
}

fn test_kernel_simple<K: Kernel<Num>>(kern: K) {
    for &x in TEST_VALS.iter() { test_kernel(kern, x, easy_compare); }
}

fn test_kernel_random<K: Kernel<Num>>(kern: K) {
    use self::rand::{SeedableRng, StdRng};
    use self::rand::distributions::{IndependentSample, Range};

    let seed: &[_] = &[1,2,3,4];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let range = Range::new(-1.0, 1.0);
    for _ in 0..999 {
        let x = range.ind_sample(&mut rng);
        test_kernel(kern, x, hard_compare);
    }
}

fn easy_compare(a: f64, b: f64) {
    assert_ulps_eq!(a,b, max_ulps=6);
}

fn hard_compare(a: f64, b: f64) {
    assert_relative_eq!(a,b, max_relative=1e-13, epsilon=1e-14);
}

#[test]
fn pow2_test() {
    let kern = Pow2::<Num>::default();
    test_kernel_simple(kern);
    test_kernel_random(kern);
}

#[test]
fn pow3_test() {
    let kern = Pow3::<Num>::default();
    test_kernel_simple(kern);
    test_kernel_random(kern);
}

#[test]
fn pow4_test() {
    let kern = Pow4::<Num>::default();
    test_kernel_simple(kern);
    test_kernel_random(kern);
}

#[test]
fn pow5_test() {
    let kern = Pow5::<Num>::default();
    test_kernel_simple(kern);
    test_kernel_random(kern);
}

#[test]
fn gauss_test() {
    let kern = Gauss::<Num>::new(cst(TEST_RADIUS));
    test_kernel_simple(kern);
    test_kernel_random(kern);
}

#[test]
fn csrbf31_test() {
    let kern = Csrbf31::<Num>::new(cst(TEST_RADIUS));
    test_kernel_simple(kern);
    test_kernel_random(kern);
}

#[test]
fn csrbf42_test() {
    let kern = Csrbf42::<Num>::new(cst(TEST_RADIUS));
    test_kernel_simple(kern);
    test_kernel_random(kern);
}

