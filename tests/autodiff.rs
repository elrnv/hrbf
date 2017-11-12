extern crate num_traits;

use num_traits::{ToPrimitive, Float, FloatConst, NumCast, Zero, One};
use std::f64;
use std::ops::{Neg, Add, Sub, Mul, Div, Rem};
use std::num::{FpCategory};

#[derive(Copy,Clone,Debug)]
pub struct Num {
    pub val: f64,
    pub eps: f64,
}

impl Neg for Num {
    type Output = Num;
    fn neg(self) -> Num {
        Num { val: -self.val,
              eps: -self.eps }
    }
}

impl Add<Num> for Num {
    type Output = Num;
    fn add(self, _rhs: Num) -> Num {
        Num { val: self.val + _rhs.val,
              eps: self.eps + _rhs.eps }
    }
}

impl Sub<Num> for Num {
    type Output = Num;
    fn sub(self, _rhs: Num) -> Num {
        Num { val: self.val - _rhs.val,
              eps: self.eps - _rhs.eps }
    }
}

impl Mul<Num> for Num {
    type Output = Num;
    fn mul(self, _rhs: Num) -> Num {
        Num { val: self.val * _rhs.val,
              eps: self.eps * _rhs.val + self.val * _rhs.eps }
    }
}

impl Mul<Num> for f64 {
    type Output = Num;
    fn mul(self, _rhs: Num) -> Num {
        Num { val: self * _rhs.val,
              eps: self * _rhs.eps }
    }
}

impl Mul<f64> for Num {
    type Output = Num;
    fn mul(self, _rhs: f64) -> Num {
        Num { val: self.val * _rhs,
              eps: self.eps * _rhs }
    }
}

impl Div<Num> for Num {
    type Output = Num;
    fn div(self, _rhs: Num) -> Num {
        Num { val: self.val / _rhs.val,
              eps: (self.eps * _rhs.val - self.val * _rhs.eps)
                 / (_rhs.val * _rhs.val) }
    }
}

impl Rem<Num> for Num {
    type Output = Num;
    fn rem(self, _rhs: Num) -> Num {
        panic!("Remainder not implemented")
    }
}

impl Default for Num {
    fn default() -> Self {
        Num {
            val: f64::default(),
            eps: 0.0
        }
    }
}

impl PartialEq<Num> for Num {
    fn eq(&self, _rhs: &Num) -> bool {
        self.val == _rhs.val
    }
}

impl PartialOrd<Num> for Num {
    fn partial_cmp(&self, other: &Num) -> Option<::std::cmp::Ordering> {
        PartialOrd::partial_cmp(&self.val, &other.val)
    }
}

impl ToPrimitive for Num {
    fn to_i64(&self)  -> Option<i64>  { self.val.to_i64()  }
    fn to_u64(&self)  -> Option<u64>  { self.val.to_u64()  }
    fn to_isize(&self)  -> Option<isize>  { self.val.to_isize()  }
    fn to_i8(&self)   -> Option<i8>   { self.val.to_i8()   }
    fn to_i16(&self)  -> Option<i16>  { self.val.to_i16()  }
    fn to_i32(&self)  -> Option<i32>  { self.val.to_i32()  }
    fn to_usize(&self) -> Option<usize> { self.val.to_usize() }
    fn to_u8(&self)   -> Option<u8>   { self.val.to_u8()   }
    fn to_u16(&self)  -> Option<u16>  { self.val.to_u16()  }
    fn to_u32(&self)  -> Option<u32>  { self.val.to_u32()  }
    fn to_f32(&self)  -> Option<f32>  { self.val.to_f32()  }
    fn to_f64(&self)  -> Option<f64>  { self.val.to_f64()  }
}

impl NumCast for Num {
    fn from<T: ToPrimitive> (n: T) -> Option<Num> {
        let _val = n.to_f64();
        match _val {
            Some(x) => Some(Num { val: x, eps: 0.0 }),
            None => None
        }
    }
}

impl Zero for Num {
    fn zero() -> Num { Num { val: 0.0, eps: 0.0 } }
    fn is_zero(&self) -> bool { self.val.is_zero() }
}

impl One for Num {
    fn one() -> Num { Num { val: 1.0, eps: 0.0 } }
}

impl ::num_traits::Num for Num {
    type FromStrRadixErr = ::num_traits::ParseFloatError;

    fn from_str_radix(src: &str, radix: u32)
        -> Result<Self, Self::FromStrRadixErr>
    {
        f64::from_str_radix(src, radix).map(|x| Num { val: x, eps: 0.0 })
    }
}

// Removed
// fn mantissa_digits(_self: Option<Self>) -> uint { f64::MANTISSA_DIGITS }
// fn digits(_self: Option<Self>) -> uint { f64::DIGITS }
// fn min_exp(_self: Option<Self>) -> int { f64::MIN_EXP }
// fn max_exp(_self: Option<Self>) -> int { f64::MAX_EXP }
// fn min_10_exp(_self: Option<Self>) -> int { f64::MIN_10_EXP }
// fn max_10_exp(_self: Option<Self>) -> int { f64::MAX_10_EXP }
// fn rsqrt(self) -> Num { Num { val: self.val.rsqrt(), eps: self.eps * -0.5 / self.val.sqrt().powi(3) } }

impl FloatConst for Num {
    fn E() -> Num { Num { val: f64::consts::E, eps: 0.0 } }
    fn FRAC_1_PI() -> Num { Num { val: f64::consts::FRAC_1_PI, eps: 0.0 } }
    fn FRAC_1_SQRT_2() -> Num { Num { val: f64::consts::FRAC_1_SQRT_2, eps: 0.0} }
    fn FRAC_2_PI() -> Num { Num { val: f64::consts::FRAC_2_PI, eps: 0.0 } }
    fn FRAC_2_SQRT_PI() -> Num { Num { val: f64::consts::FRAC_2_SQRT_PI, eps: 0.0 } }
    fn FRAC_PI_2() -> Num { Num { val: f64::consts::FRAC_PI_2, eps: 0.0 } }
    fn FRAC_PI_3() -> Num { Num { val: f64::consts::FRAC_PI_3, eps: 0.0 } } 
    fn FRAC_PI_4() -> Num { Num { val: f64::consts::FRAC_PI_4, eps: 0.0 } }
    fn FRAC_PI_6() -> Num { Num { val: f64::consts::FRAC_PI_6, eps: 0.0 } }
    fn FRAC_PI_8() -> Num { Num { val: f64::consts::FRAC_PI_8, eps: 0.0 } }
    fn LN_10() -> Num { Num { val: f64::consts::LN_10, eps: 0.0 } }
    fn LN_2() -> Num { Num { val: f64::consts::LN_2, eps: 0.0 } }
    fn LOG10_E() -> Num { Num { val: f64::consts::LOG10_E, eps: 0.0 } }
    fn LOG2_E() -> Num { Num { val: f64::consts::LOG2_E, eps: 0.0 } }
    fn PI() -> Num { Num { val: f64::consts::PI, eps: 0.0 } }
    fn SQRT_2() -> Num { Num { val: f64::consts::SQRT_2, eps: 0.0} }
}

impl Float for Num {
    fn nan() -> Num { Num { val: f64::NAN, eps: 0.0 } } 
    fn infinity() -> Num { Num { val: f64::INFINITY, eps: 0.0 } }
    fn neg_infinity() -> Num { Num { val: f64::NEG_INFINITY, eps: 0.0 } }
    fn neg_zero() -> Num { Num { val: -0.0, eps: 0.0 } }
    fn min_value() -> Num { Num { val: f64::MIN, eps: 0.0 } }
    fn min_positive_value() -> Num { Num { val: f64::MIN_POSITIVE, eps: 0.0 } }
    fn max_value() -> Num { Num { val: f64::MAX, eps: 0.0 } }
    fn is_nan(self) -> bool { self.val.is_nan() || self.eps.is_nan() }
    fn is_infinite(self) -> bool { self.val.is_infinite() || self.eps.is_infinite() }
    fn is_finite(self) -> bool { self.val.is_finite() && self.eps.is_finite() }
    fn is_normal(self) -> bool { self.val.is_normal() && self.eps.is_normal() }
    fn classify(self) -> FpCategory { self.val.classify() }

    fn floor(self) -> Num { Num { val: self.val.floor(), eps: self.eps } }
    fn ceil(self) -> Num { Num { val: self.val.ceil(), eps: self.eps } }
    fn round(self) -> Num { Num { val: self.val.round(), eps: self.eps } }
    fn trunc(self) -> Num { Num { val: self.val.trunc(), eps: self.eps } }
    fn fract(self) -> Num { Num { val: self.val.fract(), eps: self.eps } }
    fn abs(self) -> Num {
        Num {
            val: self.val.abs(),
            eps: if self.val >= 0.0 { self.eps } else { -self.eps }
        }
    }
    fn signum(self) -> Num { Num { val: self.val.signum(), eps: 0.0 } }
    fn is_sign_positive(self) -> bool { self.val.is_sign_positive() }
    fn is_sign_negative(self) -> bool { self.val.is_sign_negative() }
    fn mul_add(self, a: Num, b: Num) -> Num { self * a + b }
    fn recip(self) -> Num { Num { val: self.val.recip(), eps: -self.eps/(self.val * self.val) } }
    fn powi(self, n: i32) -> Num {
        Num {
            val: self.val.powi(n),
            eps: self.eps * n as f64 * self.val.powi(n - 1)
        }
    }
    fn powf(self, n: Num) -> Num {
        Num {
            val: Float::powf(self.val, n.val),
            eps: (Float::ln(self.val) * n.eps + n.val * self.eps / self.val) * Float::powf(self.val, n.val)
        }
    }
    fn sqrt(self) -> Num { Num { val: self.val.sqrt(), eps: self.eps * 0.5 / self.val.sqrt() } }

    fn exp(self) -> Num { Num { val: Float::exp(self.val), eps: self.eps * Float::exp(self.val) } }
    fn exp2(self) -> Num { Num { val: Float::exp2(self.val), eps: self.eps * Float::ln(2.0) * Float::exp(self.val) } }
    fn ln(self) -> Num { Num { val: Float::ln(self.val), eps: self.eps * self.val.recip() } }
    fn log(self, b: Num) -> Num {
        Num {
            val: Float::log(self.val, b.val),
            eps: -Float::ln(self.val) * b.eps / (b.val * Float::powi(Float::ln(b.val), 2)) + self.eps / (self.val * Float::ln(b.val)),
    } }
    fn log2(self) -> Num { Float::log(self, Num { val: 2.0, eps: 0.0 }) }
    fn log10(self) -> Num { Float::log(self, Num { val: 10.0, eps: 0.0 }) }
    fn max(self, other: Num) -> Num { Num { val: Float::max(self.val, other.val), eps: 0.0 } }
    fn min(self, other: Num) -> Num { Num { val: Float::min(self.val, other.val), eps: 0.0 } }
    fn abs_sub(self, other: Num) -> Num {
        if self > other {
            Num { val: Float::abs_sub(self.val, other.val), eps: (self - other).eps }
        } else {
            Num { val: 0.0, eps: 0.0 }
        }
    }
    fn cbrt(self) -> Num { Num { val: Float::cbrt(self.val), eps: 1.0/3.0 * self.val.powf(-2.0/3.0) * self.eps } }
    fn hypot(self, other: Num) -> Num {
        Float::sqrt(Float::powi(self, 2) + Float::powi(other, 2))
    }
    fn sin(self) -> Num { Num { val: Float::sin(self.val), eps: self.eps * Float::cos(self.val) } }
    fn cos(self) -> Num { Num { val: Float::cos(self.val), eps: -self.eps * Float::sin(self.val) } }
    fn tan(self) -> Num {
        let t = Float::tan(self.val);
        Num { val: t, eps: self.eps * (t * t + 1.0) }
    }
    fn asin(self) -> Num { Num { val: Float::asin(self.val), eps: self.eps / Float::sqrt(1.0 - Float::powi(self.val, 2)) } }
    fn acos(self) -> Num { Num { val: Float::acos(self.val), eps: -self.eps / Float::sqrt(1.0 - Float::powi(self.val, 2)) } }
    fn atan(self) -> Num { Num { val: Float::atan(self.val), eps: self.eps / Float::sqrt(Float::powi(self.val, 2) + 1.0) } }
    fn atan2(self, other: Num) -> Num {
        Num {
            val: Float::atan2(self.val, other.val),
            eps: (other.val * self.eps - self.val * other.eps) / (Float::powi(self.val, 2) + Float::powi(other.val, 2))
        }
    }
    fn sin_cos(self) -> (Num, Num) {
        let (s, c) = Float::sin_cos(self.val);
        let sn = Num { val: s, eps: self.eps * c };
        let cn = Num { val: c, eps: -self.eps * s };
        (sn, cn)
    }
    fn exp_m1(self) -> Num {
        Num { val: Float::exp_m1(self.val), eps: self.eps * Float::exp(self.val) }
    }
    fn ln_1p(self) -> Num {
        Num { val: Float::ln_1p(self.val), eps: self.eps / (self.val + 1.0) }
    }
    fn sinh(self) -> Num { Num { val: Float::sinh(self.val), eps: self.eps * Float::cosh(self.val) } }
    fn cosh(self) -> Num { Num { val: Float::cosh(self.val), eps: self.eps * Float::sinh(self.val) } }
    fn tanh(self) -> Num { Num { val: Float::tanh(self.val), eps: self.eps * (1.0 - Float::powi(Float::tanh(self.val), 2)) } }
    fn asinh(self) -> Num { Num { val: Float::asinh(self.val), eps: self.eps * (Float::powi(self.val, 2) + 1.0) } }
    fn acosh(self) -> Num { Num { val: Float::acosh(self.val), eps: self.eps * (Float::powi(self.val, 2) - 1.0) } }
    fn atanh(self) -> Num { Num { val: Float::atanh(self.val), eps: self.eps * (-Float::powi(self.val, 2) + 1.0) } }
    fn integer_decode(self) -> (u64, i16, i8) { self.val.integer_decode() }

    fn epsilon() -> Num { Num { val: f64::EPSILON, eps: 0.0 } }
    fn to_degrees(self) -> Num { Num { val: Float::to_degrees(self.val), eps: 0.0 } }
    fn to_radians(self) -> Num { Num { val: Float::to_radians(self.val), eps: 0.0 } }
}

//impl FloatMath for Num {
//    fn ldexp(x: Num, exp: int) -> Num { Num { val: FloatMath::ldexp(x.val, exp), eps: FloatMath::ldexp(x.eps, exp) } }
//    fn frexp(self) -> (Num, int) {
//        let (x, exp) = FloatMath::frexp(self.val);
//        (Num { val: x, eps: 0.0 }, exp)
//    }
//    fn next_after(self, other: Num) -> Num { Num { val: FloatMath::next_after(self.val, other.val), eps: 0.0 } }
//}

/// Function for creating a constant from a float
pub fn cst(x: f64) -> Num {
    Num { val: x, eps: 0.0 }
}

/// Evaluates the derivative of `func` at `x0`
pub fn diff<F>(func: F, x0: f64) -> f64 
    where F: FnOnce(Num) -> Num
{
    let x = Num { val: x0, eps: 1.0 };
    func(x).eps
}

/// Evaluates the gradient of `func` at `x0`
pub fn grad<F>(func: F, x0: Vec<f64>) -> Vec<f64>
    where F: Fn(Vec<Num>) -> Num
{
    let mut params = Vec::new();
    for x in x0.iter() {
        params.push(Num { val: *x, eps: 0.0 });
    }

    let mut results = Vec::new();

    for i in 0..params.len() {
        params[i].eps = 1.0;
        results.push(func(params.clone()).eps);
        params[i].eps = 0.0;
    }
    results
}
