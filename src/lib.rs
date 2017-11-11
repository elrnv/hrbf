extern crate num_traits;
extern crate nalgebra;

pub mod hrbf;
pub mod kernel;

pub trait Real: nalgebra::Real + num_traits::Float + ::std::fmt::Debug {}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
