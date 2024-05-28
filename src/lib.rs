#![no_std]
#![allow(incomplete_features)]
#![allow(internal_features)]
#![allow(private_bounds)]
#![feature(core_intrinsics)]
#![feature(generic_const_exprs)]

mod activation;
mod loader;
mod model;

use activation::{ sigmoid, tanh };
pub use loader::Loader;
pub use model::Model;
