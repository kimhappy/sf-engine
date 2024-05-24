#![no_std]
#![allow(incomplete_features)]
#![allow(internal_features)]
#![allow(private_bounds)]
#![feature(core_intrinsics)]
#![feature(generic_const_exprs)]

mod activation;
mod loader;
mod lstm_model;

use activation::{ sigmoid, tanh };
use loader::Loader;
pub use lstm_model::LSTMModel;
