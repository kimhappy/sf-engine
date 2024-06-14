#![allow(internal_features)]
#![feature(core_intrinsics)]

mod activation;
mod model;
mod engine;
mod utils;

use activation::{ sigmoid, tanh };
use model::LSTMModel;
pub use engine::Engine;
