#![allow(internal_features)]
#![allow(incomplete_features)]
#![feature(core_intrinsics)]
#![feature(specialization)]

mod activation;
mod loader;
mod model;
mod engine;

use activation::{ sigmoid, tanh };
use loader::Loader;
use model::LSTMModel;
pub use engine::Engine;
