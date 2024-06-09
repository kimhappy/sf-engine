// #![no_std]
#![allow(incomplete_features)]
#![allow(internal_features)]
#![allow(private_bounds)]
#![feature(core_intrinsics)]
#![feature(generic_const_exprs)]

mod activation;
mod loader;
mod model;
mod engine;
mod config;

use activation::{ sigmoid, tanh };
use loader::Loader;
use model::LSTMModel;
use config::ParamConfig;
pub use engine::Engine;
