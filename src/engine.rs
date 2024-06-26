use core::ops::Deref;
use crate::utils::Loader;
use super::LSTMModel;

pub struct Engine {
    model_name: String,
    pmodel    : PModel
}

pub enum PModel {
    P0(LSTMModel< 1, 48 >, [f32; 1], [String; 0]),
    P1(LSTMModel< 2, 48 >, [f32; 2], [String; 1]),
    P2(LSTMModel< 3, 48 >, [f32; 3], [String; 2]),
    P3(LSTMModel< 4, 48 >, [f32; 4], [String; 3]),
    P4(LSTMModel< 5, 48 >, [f32; 5], [String; 4])
}

impl Engine {
    pub fn from< D: Deref< Target = [u8] > >(data: D) -> Option< Self > {
        let mut loader  = Loader::from(data);
        let     version = loader.load::< u32 >()?;

        if version != 0 {
            return None;
        }

        let model_name = loader.load         ()?;
        let num_params = loader.load::< u32 >()?;
        let pmodel     = match num_params {
            0 => {
                let names = loader.load()?;
                let model = LSTMModel::< 1, 48 >::from(&mut loader)?;
                PModel::P0(model, [0.0; 1], names)
            },
            1 => {
                let names = loader.load()?;
                let model = LSTMModel::< 2, 48 >::from(&mut loader)?;
                PModel::P1(model, [0.0; 2], names)
            },
            2 => {
                let names = loader.load()?;
                let model = LSTMModel::< 3, 48 >::from(&mut loader)?;
                PModel::P2(model, [0.0; 3], names)
            },
            3 => {
                let names = loader.load()?;
                let model = LSTMModel::< 4, 48 >::from(&mut loader)?;
                PModel::P3(model, [0.0; 4], names)
            },
            4 => {
                let names = loader.load()?;
                let model = LSTMModel::< 5, 48 >::from(&mut loader)?;
                PModel::P4(model, [0.0; 5], names)
            },
            _ => return None
        };

        loader.end().map(|_| Self { model_name, pmodel })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn parameter_name(&self, index: usize) -> Option< &str > {
        match &self.pmodel {
            PModel::P0(.., names) => names.get(index),
            PModel::P1(.., names) => names.get(index),
            PModel::P2(.., names) => names.get(index),
            PModel::P3(.., names) => names.get(index),
            PModel::P4(.., names) => names.get(index)
        }.map(|s| s.as_str())
    }

    pub fn process(&mut self, sample: f32) -> f32 {
        match &mut self.pmodel {
            PModel::P0(model, input, ..) => {
                input[ 0 ] = sample;
                model.forward(input)
            },
            PModel::P1(model, input, ..) => {
                input[ 0 ] = sample;
                model.forward(input)
            },
            PModel::P2(model, input, ..) => {
                input[ 0 ] = sample;
                model.forward(input)
            },
            PModel::P3(model, input, ..) => {
                input[ 0 ] = sample;
                model.forward(input)
            },
            PModel::P4(model, input, ..) => {
                input[ 0 ] = sample;
                model.forward(input)
            },
        }
    }
}
