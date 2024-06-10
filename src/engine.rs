use core::ops::Deref;
use super::{ Loader, LSTMModel };

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

        let model_name = loader.load()?;
        let pmodel     = match loader.load::< u32 >()? {
            0 => LSTMModel::< 1, 48 >::from(&mut loader)
                .and_then(|m| Some(PModel::P0(m, [0.0; 1], loader.load()?))),
            1 => LSTMModel::< 2, 48 >::from(&mut loader)
                .and_then(|m| Some(PModel::P1(m, [0.0; 2], loader.load()?))),
            2 => LSTMModel::< 3, 48 >::from(&mut loader)
                .and_then(|m| Some(PModel::P2(m, [0.0; 3], loader.load()?))),
            3 => LSTMModel::< 4, 48 >::from(&mut loader)
                .and_then(|m| Some(PModel::P3(m, [0.0; 4], loader.load()?))),
            4 => LSTMModel::< 5, 48 >::from(&mut loader)
                .and_then(|m| Some(PModel::P4(m, [0.0; 5], loader.load()?))),
            _ => None
        }?;

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
