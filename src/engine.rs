use core::ops::Deref;
use super::{ Loader, Model, LSTMModel, ParamConfig };

pub struct Engine {
    param_configs: Vec< ParamConfig      >,
    models       : Vec< Box< dyn Model > >,
    input        : Vec< f32 >             ,
    simple_params: Vec< f32 >             ,
    weights      : Vec< f32 >             ,
}

impl Engine {
    pub fn from< D: Deref< Target = [u8] > >(data: D) -> Option< Self > {
        let mut loader = Loader::from(data);

        let num_pcs          = loader.load::< u32 >()?;
        let (mut sc, mut cc) = (0, 0);
        let param_configs    = (0..num_pcs).map(|_| match loader.load::< u8 >()? {
            0 => {
                let pc = ParamConfig::Simple(sc);
                sc += 1;
                Some(pc)
            }
            1 => {
                let pc = ParamConfig::Complex(cc);
                cc += 1;
                Some(pc)
            }
            _ => None
        }).collect::< Option< Vec< _ > > >()?;

        let num_models = 1 << sc;
        let models     = (0..num_models).map(|_| -> Option< Box< dyn Model > > {
            match cc {
                0 => Some(Box::new(LSTMModel::< 1, 48 >::from(&mut loader)?)),
                1 => Some(Box::new(LSTMModel::< 2, 48 >::from(&mut loader)?)),
                2 => Some(Box::new(LSTMModel::< 3, 48 >::from(&mut loader)?)),
                _ => None,
            }
        }).collect::< Option< Vec< _ > > >()?;

        if loader.end() {
            Some(Self {
                param_configs                    ,
                models                           ,
                input        : vec![0f32; cc + 1],
                simple_params: vec![0f32; sc    ],
                weights      : {
                    let mut zeros = vec![0f32; 1 << sc];
                    zeros[ 0 ] = 1.0;
                    zeros
                } })
        }
        else {
            None
        }
    }

    pub fn set_param(&mut self, index: usize, value: f32) -> Option< () > {
        Some(match self.param_configs.get(index)? {
            ParamConfig::Simple(idx) => {
                self.simple_params[ *idx ] = value;

                for i in 0..self.weights.len() {
                    self.weights[ i ] = self.simple_params.iter().enumerate().fold(1.0, |acc, (j, &p)| acc * if i & (1 << j) != 0 { p } else { 1.0 - p });
                }
            },
            ParamConfig::Complex(idx) => {
                self.input[ idx + 1 ] = value
            }
        })
    }

    pub fn process(&mut self, sample: f32) -> f32 {
        self.input[ 0 ] = sample;

        self.models.iter_mut()
            .zip(self.weights.iter())
            .fold(0.0, |acc, (m, w)|
                if   *w == 0.0 { acc                              }
                else           { acc + w * m.forward(&self.input) })
    }
}
