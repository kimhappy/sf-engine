use core::ops::Deref;
use super::{ sigmoid, tanh, Loader };

pub trait Model {
    fn forward(&mut self, inputs: &[f32]) -> f32;
}

pub struct LSTMModel<
    const IN    : usize,
    const HIDDEN: usize > {
    lstm_weight_ii: [[f32; IN    ]; HIDDEN],
    lstm_weight_if: [[f32; IN    ]; HIDDEN],
    lstm_weight_ig: [[f32; IN    ]; HIDDEN],
    lstm_weight_io: [[f32; IN    ]; HIDDEN],
    lstm_weight_hi: [[f32; HIDDEN]; HIDDEN],
    lstm_weight_hf: [[f32; HIDDEN]; HIDDEN],
    lstm_weight_hg: [[f32; HIDDEN]; HIDDEN],
    lstm_weight_ho: [[f32; HIDDEN]; HIDDEN],
    lstm_bias_i   : [ f32         ; HIDDEN],
    lstm_bias_f   : [ f32         ; HIDDEN],
    lstm_bias_g   : [ f32         ; HIDDEN],
    lstm_bias_o   : [ f32         ; HIDDEN],
    dense_weight  : [ f32         ; HIDDEN],
    dense_bias    :   f32                  ,
    lstm_o        : [ f32         ; HIDDEN],
    lstm_c        : [ f32         ; HIDDEN],
    lstm_out      : [ f32         ; HIDDEN]
}

impl<
    const IN    : usize,
    const HIDDEN: usize > LSTMModel< IN, HIDDEN > {
    pub fn from< S: Deref< Target = [u8] > >(loader: &mut Loader< S >) -> Option< Self > {
        Some(Self {
            lstm_weight_ii: loader.load()?,
            lstm_weight_if: loader.load()?,
            lstm_weight_ig: loader.load()?,
            lstm_weight_io: loader.load()?,
            lstm_weight_hi: loader.load()?,
            lstm_weight_hf: loader.load()?,
            lstm_weight_hg: loader.load()?,
            lstm_weight_ho: loader.load()?,
            lstm_bias_i   : loader.load()?,
            lstm_bias_f   : loader.load()?,
            lstm_bias_g   : loader.load()?,
            lstm_bias_o   : loader.load()?,
            dense_weight  : loader.load()?,
            dense_bias    :  0.0          ,
            lstm_o        : [0.0; HIDDEN] ,
            lstm_c        : [0.0; HIDDEN] ,
            lstm_out      : [0.0; HIDDEN]
        })
    }
}

impl<
    const IN    : usize,
    const HIDDEN: usize > Model for LSTMModel< IN, HIDDEN > {
    fn forward(&mut self, inputs: &[f32]) -> f32 {
        let mut out = inputs[ 0 ] + self.dense_bias;

        for i in 0..HIDDEN {
            let mut acc_i = self.lstm_bias_i[ i ];
            let mut acc_f = self.lstm_bias_f[ i ];
            let mut acc_g = self.lstm_bias_g[ i ];
            let mut acc_o = self.lstm_bias_o[ i ];

            for j in 0..IN {
                let x  = inputs                  [ j ];
                acc_i += self.lstm_weight_ii[ i ][ j ] * x;
                acc_f += self.lstm_weight_if[ i ][ j ] * x;
                acc_g += self.lstm_weight_ig[ i ][ j ] * x;
                acc_o += self.lstm_weight_io[ i ][ j ] * x;
            }

            for j in 0..HIDDEN {
                let x  = self.lstm_out           [ j ];
                acc_i += self.lstm_weight_hi[ i ][ j ] * x;
                acc_f += self.lstm_weight_hf[ i ][ j ] * x;
                acc_g += self.lstm_weight_hg[ i ][ j ] * x;
                acc_o += self.lstm_weight_ho[ i ][ j ] * x;
            }

            self.lstm_o[ i ]  = sigmoid(acc_o);
            self.lstm_c[ i ] *= sigmoid(acc_i) * tanh(acc_g) + sigmoid(acc_f);
        }

        for i in 0..HIDDEN {
            self.lstm_out[ i ]  = self.lstm_o      [ i ] * tanh(self.lstm_c  [ i ]);
            out                += self.dense_weight[ i ] *      self.lstm_out[ i ] ;
        }

        out
    }
}
