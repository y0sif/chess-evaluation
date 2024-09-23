use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
};
use nn::Sigmoid;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    activation: Sigmoid
}

#[derive(Config, Debug)]
pub struct ModelConfig;

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model{
            linear1: LinearConfig::new(64 * 64 * 5 * 2, 256).init(device),
            linear2: LinearConfig::new(512, 32).init(device),
            linear3: LinearConfig::new(32, 32).init(device),
            linear4: LinearConfig::new(32, 1).init(device),
            activation: Sigmoid::new(),
        }
    }
}

impl <B:Backend> Model<B> {
    pub fn forward(&self, side_to_move: Tensor<B, 2>, other_side: Tensor<B, 2>) -> Tensor<B, 2> {
        let left = self.linear1.forward(side_to_move);
        let left = clipped_relu(left);
        let right = self.linear1.forward(other_side);
        let right = clipped_relu(right);
        let x = Tensor::cat([left, right].to_vec(), 1);
        let x = self.linear2.forward(x);
        let x = clipped_relu(x); 
        let x = self.linear3.forward(x);
        let x = clipped_relu(x);
        self.linear4.forward(x)
    }
}

fn clipped_relu<B: Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    tensor.clamp_max(1).clamp_min(0)
}