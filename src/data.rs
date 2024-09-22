use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn_dataset::transform::Mapper;
use burn_dataset::transform::MapperDataset;
use burn_dataset::transform::PartialDataset;
use burn_dataset::transform::ShuffledDataset;
use burn_dataset::Dataset;
use burn_dataset::HuggingfaceDatasetLoader;
use burn_dataset::InMemDataset;
use burn_dataset::SqliteDataset;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChessPositionRaw {
    #[serde(rename = "FEN")]
    pub fen: String,
    #[serde(rename = "Evaluation")]
    pub evaluation: String,
}

#[derive(Clone, Debug)]
pub struct ChessPositionItem{
    pub fen_to_pieces: Vec<f32>,
    pub evaluation: f32,
} 

pub struct RawToItem;

impl Mapper<ChessPositionRaw, ChessPositionItem> for RawToItem{
    fn map(&self, item: &ChessPositionRaw) -> ChessPositionItem {
        /*
        White
        king
        pawn
        knight
        bishop
        rook
        queen
        Black
        king
        pawn
        knight
        bishop
        rook
        queen
         */
        let map = HashMap::from([
            ('K', 0),
            ('P', 1),
            ('N', 2),
            ('B', 3),
            ('R', 4),
            ('Q', 5),
            ('k', 6),
            ('p', 7),
            ('n', 8),
            ('b', 9),
            ('r', 10),
            ('q', 11),
        ]);
        let mut position = [[0.0; 64]; 12];

        let fen_str: Vec<&str> = item.fen.split_whitespace().collect();
        let mut count: usize = 0;
        for piece in fen_str[0].chars(){
            if piece == '/'{
                continue;
            }
            if piece.is_numeric(){
                count += piece.to_digit(10).unwrap() as usize;
                continue;
            }
            if let Some(value) = map.get(&piece){
                position[*value][count] = 1.0;
                count += 1
            }
        }

        let position: Vec<f32> = position.into_iter()
                                .flat_map(|item| item).collect();

        ChessPositionItem {
            fen_to_pieces: position,
            evaluation: match item.evaluation.parse::<f32>() {
                Ok(val) => val,
                Err(_) => {
                    let sign = &item.evaluation[0..1];
                    let len = item.evaluation.len() - 1;
                    let num = &item.evaluation[1..len];
                    match sign{
                        "-" => {
                            -1.0 * num.parse::<f32>().unwrap()
                        },
                        "+" => {
                            num.parse::<f32>().unwrap()
                        },
                        _ => 0.0
                    }
                },
            }
        }
    }
}

type MappedDataset = MapperDataset<InMemDataset<ChessPositionRaw>, RawToItem, ChessPositionRaw>;
pub struct ChessPositionDataSet{
    dataset: MappedDataset,
}

impl Dataset<ChessPositionItem> for ChessPositionDataSet {
    fn get(&self, index: usize) -> Option<ChessPositionItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}



impl ChessPositionDataSet {
    pub fn train() -> Self{
        Self::new("train")
    }

    pub fn test() -> Self{
        Self::new("test")
    }

    fn new(split: &str) -> Self{
        type ChessEval = SqliteDataset<ChessPositionRaw>;
        let root: SqliteDataset<ChessPositionRaw> = HuggingfaceDatasetLoader::new("ssingh22/chess-evaluations")
            .with_subset("randoms")
            .dataset("train") // The training split.
            .unwrap();

        let dataset = ShuffledDataset::<ChessEval,ChessPositionRaw>::with_seed(root, 42);
        let len = dataset.len();
        type PartialData = PartialDataset<ShuffledDataset<ChessEval, ChessPositionRaw>, ChessPositionRaw>;

        let data_split = match split {
            "train" => PartialData::new(dataset, 0, len * 8 / 10), // Get first 80% dataset
            "test" => PartialData::new(dataset, len * 8 / 10, len), // Take remaining 20%
            _ => panic!("Invalid split type"),                     // Handle unexpected split types
        };

        let mut fens = Vec::new();
        let mut evals = Vec::new();
        for item in data_split.iter(){
            fens.push(item.fen);
            evals.push(item.evaluation);
        }

        let items: Vec<_> = fens
            .into_iter()
            .zip(evals)
            .map(|(fen, eval)| ChessPositionRaw { fen:fen, evaluation:eval})
            .collect();
        let dataset = InMemDataset::new(items);
        let dataset = MapperDataset::new(dataset, RawToItem);

        ChessPositionDataSet { dataset: dataset}
    }
}

#[derive(Clone)]
pub struct ChessPositionBatcher<B: Backend>{
    device: B::Device,
}

impl<B: Backend> ChessPositionBatcher<B> {
    pub fn new(device: B::Device) -> Self{
        Self{ device }
    }

    pub fn min_max_norm<const D: usize>(&self, inp: Tensor<B, D>) -> Tensor<B, D> {
        let min = inp.clone().min_dim(0);
        let max = inp.clone().max_dim(0);
        (inp.clone() - min.clone()).div(max - min)
    }
}

#[derive(Debug, Clone)]
pub struct ChessPositionBatch<B: Backend> {
    pub fens: Tensor<B, 2>,
    pub evaluations: Tensor<B, 1>,
}

impl <B: Backend> Batcher<ChessPositionItem, ChessPositionBatch<B>> for ChessPositionBatcher<B>{
    fn batch(&self, items: Vec<ChessPositionItem>) -> ChessPositionBatch<B> {
        let mut fens: Vec<Tensor<B, 2>> = Vec::new();
        for item in items.iter() {
            let fens_tensor = Tensor::<B, 1>::from_data(item.fen_to_pieces.as_slice(), &self.device);
            fens.push(fens_tensor.unsqueeze());
        }

        let fens = Tensor::cat(fens, 0);

        let evaluations = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_data([item.evaluation as f32], &self.device))
            .collect();
        let evaluations = Tensor::cat(evaluations, 0); 
        //let evaluations = self.min_max_norm(evaluations);

        ChessPositionBatch { fens: fens, evaluations: evaluations }
    }
}