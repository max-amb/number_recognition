#[derive(Debug)]
pub struct Pool {
    size: u64,
    stride: u64,
    pooling_function: fn(DMatrix<f32>) -> DMatrix<f32>
}
