#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Shape {
    nrows: usize,
    ncols: usize,
    channels: usize
}

impl Shape {
    pub fn magnitude(&self) -> usize {
        self.nrows*self.ncols*self.channels
    }

    pub fn flat_shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
}

impl From<(usize, usize)> for Shape {
    fn from(value: (usize, usize)) -> Self {
        Self { nrows: value.0, ncols: value.1, channels: 1 }
    } 
}
