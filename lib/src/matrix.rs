pub(crate) struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Box<[f32]>,
}

impl Matrix {
    pub(crate) fn new(rows: usize, cols: usize) -> Self {
        todo!()
    }

    pub(crate) fn transpose(&self) -> Matrix {
        todo!()
    }

    pub(crate) fn add_in_place(&mut self, other: &Matrix) {
        todo!()
    }

    pub(crate) fn add_row_vector(&mut self, row_vector: &Matrix) {
        todo!()
    }

    pub(crate) fn map(&mut self, func: impl Fn(f32) -> f32) {
        todo!()
    }

    pub(crate) fn dot(&self, other: &Matrix) -> Matrix {
        todo!()
    }
}
