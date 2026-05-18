use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Box<[f32]>,
}

impl Matrix {
    pub(crate) fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols].into_boxed_slice(),
        }
    }

    pub(crate) fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                result.data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        result
    }

    pub(crate) fn add_in_place(&mut self, other: &Matrix) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for i in 0..self.data.len() {
            self.data[i] += other.data[i];
        }
    }

    pub(crate) fn add_row_vector(&mut self, row_vector: &Matrix) {
        assert_eq!(row_vector.cols, 1);
        assert_eq!(self.rows, row_vector.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                self.data[r * self.cols + c] += row_vector.data[r];
            }
        }
    }

    pub(crate) fn map(&mut self, func: impl Fn(f32) -> f32) {
        for val in self.data.iter_mut() {
            *val = func(*val);
        }
    }

    pub(crate) fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);
        let mut result = Matrix::new(self.rows, other.cols);
        for r in 0..self.rows {
            for c in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[r * self.cols + k] * other.data[k * other.cols + c];
                }
                result.data[r * other.cols + c] = sum;
            }
        }
        result
    }
}
