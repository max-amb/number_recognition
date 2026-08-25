use std::process::exit;

use crate::tensor::{Shape, Ten};
use nalgebra::{DMatrix, DVector, Dyn};

pub trait Convolvable {
    fn filter_shape(&self) -> Shape;
    fn zero_padding(&self) -> usize;
    fn out_depth(&self) -> usize;
    fn stride(&self) -> usize;
}

pub fn col2im(m: &DMatrix<f32>, conv: &dyn Convolvable, outshape: Shape) -> Ten {
    let mut res: Vec<DMatrix<f32>> = Vec::from_iter(
        (0..outshape.channels).map(|_| DMatrix::from_element(outshape.nrows, outshape.ncols, 0.0)),
    );

    let filtersize = conv.filter_shape().nrows * conv.filter_shape().ncols;
    let output_of_layer_shape = out_shape(outshape, conv);

    for (i, kern_tensor) in m.column_iter().enumerate() {
        // For column in m, each of these is r_k * c_k * d_{in}

        for (j, mat_in_kern) in (0..kern_tensor.len()).step_by(filtersize).enumerate() {
            // For matrix in that column, each of these is r_k * c_k

            let starting_row = (i % output_of_layer_shape.nrows) * conv.stride();
            let starting_col = (i / output_of_layer_shape.nrows) * conv.stride();
            let mut view = res[j].index_mut((
                starting_row..(starting_row + conv.filter_shape().nrows),
                starting_col..(starting_col + conv.filter_shape().ncols),
            ));

            let filter = kern_tensor.view((mat_in_kern, 0), (filtersize, 1));
            view += filter.reshape_generic(
                Dyn(conv.filter_shape().nrows),
                Dyn(conv.filter_shape().ncols),
            );
        }
    }
    Ten {
        data: DVector::from_iterator(outshape.magnitude(), res.iter().flatten().copied()),
        shape: outshape,
    }
}

pub fn im2col(m: &Ten, conv: &dyn Convolvable) -> DMatrix<f32> {
    let outshape = out_shape(m.shape, conv);
    let (nrows, ncols) = m.shape.flat_shape();
    let (krows, kcols) = conv.filter_shape().flat_shape();
    let jump_size = nrows * ncols;
    let zp = conv.zero_padding();
    let stride = conv.stride();

    // Transforms input tensor into a list of matrices with correct shape
    let reshaped_mats = Vec::from_iter((0..m.shape.magnitude()).step_by(jump_size).map(|x| {
        m.data
            .view((x, 0), (jump_size, 1))
            .reshape_generic(Dyn(nrows), Dyn(ncols))
            .resize(
                nrows + zp,
                ncols + zp,
                0.0,
            )
    }));

    let mut columns = Vec::with_capacity(krows*kcols*outshape.magnitude());
    for mat in reshaped_mats {
        for i in (0..=((ncols + zp) - kcols)).step_by(stride) {
            for j in (0..=((nrows + zp) - krows)).step_by(stride) {
                columns.extend(
                    mat.view((j, i), (krows, kcols))
                );
            }
        }
    }
    DMatrix::from_vec(
        krows * kcols * m.shape.channels,
        outshape.nrows * outshape.ncols,
        columns,
    )
}

/// Given an input shape, i.e. the shape of the previous layer's output,
/// we give the output shape based on the convolution being applied (represented by the convolvable
/// trait). It takes into account all characteristics in the Convolvable trait.
pub fn out_shape(in_shape: Shape, conv: &dyn Convolvable) -> Shape {
    assert!(
        conv.filter_shape().nrows <= in_shape.nrows + conv.zero_padding()
            && conv.filter_shape().ncols <= in_shape.ncols + conv.zero_padding()
    );
    let new_nrows =
        ((in_shape.nrows + conv.zero_padding()) - conv.filter_shape().nrows) / conv.stride() + 1;
    let new_ncols =
        ((in_shape.ncols + conv.zero_padding()) - conv.filter_shape().ncols) / conv.stride() + 1;
    (new_nrows, new_ncols, conv.out_depth()).into()
}
