use criterion::*;
use tract_data::prelude::*;
use tract_linalg::mmm::{FusedKerSpec, EagerPackedInput};
use tract_linalg::generic::mmm::generic_f32_4x4;
use tract_linalg::pack::Packing;

fn bench_sparse_mmm(c: &mut Criterion) {
    let m = 64;
    let n = 64;
    let k = 256;

    // 1. Setup Packing Logic
    let a_packer = f32::packing(4); // MR=4
    let b_packer = f32::packing(4); // NR=4

    // 2. Dense Data Generation
    let a_dense: Vec<f32> = (0..m * k).map(|i| (i % 10) as f32).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i % 10) as f32).collect();

    // 3. Sparse Data Generation (50% sparsity in blocks of 4 along K)
    // A is (M, K). Packing is Panel-Major (K-major inside panel). 
    // We want contiguous zeros in the Packed buffer.
    // Since we pack by iterating K inside the panel, we zero out blocks of 4 in K.
    let mut a_sparse = a_dense.clone();
    for r in 0..m {
        for c in 0..k {
            // If K is in a "zero block", set to 0.0
            // Block size 4. Skip 4, Zero 4, Skip 4, Zero 4...
            if (c / 4) % 2 == 1 {
                a_sparse[r * k + c] = 0.0;
            }
        }
    }

    // 4. Helper to Pack Data
    let pack = |packer: &tract_linalg::pack::PackedFormat, data: &[f32], rows: usize, cols: usize, k_axis: usize| -> Vec<f32> {
        let tensor = Tensor::from_shape(&[rows, cols], data).unwrap();
        // Pack (A: M*K, B: K*N). For A: K is axis 1. For B: K is axis 0.
        let packed_store = packer.pack_tensor_view(&tensor.view(), k_axis, 1 - k_axis).unwrap();
        
        // Extract raw bytes/floats from the Opaque Packed Store for the benchmark
        // This relies on knowing the underlying storage is contiguous
        // Ideally we pass the pointers from the input_store, but for benchmark we can just access the blob.
        // EagerPackedInput usually holds a Blob.
        let internal = packed_store.downcast_ref::<EagerPackedInput>().unwrap();
        let blob = &internal.packed;
        let slice = unsafe { std::slice::from_raw_parts(blob.as_ptr() as *const f32, blob.len() / 4) };
        slice.to_vec()
    };

    let a_dense_packed = pack(&a_packer, &a_dense, m, k, 1);
    let a_sparse_packed = pack(&a_packer, &a_sparse, m, k, 1);
    let b_packed = pack(&b_packer, &b_data, k, n, 0);

    let mut group = c.benchmark_group("mmm_sparse_opt");

    // 5. Benchmark Dense
    group.bench_function("dense_f32_4x4", |b| {
        b.iter(|| {
            unsafe {
                let specs = [
                    FusedKerSpec::AddMatMul {
                        k,
                        pa: a_dense_packed.as_ptr() as _,
                        pb: b_packed.as_ptr() as _,
                        packing: 0,
                    },
                    FusedKerSpec::Done
                ];
                (generic_f32_4x4.kernel)(&specs);
            }
        })
    });

    // 6. Benchmark Sparse
    group.bench_function("sparse_f32_4x4", |b| {
        b.iter(|| {
            unsafe {
                let specs = [
                    FusedKerSpec::AddMatMul {
                        k,
                        pa: a_sparse_packed.as_ptr() as _,
                        pb: b_packed.as_ptr() as _,
                        packing: 0,
                    },
                    FusedKerSpec::Done
                ];
                (generic_f32_4x4.kernel)(&specs);
            }
        })
    });
    
    group.finish();
}

criterion_group!(benches, bench_sparse_mmm);
criterion_main!(benches);
