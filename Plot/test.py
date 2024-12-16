import itertools
import numpy as np
import cupy as cp
import os

# 创建 Pinned Memory Pool
pinned_memory_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

# 定义 CUDA 内核
generate_product_kernel = cp.ElementwiseKernel(
    'raw T output, raw T input_arrays, raw int32 input_shapes, int32 n_inputs',
    '',
    '''
    int idx = i;
    int pos = idx;
    for (int j = 0; j < n_inputs; ++j) {
        int stride = input_shapes[j];
        output[i * n_inputs + j] = input_arrays[j * stride + pos % stride];
        pos /= stride;
    }
    ''',
    'generate_product_kernel'
)


def generate_product_gpu(input_arrays, batch_size=10 ** 6):
    """
    在 GPU 上生成多个数组的笛卡尔积，并按批次返回结果。

    :param input_arrays: 输入数组列表
    :param batch_size: 每个批次的大小
    :return: 生成器，每次返回一个批次的笛卡尔积
    """
    input_shapes = [len(arr) for arr in input_arrays]
    n_inputs = len(input_arrays)
    total_product = np.prod(input_shapes)

    # 将输入数组展平并合并
    input_arrays_flattened = np.concatenate([arr for arr in input_arrays])

    # 使用 Pinned Memory 传输数据
    input_arrays_pinned = cp.cuda.alloc_pinned_memory(input_arrays_flattened.nbytes)
    input_arrays_pinned[:] = input_arrays_flattened
    input_arrays_gpu = cp.ndarray(input_arrays_flattened.shape, dtype=input_arrays_flattened.dtype,
                                  memptr=cp.cuda.MemoryPointer(cp.cuda.malloc(input_arrays_flattened.nbytes), 0))
    cp.cuda.runtime.memcpyAsync(input_arrays_gpu.data.ptr, input_arrays_pinned.ctypes.data,
                                input_arrays_flattened.nbytes, cp.cuda.runtime.memcpyDefault)

    input_shapes_gpu = cp.array(input_shapes, dtype=np.int32)

    for i in range(0, total_product, batch_size):
        batch_size_actual = min(batch_size, total_product - i)
        batch_gpu = cp.zeros((batch_size_actual, n_inputs), dtype=input_arrays[0].dtype)

        # 调用 CUDA 内核生成笛卡尔积
        generate_product_kernel(batch_gpu, input_arrays_gpu, input_shapes_gpu, n_inputs)

        yield batch_gpu


# 示例使用
if __name__ == "__main__":
    input_arrays = [np.array([1, 2]), np.array([3, 4, 5]), np.array([6, 7])]

    for batch in generate_product_gpu(input_arrays, batch_size=10):
        print(f'Batch:\n{batch}')
