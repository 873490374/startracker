import numpy as np
from timeit import default_timer as timer
from numba import cuda, vectorize


@vectorize(['float32(float32, float32)'], target='cuda')
def VectorAdd(a, b):
    return a + b


def main():

    N = 32000000  # Number of elements per Array
    limit = 12582720  # 65535

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.zeros(N, dtype=np.float32)

    start_time = timer()
    times_over_limit = 1
    if N > limit:
        times_over_limit = int(np.ceil(N/limit))
    for i in range(times_over_limit):
        if i > 1:
            start = 0
        else:
            start = i*limit
        if i*times_over_limit > N:
            end = N
        else:
            end = (i+1)*limit
        C[start:end] = VectorAdd(A[start:end], B[start:end])
    vector_add_time = timer() - start_time

    assert all(C == np.ones(N, dtype=np.float32)*2)

    print('C[:5] = ' + str(C[:5]))
    print('C[-5:] = ' + str(C[-5:]))

    print('VectorAdd took %f seconds' % vector_add_time)


if __name__ == '__main__':
    main()


def get_gpu_info():

    import numba.cuda.api, numba.cuda.cudadrv.libs
    numba.cuda.cudadrv.libs.test()
    numba.cuda.api.detect()

    gpu = cuda.get_current_device()

    print("name = %s" % gpu.name)
    print("maxThreadsPerBlock = %s" % str(gpu.MAX_THREADS_PER_BLOCK))
    print("maxBlockDimX = %s" % str(gpu.MAX_BLOCK_DIM_X))
    print("maxBlockDimY = %s" % str(gpu.MAX_BLOCK_DIM_Y))
    print("maxBlockDimZ = %s" % str(gpu.MAX_BLOCK_DIM_Z))
    print("maxGridDimX = %s" % str(gpu.MAX_GRID_DIM_X))
    print("maxGridDimY = %s" % str(gpu.MAX_GRID_DIM_Y))
    print("maxGridDimZ = %s" % str(gpu.MAX_GRID_DIM_Z))
    print(
        "maxSharedMemoryPerBlock = %s" % str(gpu.MAX_SHARED_MEMORY_PER_BLOCK))
    print("asyncEngineCount = %s" % str(gpu.ASYNC_ENGINE_COUNT))
    print("canMapHostMemory = %s" % str(gpu.CAN_MAP_HOST_MEMORY))
    print("multiProcessorCount = %s" % str(gpu.MULTIPROCESSOR_COUNT))
    print("warpSize = %s" % str(gpu.WARP_SIZE))
    print("unifiedAddressing = %s" % str(gpu.UNIFIED_ADDRESSING))
    print("pciBusID = %s" % str(gpu.PCI_BUS_ID))
    print("pciDeviceID = %s" % str(gpu.PCI_DEVICE_ID))
