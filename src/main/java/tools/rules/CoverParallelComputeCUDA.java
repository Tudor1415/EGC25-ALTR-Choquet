package tools.rules;

import java.util.Arrays;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

public class CoverParallelComputeCUDA {
    private static CUcontext context;

    static {
        JCudaDriver.setExceptionsEnabled(true);
        // Initialize the CUDA driver
        JCudaDriver.cuInit(0);

        // Obtain a handle to the device
        CUdevice device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);

        // Create a context for the device
        context = new CUcontext();
        JCudaDriver.cuCtxCreate(context, 0, device);
    }

    public static void runCudaSparseAnd(int[] indices1, int[] indices2, int[] resultIndices) {
        // Ensure the context is current
        JCudaDriver.cuCtxSetCurrent(context);

        int nnz1 = indices1.length;
        int nnz2 = indices2.length;

        // Ensure indices are sorted
        Arrays.sort(indices1);
        Arrays.sort(indices2);

        // Allocate device memory for indices
        CUdeviceptr d_indices1 = new CUdeviceptr();
        CUdeviceptr d_indices2 = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(d_indices1, nnz1 * Sizeof.INT);
        JCudaDriver.cuMemAlloc(d_indices2, nnz2 * Sizeof.INT);
        JCudaDriver.cuMemcpyHtoD(d_indices1, Pointer.to(indices1), nnz1 * Sizeof.INT);
        JCudaDriver.cuMemcpyHtoD(d_indices2, Pointer.to(indices2), nnz2 * Sizeof.INT);

        // Allocate device memory for the result indices
        int maxResultSize = Math.min(nnz1, nnz2);
        CUdeviceptr d_resultIndices = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(d_resultIndices, maxResultSize * Sizeof.INT);

        // Allocate device memory for the size of the result
        CUdeviceptr d_resultSize = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(d_resultSize, Sizeof.INT);
        JCudaDriver.cuMemsetD32(d_resultSize, 0, 1);

        // Load the kernel
        CUmodule module = new CUmodule();
        JCudaDriver.cuModuleLoad(module, "src/main/java/tools/rules/kernels/set_intersection_kernel.ptx");
        CUfunction function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, "setIntersectionThrust");

        // Prepare parameters for the kernel
        Pointer kernelParameters = Pointer.to(
            Pointer.to(d_indices1),
            Pointer.to(new int[]{nnz1}),
            Pointer.to(d_indices2),
            Pointer.to(new int[]{nnz2}),
            Pointer.to(d_resultIndices),
            Pointer.to(d_resultSize)
        );

        int blockSize = 256;
        int gridSize = (int) Math.ceil((double) nnz1 / blockSize);

        JCudaDriver.cuLaunchKernel(function,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0, null,
                kernelParameters, null);

        JCudaDriver.cuCtxSynchronize();

        // Copy the size of the result back to host
        int[] h_resultSize = new int[1];
        JCudaDriver.cuMemcpyDtoH(Pointer.to(h_resultSize), d_resultSize, Sizeof.INT);

        // Copy the result indices back to host
        JCudaDriver.cuMemcpyDtoH(Pointer.to(resultIndices), d_resultIndices, h_resultSize[0] * Sizeof.INT);

        // Free device memory
        JCudaDriver.cuMemFree(d_indices1);
        JCudaDriver.cuMemFree(d_indices2);
        JCudaDriver.cuMemFree(d_resultIndices);
        JCudaDriver.cuMemFree(d_resultSize);
    }
}
