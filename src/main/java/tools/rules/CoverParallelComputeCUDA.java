package tools.rules;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

public class CoverParallelComputeCUDA {
    private static CUcontext context;

    static {
        JCudaDriver.setExceptionsEnabled(true);
        // Initialize the CUDA driver
        JCudaDriver.cuInit(0);

        // Obtain the number of devices
        int deviceCountArray[] = {0};
        JCudaDriver.cuDeviceGetCount(deviceCountArray);
        int deviceCount = deviceCountArray[0];
        if (deviceCount == 0) {
            throw new RuntimeException("No CUDA devices found.");
        }

        // Obtain a handle to the device
        CUdevice device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);

        // Create a context for the device
        context = new CUcontext();
        JCudaDriver.cuCtxCreate(context, 0, device);
    }

    public static void runCudaKernel(int[] cover1, int[] cover2, int[] result, int size) {
        // Ensure the context is current
        JCudaDriver.cuCtxSetCurrent(context);

        // Load the module and proceed as before
        CUmodule module = new CUmodule();
        JCudaDriver.cuModuleLoad(module, "/home/tudor/Documents/GITHUB/EGC25-ALTR-Choquet/src/main/java/tools/rules/kernels/bitwise_and_kernel.ptx");

        CUfunction function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, "computeAndOperation");

        CUdeviceptr deviceCover1 = new CUdeviceptr();
        CUdeviceptr deviceCover2 = new CUdeviceptr();
        CUdeviceptr deviceResult = new CUdeviceptr();

        JCudaDriver.cuMemAlloc(deviceCover1, size * Sizeof.INT);
        JCudaDriver.cuMemAlloc(deviceCover2, size * Sizeof.INT);
        JCudaDriver.cuMemAlloc(deviceResult, size * Sizeof.INT);

        JCudaDriver.cuMemcpyHtoD(deviceCover1, Pointer.to(cover1), size * Sizeof.INT);
        JCudaDriver.cuMemcpyHtoD(deviceCover2, Pointer.to(cover2), size * Sizeof.INT);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(deviceCover1),
                Pointer.to(deviceCover2),
                Pointer.to(deviceResult),
                Pointer.to(new int[]{size})
        );

        int blockSize = 256;
        int gridSize = (int) Math.ceil((double) size / blockSize);

        JCudaDriver.cuLaunchKernel(function,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0, null,
                kernelParameters, null);

        JCudaDriver.cuCtxSynchronize();

        JCudaDriver.cuMemcpyDtoH(Pointer.to(result), deviceResult, size * Sizeof.INT);

        JCudaDriver.cuMemFree(deviceCover1);
        JCudaDriver.cuMemFree(deviceCover2);
        JCudaDriver.cuMemFree(deviceResult);
    }   
}
