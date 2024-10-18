package tools.rules;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import com.zaxxer.sparsebits.SparseBitSet;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import tools.data.Dataset;
import tools.utils.SetUtil;

/**
 * Class for parallel computation of covers.
 */
public class CoverParallelCompute {
    private Dataset dataset;
    private Map<String, SparseBitSet> itemsMap;

    private Set<SparseBitSet> coversToCompute;
    private Set<int[]> coversToComputeGPU;

    boolean onGPU = true;

    private static CUcontext context;

    /**
     * Constructor for CoverParallelCompute.
     * 
     * @param dataset  The dataset.
     * @param itemsMap The map of items to SparseBitSet covers.
     */
    public CoverParallelCompute(Dataset dataset) {
        this.dataset = dataset;
        this.itemsMap = dataset.getItemsMap();
        initializeGPUContext();
    }

    /**
     * Initialize the CUDA context and check if GPU is available.
     */
    private void initializeGPUContext() {
        JCudaDriver.setExceptionsEnabled(true);
        // Initialize the CUDA driver
        JCudaDriver.cuInit(0);

        // Obtain the number of devices
        int[] deviceCountArray = { 0 };
        JCudaDriver.cuDeviceGetCount(deviceCountArray);
        int deviceCount = deviceCountArray[0];

        // Check if any devices are available
        if (deviceCount == 0 || dataset.getNbTransactions() < 10_000) {
            onGPU = false;
            return;
        }

        // If a device is available, set up the context
        onGPU = true;
        CUdevice device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);

        context = new CUcontext();
        JCudaDriver.cuCtxCreate(context, 0, device);
    }

    public boolean isOnGPU() {
        return onGPU;
    }

    public CUcontext getContext() {
        return context;
    }

    /**
     * Computes the cover in parallel for a given set of items.
     * 
     * @param itemsInSet The set of items.
     * @return The computed cover.
     */
    public SparseBitSet compute(Set<String> itemsInSet) {
        // Copy Items in Set Covers
        this.coversToCompute = copyItemsInSetCovers(itemsInSet);

        // Divide and Conquer Loop
        divideAndConquer(this.coversToCompute);

        // Return Final Cover
        return returnFinalCover(this.coversToCompute);
    }

    public int[] computeGPU(Set<String> itemsInSet) {
        if (!isOnGPU()) {
            throw new RuntimeException("GPU computation not recommended !");
        }

        // Copy Items in Set Covers
        this.coversToComputeGPU = copyItemsInSetCoversGPU(itemsInSet);

        // Divide and Conquer Loop
        divideAndConquerGPU(this.coversToComputeGPU);

        // Return Final Cover
        return returnFinalCoverGPU(this.coversToComputeGPU);
    }

    /**
     * Copies the covers of items in the given set.
     * 
     * @param itemsInSet The set of items.
     * @return The set of copied covers.
     */
    protected Set<SparseBitSet> copyItemsInSetCovers(Set<String> itemsInSet) {
        Set<SparseBitSet> coversToCompute = new HashSet<>();
        for (String itemValue : itemsInSet) {
            SparseBitSet originalCover = itemsMap.get(itemValue);
            SparseBitSet copiedCover = (originalCover != null) ? SetUtil.copyCover(originalCover) : new SparseBitSet();
            coversToCompute.add(copiedCover);
        }
        return coversToCompute;
    }

    protected Set<int[]> copyItemsInSetCoversGPU(Set<String> itemsInSet) {
        Set<int[]> coversToCompute = new HashSet<>();
        for (String itemValue : itemsInSet) {
            int[] cover = dataset.getIntArrayItemsMap().get(itemValue).clone();
            coversToCompute.add(cover);
        }
        return coversToCompute;
    }

    protected SparseBitSet intArrayToBitSet(int[] array) {
        SparseBitSet bitSet = new SparseBitSet();
        for (int i = 0; i < array.length; i++) {
            if (array[i] == 1) {
                bitSet.set(i); // Set the bit if the value in the array is 1
            }
        }
        return bitSet;
    }

    /**
     * Divides and conquers the computed covers until only one cover remains.
     *
     * @param coversToCompute The set of computed covers.
     */
    void divideAndConquer(Set<SparseBitSet> coversToCompute) {
        while (this.coversToCompute.size() > 1) {
            // Pair covers
            Set<SparseBitSet[]> coverPairs = pairCovers(this.coversToCompute);

            // Compute covers in parallel
            Set<SparseBitSet> newCoversToCompute = computeCoversParallel(coverPairs);

            // Update the computed covers for the next iteration
            this.coversToCompute = newCoversToCompute;
        }
    }

    void divideAndConquerGPU(Set<int[]> coversToCompute) {
        while (coversToCompute.size() > 1) {
            // Pair covers
            Set<int[][]> coverPairs = pairCoversGPU(coversToCompute);

            // Compute covers in parallel
            Set<int[]> newCoversToCompute = computeCoversParallelGPU(coverPairs);

            // Update the computed covers for the next iteration
            coversToCompute = newCoversToCompute;
        }

        // Update the class variable with the final result
        this.coversToComputeGPU = coversToCompute;
    }

    /**
     * Computes covers in parallel for the given cover pairs.
     *
     * @param coverPairs The set of cover pairs.
     * @return A set containing newly computed covers.
     */
    Set<SparseBitSet> computeCoversParallel(Set<SparseBitSet[]> coverPairs) {
        // Set to store newly computed covers
        Set<SparseBitSet> coversToCompute = new HashSet<>();

        coverPairs.parallelStream().forEach(pair -> {
            // Compute the bitwise AND operation for the pair
            SparseBitSet newCover;

            if (pair.length == 2)
                newCover = computeAndOperation(pair);
            else
                newCover = pair[0];

            synchronized (coversToCompute) {
                // Add the computed cover to the set
                coversToCompute.add(newCover);
            }
        });

        return coversToCompute;
    }

    Set<int[]> computeCoversParallelGPU(Set<int[][]> coverPairs) {
        // Set to store newly computed covers
        Set<int[]> coversToCompute = new HashSet<>();

        coverPairs.parallelStream().forEach(pair -> {
            // Compute the bitwise AND operation for the pair
            int[] newCover = new int[pair[0].length];

            if (pair.length == 2)
                computeAndOperationGPU(pair[0], pair[1], newCover, pair[0].length);
            else
                newCover = pair[0];

            synchronized (coversToCompute) {
                // Add the computed cover to the set
                coversToCompute.add(newCover);
            }
        });

        return coversToCompute;
    }

    public static void computeAndOperationGPU(int[] cover1, int[] cover2, int[] result, int size) {
        // Ensure the context is current
        JCudaDriver.cuCtxSetCurrent(context);

        // Load the module and proceed as before
        CUmodule module = new CUmodule();
        JCudaDriver.cuModuleLoad(module,
                "/home/tudor/Documents/GITHUB/EGC25-ALTR-Choquet/src/main/java/tools/rules/kernels/bitwise_and_kernel.ptx");

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
                Pointer.to(new int[] { size }));

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

    /**
     * Pair the covers in the provided set.
     *
     * @param coversToCompute The set of covers to pair.
     * @return A set containing pairs of covers.
     */
    Set<SparseBitSet[]> pairCovers(Set<SparseBitSet> coversToCompute) {
        Set<SparseBitSet[]> coverPairs = new HashSet<>();
        Iterator<SparseBitSet> iterator = coversToCompute.iterator();

        while (iterator.hasNext()) {
            SparseBitSet cover1 = iterator.next();
            if (iterator.hasNext()) {
                SparseBitSet cover2 = iterator.next();
                coverPairs.add(new SparseBitSet[] { cover1, cover2 });
            } else {
                coverPairs.add(
                        new SparseBitSet[] { cover1 });
            }
        }

        return coverPairs;
    }

    Set<int[][]> pairCoversGPU(Set<int[]> coversToCompute) {
        Set<int[][]> coverPairs = new HashSet<>();
        Iterator<int[]> iterator = coversToCompute.iterator();

        while (iterator.hasNext()) {
            int[] cover1 = iterator.next();
            if (iterator.hasNext()) {
                int[] cover2 = iterator.next();
                coverPairs.add(new int[][] { cover1, cover2 });
            } else {
                coverPairs.add(
                        new int[][] { cover1 });
            }
        }

        return coverPairs;
    }

    /**
     * Returns the final cover from the computed covers set.
     * 
     * @param coversToCompute The set of computed covers.
     * @return The final cover.
     */
    SparseBitSet returnFinalCover(Set<SparseBitSet> coversToCompute) {
        return coversToCompute.iterator().next();
    }

    int[] returnFinalCoverGPU(Set<int[]> coversToComputeGPU) {
        return coversToComputeGPU.iterator().next();
    }

    /**
     * Computes the bitwise AND operation between two covers.
     * 
     * @param pair The pair of covers.
     * @return The result of the AND operation.
     */
    SparseBitSet computeAndOperation(SparseBitSet[] pair) {
        SparseBitSet cover1 = pair[0];
        SparseBitSet cover2 = pair[1];
        cover1.and(cover2); // Perform bitwise AND operation in place
        return cover1;
    }
}
