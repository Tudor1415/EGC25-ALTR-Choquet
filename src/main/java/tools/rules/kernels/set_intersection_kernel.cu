#include <thrust/device_vector.h>
#include <thrust/set_operations.h>

extern "C"
void setIntersectionThrust(
    const int* indices1, int nnz1,
    const int* indices2, int nnz2,
    int* resultIndices, int* resultSize)
{
    thrust::device_ptr<const int> d_indices1(indices1);
    thrust::device_ptr<const int> d_indices2(indices2);
    thrust::device_ptr<int> d_resultIndices(resultIndices);

    thrust::device_vector<int> vec1(d_indices1, d_indices1 + nnz1);
    thrust::device_vector<int> vec2(d_indices2, d_indices2 + nnz2);

    thrust::device_vector<int> resultVec(std::min(nnz1, nnz2));

    auto end = thrust::set_intersection(
        vec1.begin(), vec1.end(),
        vec2.begin(), vec2.end(),
        resultVec.begin());

    int numElements = end - resultVec.begin();
    *resultSize = numElements;

    // Copy result to output array
    thrust::copy(resultVec.begin(), end, d_resultIndices);
}
