#include "internal.h"
#include "containers/safe_call.hpp"

#if __CUDA_ARCH__ < 300
__inline__ __device__
float __shfl_down(float val, int offset, int width = 32)
{
    static __shared__ float shared[MAX_THREADS];
    int lane = threadIdx.x % 32;
    shared[threadIdx.x] = val;
    __syncthreads();
    val = (lane + offset < width) ? shared[threadIdx.x + offset] : 0;
    __syncthreads();
    return val;
}
#endif

#if __CUDA_ARCH__ < 350
template<typename T>
__device__ __forceinline__ T __ldg(const T* ptr)
{
    return *ptr;
}
#endif

__inline__  __device__ void warpReduceSum(jtjjtr & val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        #pragma unroll
        for(int i = 0; i < 29; i++)
        {
            val[i] += __shfl_down(val[i], offset);
        }
    }
}

__inline__  __device__ void blockReduceSum(jtjjtr & val)
{
    static __shared__ jtjjtr shared[32];

    int lane = threadIdx.x % warpSize;

    int wid = threadIdx.x / warpSize;

    warpReduceSum(val);

    //write reduced value to shared memory
    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();

    const jtjjtr zero = {0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0};

    //ensure we only grab a value from shared memory if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

    if(wid == 0)
    {
        warpReduceSum(val);
    }
}

__global__ void reduceSum(jtjjtr * in, jtjjtr * out, int N)
{
    jtjjtr sum = {0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0};

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum += in[i];
    }

    blockReduceSum(sum);

    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = sum;
    }
}

struct ICPReduction
{
    Eigen::Matrix<float, 3, 3, Eigen::DontAlign> R_prev_curr;
    Eigen::Matrix<float, 3, 1, Eigen::DontAlign> t_prev_curr;

    PtrStep<float> vmap_curr;
    PtrStep<float> nmap_curr;

    Intr intr;

    PtrStep<float> vmap_prev;
    PtrStep<float> nmap_prev;

    float distThres;
    float angleThres;

    int cols;
    int rows;
    int N;

    jtjjtr * out;

    __device__ __forceinline__ bool
    search (const int & x,
            const int & y,
            Eigen::Matrix<float,3,1,Eigen::DontAlign>& n,
            Eigen::Matrix<float,3,1,Eigen::DontAlign>& d,
            Eigen::Matrix<float,3,1,Eigen::DontAlign>& s) const
    {
        const Eigen::Matrix<float,3,1,Eigen::DontAlign> v_curr(vmap_curr.ptr(y)[x],
                                                               vmap_curr.ptr(y + rows)[x],
                                                               vmap_curr.ptr(y + 2 * rows)[x]);

        const Eigen::Matrix<float,3,1,Eigen::DontAlign> v_curr_in_prev = R_prev_curr * v_curr + t_prev_curr;

        const Eigen::Matrix<int,2,1,Eigen::DontAlign> p_curr_in_prev(__float2int_rn(v_curr_in_prev(0) * intr.fx / v_curr_in_prev(2) + intr.cx),
                                                                     __float2int_rn(v_curr_in_prev(1) * intr.fy / v_curr_in_prev(2) + intr.cy));

        if(p_curr_in_prev(0) < 0 || p_curr_in_prev(1) < 0 || p_curr_in_prev(0) >= cols || p_curr_in_prev(1) >= rows || v_curr_in_prev(2) < 0)
            return false;

        const Eigen::Matrix<float,3,1,Eigen::DontAlign> v_prev(__ldg(&vmap_prev.ptr(p_curr_in_prev(1))[p_curr_in_prev(0)]),
                                                               __ldg(&vmap_prev.ptr(p_curr_in_prev(1) + rows)[p_curr_in_prev(0)]),
                                                               __ldg(&vmap_prev.ptr(p_curr_in_prev(1) + 2 * rows)[p_curr_in_prev(0)]));

        const Eigen::Matrix<float,3,1,Eigen::DontAlign> n_curr(nmap_curr.ptr(y)[x],
                                                               nmap_curr.ptr(y + rows)[x],
                                                               nmap_curr.ptr(y + 2 * rows)[x]);

        const Eigen::Matrix<float,3,1,Eigen::DontAlign> n_curr_in_prev = R_prev_curr * n_curr;

        const Eigen::Matrix<float,3,1,Eigen::DontAlign> n_prev(__ldg(&nmap_prev.ptr(p_curr_in_prev(1))[p_curr_in_prev(0)]),
                                                               __ldg(&nmap_prev.ptr(p_curr_in_prev(1) + rows)[p_curr_in_prev(0)]),
                                                               __ldg(&nmap_prev.ptr(p_curr_in_prev(1) + 2 * rows)[p_curr_in_prev(0)]));

        const float dist = (v_prev - v_curr_in_prev).norm();
        const float sine = n_curr_in_prev.cross(n_prev).norm();

        n = n_prev;
        d = v_prev;
        s = v_curr_in_prev;

        return (sine < angleThres && dist <= distThres && !isnan(n_curr(0)) && !isnan(n_prev(0)));
    }

    //And now for some template metaprogramming magic
    template<int outer, int inner, int end>
    struct SquareUpperTriangularProduct
    {
        __device__ __forceinline__ static void apply(jtjjtr & values, const float (&rows)[end + 1])
        {
            values[((end + 1) * outer) + inner - (outer * (outer + 1) / 2)] = rows[outer] * rows[inner];

            SquareUpperTriangularProduct<outer, inner + 1, end>::apply(values, rows);
        }
    };

    //Inner loop base
    template<int outer, int end>
    struct SquareUpperTriangularProduct<outer, end, end>
    {
        __device__ __forceinline__ static void apply(jtjjtr & values, const float (&rows)[end + 1])
        {
            values[((end + 1) * outer) + end - (outer * (outer + 1) / 2)] = rows[outer] * rows[end];

            SquareUpperTriangularProduct<outer + 1, outer + 1, end>::apply(values, rows);
        }
    };

    //Outer loop base
    template<int end>
    struct SquareUpperTriangularProduct<end, end, end>
    {
        __device__ __forceinline__ static void apply(jtjjtr & values, const float (&rows)[end + 1])
        {
            values[((end + 1) * end) + end - (end * (end + 1) / 2)] = rows[end] * rows[end];
        }
    };

    __device__ __forceinline__ void
    operator () () const
    {
        jtjjtr sum = {0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0};

        SquareUpperTriangularProduct<0, 0, 6> sutp;

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            int y = i / cols;
            int x = i - (y * cols);

            Eigen::Matrix<float,3,1,Eigen::DontAlign> n, d, s;

            bool found = search(x, y, n, d, s);

            float row[7] = {0, 0, 0, 0, 0, 0, 0};

            if(found)
            {
                *(Eigen::Matrix<float,3,1,Eigen::DontAlign>*)&row[0] = n;
                *(Eigen::Matrix<float,3,1,Eigen::DontAlign>*)&row[3] = s.cross(n);
                row[6] = n.dot(d - s);
            }

            jtjjtr values;

            sutp.apply(values, row);

            sum += values;
        }

        blockReduceSum(sum);

        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = sum;
        }
    }
};

__global__ void icpKernel(const ICPReduction icp)
{
    icp();
}

void icpStep(const Eigen::Matrix<float,3,3,Eigen::DontAlign> & R_prev_curr,
             const Eigen::Matrix<float,3,1,Eigen::DontAlign> & t_prev_curr,
             const DeviceArray2D<float>& vmap_curr,
             const DeviceArray2D<float>& nmap_curr,
             const Intr& intr,
             const DeviceArray2D<float>& vmap_prev,
             const DeviceArray2D<float>& nmap_prev,
             float distThres,
             float angleThres,
             DeviceArray<jtjjtr> & sum,
             DeviceArray<jtjjtr> & out,
             float * matrixA_host,
             float * vectorB_host,
             float * residual_host,
             int threads, int blocks)
{
    int cols = vmap_curr.cols ();
    int rows = vmap_curr.rows () / 3;

    ICPReduction icp;

    icp.R_prev_curr = R_prev_curr;
    icp.t_prev_curr = t_prev_curr;

    icp.vmap_curr = vmap_curr;
    icp.nmap_curr = nmap_curr;

    icp.intr = intr;

    icp.vmap_prev = vmap_prev;
    icp.nmap_prev = nmap_prev;

    icp.distThres = distThres;
    icp.angleThres = angleThres;

    icp.cols = cols;
    icp.rows = rows;

    icp.N = cols * rows;
    icp.out = sum;

    icpKernel<<<blocks, threads>>>(icp);

    reduceSum<<<1, MAX_THREADS>>>(sum, out, blocks);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    float host_data[32];
    out.download((jtjjtr *)&host_data[0]);

    int shift = 0;
    for (int i = 0; i < 6; ++i)  //rows
    {
        for (int j = i; j < 7; ++j)    // cols + b
        {
            float value = host_data[shift++];
            if (j == 6)       // vector b
                vectorB_host[i] = value;
            else
                matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
        }
    }

    residual_host[0] = host_data[27];
    residual_host[1] = host_data[28];
}
