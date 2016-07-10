#include "internal.h"
#include "vector_math.hpp"
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

__inline__  __device__ jtjjtr warpReduceSum(jtjjtr val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val.aa += __shfl_down(val.aa, offset);
        val.ab += __shfl_down(val.ab, offset);
        val.ac += __shfl_down(val.ac, offset);
        val.ad += __shfl_down(val.ad, offset);
        val.ae += __shfl_down(val.ae, offset);
        val.af += __shfl_down(val.af, offset);
        val.ag += __shfl_down(val.ag, offset);

        val.bb += __shfl_down(val.bb, offset);
        val.bc += __shfl_down(val.bc, offset);
        val.bd += __shfl_down(val.bd, offset);
        val.be += __shfl_down(val.be, offset);
        val.bf += __shfl_down(val.bf, offset);
        val.bg += __shfl_down(val.bg, offset);

        val.cc += __shfl_down(val.cc, offset);
        val.cd += __shfl_down(val.cd, offset);
        val.ce += __shfl_down(val.ce, offset);
        val.cf += __shfl_down(val.cf, offset);
        val.cg += __shfl_down(val.cg, offset);

        val.dd += __shfl_down(val.dd, offset);
        val.de += __shfl_down(val.de, offset);
        val.df += __shfl_down(val.df, offset);
        val.dg += __shfl_down(val.dg, offset);

        val.ee += __shfl_down(val.ee, offset);
        val.ef += __shfl_down(val.ef, offset);
        val.eg += __shfl_down(val.eg, offset);

        val.ff += __shfl_down(val.ff, offset);
        val.fg += __shfl_down(val.fg, offset);

        val.residual += __shfl_down(val.residual, offset);
        val.inliers += __shfl_down(val.inliers, offset);
    }

    return val;
}

__inline__  __device__ jtjjtr blockReduceSum(jtjjtr val)
{
    static __shared__ jtjjtr shared[32];

    int lane = threadIdx.x % warpSize;

    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

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
        val = warpReduceSum(val);
    }

    return val;
}

__global__ void reduceSum(jtjjtr * in, jtjjtr * out, int N)
{
    jtjjtr sum = {0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0};

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum.add(in[i]);
    }

    sum = blockReduceSum(sum);

    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = sum;
    }
}

struct ICPReduction
{
    Mat33 R_prev_curr;
    float3 t_prev_curr;

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
    search (int & x, int & y, float3& n, float3& d, float3& s) const
    {
        float3 v_curr;
        v_curr.x = vmap_curr.ptr (y       )[x];
        v_curr.y = vmap_curr.ptr (y + rows)[x];
        v_curr.z = vmap_curr.ptr (y + 2 * rows)[x];

        float3 v_curr_in_prev = R_prev_curr * v_curr + t_prev_curr;

        int2 ukr;
        ukr.x = __float2int_rn (v_curr_in_prev.x * intr.fx / v_curr_in_prev.z + intr.cx);
        ukr.y = __float2int_rn (v_curr_in_prev.y * intr.fy / v_curr_in_prev.z + intr.cy);

        if(ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || v_curr_in_prev.z < 0)
            return false;

        float3 v_prev;
        v_prev.x = __ldg(&vmap_prev.ptr (ukr.y       )[ukr.x]);
        v_prev.y = __ldg(&vmap_prev.ptr (ukr.y + rows)[ukr.x]);
        v_prev.z = __ldg(&vmap_prev.ptr (ukr.y + 2 * rows)[ukr.x]);

        float3 n_curr;
        n_curr.x = nmap_curr.ptr (y)[x];
        n_curr.y = nmap_curr.ptr (y + rows)[x];
        n_curr.z = nmap_curr.ptr (y + 2 * rows)[x];

        float3 n_curr_in_prev = R_prev_curr * n_curr;

        float3 n_prev;
        n_prev.x = __ldg(&nmap_prev.ptr (ukr.y)[ukr.x]);
        n_prev.y = __ldg(&nmap_prev.ptr (ukr.y + rows)[ukr.x]);
        n_prev.z = __ldg(&nmap_prev.ptr (ukr.y + 2 * rows)[ukr.x]);

        float dist = norm(v_prev - v_curr_in_prev);
        float sine = norm(cross(n_curr_in_prev, n_prev));

        n = n_prev;
        d = v_prev;
        s = v_curr_in_prev;

        return (sine < angleThres && dist <= distThres && !isnan(n_curr.x) && !isnan(n_prev.x));
    }

    __device__ __forceinline__ jtjjtr
    getProducts(int & i) const
    {
        int y = i / cols;
        int x = i - (y * cols);

        float3 n, d, s;

        bool found_coresp = search (x, y, n, d, s);

        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        if(found_coresp)
        {
            *(float3*)&row[0] = n;
            *(float3*)&row[3] = cross(s, n);
            row[6] = dot(n, d - s);
        }

        jtjjtr values = {row[0] * row[0],
                         row[0] * row[1],
                         row[0] * row[2],
                         row[0] * row[3],
                         row[0] * row[4],
                         row[0] * row[5],
                         row[0] * row[6],

                         row[1] * row[1],
                         row[1] * row[2],
                         row[1] * row[3],
                         row[1] * row[4],
                         row[1] * row[5],
                         row[1] * row[6],

                         row[2] * row[2],
                         row[2] * row[3],
                         row[2] * row[4],
                         row[2] * row[5],
                         row[2] * row[6],

                         row[3] * row[3],
                         row[3] * row[4],
                         row[3] * row[5],
                         row[3] * row[6],

                         row[4] * row[4],
                         row[4] * row[5],
                         row[4] * row[6],

                         row[5] * row[5],
                         row[5] * row[6],

                         row[6] * row[6],
                         found_coresp};

        return values;
    }

    __device__ __forceinline__ void
    operator () () const
    {
        jtjjtr sum = {0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0};

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            jtjjtr val = getProducts(i);

            sum.add(val);
        }

        sum = blockReduceSum(sum);

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

void icpStep(const Mat33& R_prev_curr,
             const float3& t_prev_curr,
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
