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
    Mat33 Rcurr;
    float3 tcurr;

    PtrStep<float> vmap_curr;
    PtrStep<float> nmap_curr;

    Mat33 Rprev_inv;
    float3 tprev;

    Intr intr;

    PtrStep<float> vmap_g_prev;
    PtrStep<float> nmap_g_prev;

    float distThres;
    float angleThres;

    int cols;
    int rows;
    int N;

    jtjjtr * out;

    __device__ __forceinline__ bool
    search (int & x, int & y, float3& n, float3& d, float3& s) const
    {

        float3 vcurr;
        vcurr.x = vmap_curr.ptr (y       )[x];
        vcurr.y = vmap_curr.ptr (y + rows)[x];
        vcurr.z = vmap_curr.ptr (y + 2 * rows)[x];

        float3 vcurr_g = Rcurr * vcurr + tcurr;
        float3 vcurr_cp = Rprev_inv * (vcurr_g - tprev);         // prev camera coo space

        int2 ukr;         //projection
        ukr.x = __float2int_rn (vcurr_cp.x * intr.fx / vcurr_cp.z + intr.cx);      //4
        ukr.y = __float2int_rn (vcurr_cp.y * intr.fy / vcurr_cp.z + intr.cy);                      //4

        float3 vprev_g;
        vprev_g.x = __ldg(&vmap_g_prev.ptr (ukr.y       )[ukr.x]);
        vprev_g.y = __ldg(&vmap_g_prev.ptr (ukr.y + rows)[ukr.x]);
        vprev_g.z = __ldg(&vmap_g_prev.ptr (ukr.y + 2 * rows)[ukr.x]);

        float3 ncurr;
        ncurr.x = nmap_curr.ptr (y)[x];
        ncurr.y = nmap_curr.ptr (y + rows)[x];
        ncurr.z = nmap_curr.ptr (y + 2 * rows)[x];

        float3 ncurr_g = Rcurr * ncurr;

        float3 nprev_g;
        nprev_g.x =  __ldg(&nmap_g_prev.ptr (ukr.y)[ukr.x]);
        nprev_g.y = __ldg(&nmap_g_prev.ptr (ukr.y + rows)[ukr.x]);
        nprev_g.z = __ldg(&nmap_g_prev.ptr (ukr.y + 2 * rows)[ukr.x]);

        float dist = norm (vprev_g - vcurr_g);
        float sine = norm (cross (ncurr_g, nprev_g));

        n = nprev_g;
        d = vprev_g;
        s = vcurr_g;

        return (sine < angleThres && dist <= distThres && !isnan (ncurr.x) && !isnan (nprev_g.x) && !(ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || vcurr_cp.z < 0));
    }

    __device__ __forceinline__ jtjjtr
    getProducts(int & i) const
    {
        int y = i / cols;
        int x = i - (y * cols);

        float3 n_cp, d_cp, s_cp;

        bool found_coresp = search (x, y, n_cp, d_cp, s_cp);

        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        if(found_coresp)
        {
            s_cp = Rprev_inv * (s_cp - tprev);         // prev camera coo space
            d_cp = Rprev_inv * (d_cp - tprev);         // prev camera coo space
            n_cp = Rprev_inv * (n_cp);                // prev camera coo space

            *(float3*)&row[0] = n_cp;
            *(float3*)&row[3] = cross (s_cp, n_cp);
            row[6] = dot (n_cp, s_cp - d_cp);
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

void icpStep(const Mat33& Rcurr,
             const float3& tcurr,
             const DeviceArray2D<float>& vmap_curr,
             const DeviceArray2D<float>& nmap_curr,
             const Mat33& Rprev_inv,
             const float3& tprev,
             const Intr& intr,
             const DeviceArray2D<float>& vmap_g_prev,
             const DeviceArray2D<float>& nmap_g_prev,
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

    icp.Rcurr = Rcurr;
    icp.tcurr = tcurr;

    icp.vmap_curr = vmap_curr;
    icp.nmap_curr = nmap_curr;

    icp.Rprev_inv = Rprev_inv;
    icp.tprev = tprev;

    icp.intr = intr;

    icp.vmap_g_prev = vmap_g_prev;
    icp.nmap_g_prev = nmap_g_prev;

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
