/* Tunel de Vento Virtual 3D - Lattice Boltzmann Method (D3Q19)
 * VERSAO CUDA - paraleliza entre os milhares de nucleos da GPU.
 *
 * Mesmo algoritmo das versoes serial/OpenMP (mesma grade, mesmos
 * parametros, mesmo numero de passos): aqui, em vez de dividir as
 * celulas entre poucas threads de CPU, cada celula (ou um pequeno grupo
 * delas, via "grid-stride loop") vira uma thread da GPU, todas rodando
 * ao mesmo tempo. Cada etapa do algoritmo (colisao, propagacao,
 * entrada, saida) e um "kernel" - uma funcao que a GPU executa em
 * paralelo para cada celula.
 *
 * Usa Unified Memory (cudaMallocManaged): um unico ponteiro que tanto a
 * CPU quanto a GPU enxergam, sem precisar copiar os dados manualmente
 * entre as duas - so avisamos com cudaMemPrefetchAsync pra onde os
 * dados devem ir antes de cada uso, pra evitar page faults.
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../common/lbm_params.h"
#include "../common/lbm_io.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "Erro CUDA em %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

/* Copias das constantes do lattice em memoria de constante da GPU
 * (populadas a partir dos arrays do host via cudaMemcpyToSymbol). */
__constant__ int    d_CX[Q];
__constant__ int    d_CY[Q];
__constant__ int    d_CZ[Q];
__constant__ double d_W[Q];
__constant__ int    d_OPP[Q];

/* Versao para GPU da funcao de equilibrio (identica em matematica a
 * feq_host, so usa as constantes d_CX/d_CY/d_CZ/d_W da memoria de
 * constante da GPU em vez dos arrays do host). */
__device__ inline double feq_device(int i, double rho, double ux, double uy, double uz) {
    double cu = d_CX[i]*ux + d_CY[i]*uy + d_CZ[i]*uz;
    double uu = ux*ux + uy*uy + uz*uz;
    return d_W[i] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*uu);
}

/* Converte o indice linear de uma celula (0..NCELLS-1) de volta para
 * suas coordenadas 3D (x,y,z) - o inverso do macro IDX3D. */
__device__ inline void idx_to_xyz(int idx, int *x, int *y, int *z) {
    *x = idx % NX;
    *y = (idx / NX) % NY;
    *z = idx / (NX * NY);
}

/* Colisao (BGK): grid-stride loop, uma thread cuida de uma ou mais celulas */
__global__ void collide_kernel(double *f, const unsigned char *obstacle) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < NCELLS;
         idx += blockDim.x * gridDim.x) {
        if (obstacle[idx]) continue;
        int base = idx * Q;
        double rho = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
        for (int i = 0; i < Q; i++) {
            double fi = f[base + i];
            rho += fi; mx += fi*d_CX[i]; my += fi*d_CY[i]; mz += fi*d_CZ[i];
        }
        double vx = mx/rho, vy = my/rho, vz = mz/rho;
        for (int i = 0; i < Q; i++) {
            double fi = f[base + i];
            f[base + i] = fi - (fi - feq_device(i, rho, vx, vy, vz)) / TAU;
        }
    }
}

/* Propagacao com bounce-back em obstaculo e paredes y/z */
__global__ void stream_kernel(const double *f, double *fn, const unsigned char *obstacle) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < NCELLS;
         idx += blockDim.x * gridDim.x) {
        if (obstacle[idx]) continue;
        int x, y, z;
        idx_to_xyz(idx, &x, &y, &z);
        int base = idx * Q;
        for (int i = 0; i < Q; i++) {
            int nx = x + d_CX[i], ny = y + d_CY[i], nz = z + d_CZ[i];
            double val = f[base + i];
            if (ny < 0 || ny >= NY || nz < 0 || nz >= NZ) {
                fn[base + d_OPP[i]] = val;
            } else if (nx < 0 || nx >= NX) {
                continue; /* tratado pelas condicoes de entrada/saida */
            } else {
                int nidx = nx + NX * (ny + NY * nz);
                if (obstacle[nidx]) {
                    fn[base + d_OPP[i]] = val;
                } else {
                    fn[nidx * Q + i] = val;
                }
            }
        }
    }
}

/* Entrada de vento (x=0): velocidade constante. Um thread por celula do plano YZ. */
__global__ void inlet_bc_kernel(double *fn, const unsigned char *obstacle) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NY * NZ) return;
    int y = t % NY, z = t / NY;
    int idx = 0 + NX * (y + NY * z);
    if (obstacle[idx]) return;
    int base = idx * Q;
    for (int i = 0; i < Q; i++)
        fn[base + i] = feq_device(i, 1.0, U_IN, 0.0, 0.0);
}

/* Saida (x=NX-1): fluxo livre (zero-gradient) */
__global__ void outlet_bc_kernel(double *fn, const unsigned char *obstacle) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NY * NZ) return;
    int y = t % NY, z = t / NY;
    int idx  = (NX-1) + NX * (y + NY * z);
    int idx2 = (NX-2) + NX * (y + NY * z);
    if (obstacle[idx]) return;
    for (int i = 0; i < Q; i++)
        fn[idx*Q + i] = fn[idx2*Q + i];
}

/* Extrai densidade/velocidade macroscopicas para visualizacao */
__global__ void macroscopic_kernel(const double *f, const unsigned char *obstacle,
                                    double *ux, double *uy, double *uz) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < NCELLS;
         idx += blockDim.x * gridDim.x) {
        if (obstacle[idx]) { ux[idx] = uy[idx] = uz[idx] = 0.0; continue; }
        int base = idx * Q;
        double rho = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
        for (int i = 0; i < Q; i++) {
            double fi = f[base + i];
            rho += fi; mx += fi*d_CX[i]; my += fi*d_CY[i]; mz += fi*d_CZ[i];
        }
        ux[idx] = mx/rho; uy[idx] = my/rho; uz[idx] = mz/rho;
    }
}

/* Mesma funcao de equilibrio, rodando na CPU (host) - usada so na
 * inicializacao dos dados, antes de qualquer kernel da GPU rodar. */
static double feq_host(int i, double rho, double ux, double uy, double uz) {
    double cu = LBM_CX[i]*ux + LBM_CY[i]*uy + LBM_CZ[i]*uz;
    double uu = ux*ux + uy*uy + uz*uz;
    return W[i] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*uu);
}

/* Prepara a GPU (copia as constantes do lattice, aloca memoria
 * unificada, inicializa e faz o prefetch dos dados), depois roda
 * N_STEPS passos lancando os kernels na ordem correta (colisao ->
 * propagacao -> entrada -> saida), salvando cortes de velocidade
 * periodicamente e imprimindo o tempo total no final - esse tempo e o
 * que se compara com as versoes serial e OpenMP pra calcular o
 * speedup. */
int main(void) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Tunel de Vento LBM D3Q19 (CUDA) - grade %dx%dx%d, %d passos\n", NX, NY, NZ, N_STEPS);
    printf("GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    CUDA_CHECK(cudaMemcpyToSymbol(d_CX,  LBM_CX,  sizeof(LBM_CX)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_CY,  LBM_CY,  sizeof(LBM_CY)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_CZ,  LBM_CZ,  sizeof(LBM_CZ)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_W,   W,   sizeof(W)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_OPP, LBM_OPP, sizeof(LBM_OPP)));

    size_t ncells = (size_t)NCELLS;
    double *f, *fn, *ux, *uy, *uz;
    unsigned char *obstacle;

    /* Unified Memory: um unico ponteiro acessivel tanto pela CPU quanto pela GPU */
    CUDA_CHECK(cudaMallocManaged(&f,  ncells * Q * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&fn, ncells * Q * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&ux, ncells * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&uy, ncells * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&uz, ncells * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&obstacle, ncells * sizeof(unsigned char)));

    /* Inicializacao na CPU (host) */
    for (int z = 0; z < NZ; z++)
        for (int y = 0; y < NY; y++)
            for (int x = 0; x < NX; x++) {
                int obst = is_obstacle(x, y, z);
                obstacle[IDX3D(x,y,z)] = (unsigned char)obst;
                double ux0 = obst ? 0.0 : U_IN;
                for (int i = 0; i < Q; i++)
                    f[IDXF(x,y,z,i)] = feq_host(i, 1.0, ux0, 0.0, 0.0);
            }

    /* Sabendo de antemao que f, fn e obstacle serao usados na GPU logo em
     * seguida, fazemos o prefetching assincrono deles antes do loop
     * principal (evita que cada kernel dispare page faults individuais). */
    CUDA_CHECK(cudaMemPrefetchAsync(f,  ncells * Q * sizeof(double), device, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(fn, ncells * Q * sizeof(double), device, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(obstacle, ncells * sizeof(unsigned char), device, 0));

    const int block   = 256; /* multiplo de 32 (tamanho do warp) */
    const int grid     = (NCELLS   + block - 1) / block;
    const int gridBC   = (NY * NZ + block - 1) / block;

    double t0 = now_seconds();

    for (int step = 0; step < N_STEPS; step++) {
        collide_kernel<<<grid, block>>>(f, obstacle);
        stream_kernel<<<grid, block>>>(f, fn, obstacle);
        inlet_bc_kernel<<<gridBC, block>>>(fn, obstacle);
        outlet_bc_kernel<<<gridBC, block>>>(fn, obstacle);
        CUDA_CHECK(cudaGetLastError());

        double *tmp = f; f = fn; fn = tmp;

        if ((step + 1) % OUTPUT_EVERY == 0 || step == N_STEPS - 1) {
            macroscopic_kernel<<<grid, block>>>(f, obstacle, ux, uy, uz);
            CUDA_CHECK(cudaMemPrefetchAsync(ux, ncells * sizeof(double), cudaCpuDeviceId, 0));
            CUDA_CHECK(cudaMemPrefetchAsync(uy, ncells * sizeof(double), cudaCpuDeviceId, 0));
            CUDA_CHECK(cudaMemPrefetchAsync(uz, ncells * sizeof(double), cudaCpuDeviceId, 0));
            CUDA_CHECK(cudaDeviceSynchronize());

            char fname[64];
            snprintf(fname, sizeof(fname), "slice_cuda_%05d.pgm", step + 1);
            write_velocity_slice_pgm(fname, ux, uy, uz, NX, NY, NZ, U_IN * 1.5);
            printf("passo %5d: slice salvo em %s\n", step + 1, fname);

            CUDA_CHECK(cudaMemPrefetchAsync(ux, ncells * sizeof(double), device, 0));
            CUDA_CHECK(cudaMemPrefetchAsync(uy, ncells * sizeof(double), device, 0));
            CUDA_CHECK(cudaMemPrefetchAsync(uz, ncells * sizeof(double), device, 0));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    double total_s = t1 - t0;
    double ms_per_step = (total_s * 1000.0) / N_STEPS;
    printf("Tempo total: %.3f s | %.4f ms/passo | %.2f FPS (equivalente)\n",
           total_s, ms_per_step, 1000.0 / ms_per_step);

    cudaFree(f); cudaFree(fn); cudaFree(ux); cudaFree(uy); cudaFree(uz); cudaFree(obstacle);
    return 0;
}
