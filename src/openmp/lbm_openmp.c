/* Tunel de Vento Virtual 3D - Lattice Boltzmann Method (D3Q19)
 * VERSAO OpenMP - paraleliza entre os nucleos de CPU.
 *
 * Mesmo algoritmo da versao serial (mesma grade, mesmos parametros,
 * mesmo numero de passos), so que cada laco sobre a grade ganha
 * #pragma omp parallel for collapse(2): as 221 mil (ou mais) celulas
 * sao divididas entre as threads disponiveis, cada uma cuidando de uma
 * fatia, todas calculando ao mesmo tempo. Nenhuma etapa depende do
 * resultado de outra celula sendo calculado "ao vivo" nesse mesmo
 * passo - cada celula so usa dados do passo anterior, ja prontos - por
 * isso dividir entre threads e seguro sem nenhuma secao critica.
 *
 * Mesma estrutura por passo de tempo que a versao serial (colisao,
 * propagacao, condicoes de contorno) - ver lbm_serial.c para os
 * comentarios linha a linha de cada etapa.
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "../common/lbm_params.h"
#include "../common/lbm_io.h"

/* Funcao de distribuicao de equilibrio (Maxwell-Boltzmann discretizada)
 * para a direcao i, dada a densidade e a velocidade macroscopica de uma
 * celula. E para onde as 19 f_i da celula "relaxam" a cada colisao. */
static double feq(int i, double rho, double ux, double uy, double uz) {
    double cu = LBM_CX[i]*ux + LBM_CY[i]*uy + LBM_CZ[i]*uz;
    double uu = ux*ux + uy*uy + uz*uz;
    return W[i] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*uu);
}

/* Igual ao main() da versao serial, mas os lacos sobre a grade usam
 * #pragma omp parallel for pra distribuir as celulas entre as threads
 * de CPU disponiveis. Imprime o numero de threads usadas e o tempo
 * total no final, pra comparar com a versao serial. */
int main(void) {
    size_t ncells = (size_t)NCELLS;
    double *f  = malloc(ncells * Q * sizeof(double));
    double *fn = malloc(ncells * Q * sizeof(double));
    unsigned char *obstacle = malloc(ncells * sizeof(unsigned char));
    double *ux = malloc(ncells * sizeof(double));
    double *uy = malloc(ncells * sizeof(double));
    double *uz = malloc(ncells * sizeof(double));
    if (!f || !fn || !obstacle || !ux || !uy || !uz) {
        fprintf(stderr, "lbm_openmp: falha ao alocar memoria\n");
        return 1;
    }

    /* Inicializacao (paralela: cada thread "toca primeiro" a memoria que
     * vai usar nos passos seguintes - bom para afinidade em maquinas NUMA) */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < NZ; z++) {
        for (int y = 0; y < NY; y++) {
            for (int x = 0; x < NX; x++) {
                int obst = is_obstacle(x, y, z);
                obstacle[IDX3D(x,y,z)] = (unsigned char)obst;
                double ux0 = obst ? 0.0 : U_IN;
                for (int i = 0; i < Q; i++)
                    f[IDXF(x,y,z,i)] = feq(i, 1.0, ux0, 0.0, 0.0);
            }
        }
    }

    printf("Tunel de Vento LBM D3Q19 (OpenMP) - grade %dx%dx%d, %d passos, %d threads\n",
           NX, NY, NZ, N_STEPS, omp_get_max_threads());

    double t0 = omp_get_wtime();

    for (int step = 0; step < N_STEPS; step++) {
        /* 1. Colisao (BGK) em todas as celulas fluidas */
        #pragma omp parallel for collapse(2) schedule(static)
        for (int z = 0; z < NZ; z++) {
            for (int y = 0; y < NY; y++) {
                for (int x = 0; x < NX; x++) {
                    if (obstacle[IDX3D(x,y,z)]) continue;
                    double rho = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
                    for (int i = 0; i < Q; i++) {
                        double fi = f[IDXF(x,y,z,i)];
                        rho += fi;
                        mx += fi * LBM_CX[i];
                        my += fi * LBM_CY[i];
                        mz += fi * LBM_CZ[i];
                    }
                    double vx = mx/rho, vy = my/rho, vz = mz/rho;
                    for (int i = 0; i < Q; i++) {
                        double *fi = &f[IDXF(x,y,z,i)];
                        *fi -= (*fi - feq(i, rho, vx, vy, vz)) / TAU;
                    }
                }
            }
        }

        /* 2. Propagacao com bounce-back em obstaculo e paredes y/z
         *    (escritas em destinos disjuntos - ver comentario no topo) */
        #pragma omp parallel for collapse(2) schedule(static)
        for (int z = 0; z < NZ; z++) {
            for (int y = 0; y < NY; y++) {
                for (int x = 0; x < NX; x++) {
                    if (obstacle[IDX3D(x,y,z)]) continue;
                    for (int i = 0; i < Q; i++) {
                        int nx = x + LBM_CX[i], ny = y + LBM_CY[i], nz = z + LBM_CZ[i];
                        double val = f[IDXF(x,y,z,i)];
                        if (ny < 0 || ny >= NY || nz < 0 || nz >= NZ) {
                            fn[IDXF(x,y,z,LBM_OPP[i])] = val;
                        } else if (nx < 0 || nx >= NX) {
                            continue; /* tratado pelas condicoes de entrada/saida abaixo */
                        } else if (obstacle[IDX3D(nx,ny,nz)]) {
                            fn[IDXF(x,y,z,LBM_OPP[i])] = val;
                        } else {
                            fn[IDXF(nx,ny,nz,i)] = val;
                        }
                    }
                }
            }
        }

        /* 3a. Entrada de vento (x=0): velocidade constante */
        #pragma omp parallel for schedule(static)
        for (int z = 0; z < NZ; z++)
            for (int y = 0; y < NY; y++) {
                if (obstacle[IDX3D(0,y,z)]) continue;
                for (int i = 0; i < Q; i++)
                    fn[IDXF(0,y,z,i)] = feq(i, 1.0, U_IN, 0.0, 0.0);
            }

        /* 3b. Saida (x=NX-1): fluxo livre (zero-gradient) */
        #pragma omp parallel for schedule(static)
        for (int z = 0; z < NZ; z++)
            for (int y = 0; y < NY; y++) {
                if (obstacle[IDX3D(NX-1,y,z)]) continue;
                for (int i = 0; i < Q; i++)
                    fn[IDXF(NX-1,y,z,i)] = fn[IDXF(NX-2,y,z,i)];
            }

        /* Troca os buffers (ping-pong) para o proximo passo */
        double *tmp = f; f = fn; fn = tmp;

        if ((step + 1) % OUTPUT_EVERY == 0 || step == N_STEPS - 1) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int z = 0; z < NZ; z++) {
                for (int y = 0; y < NY; y++) {
                    for (int x = 0; x < NX; x++) {
                        size_t c = IDX3D(x,y,z);
                        if (obstacle[c]) { ux[c] = uy[c] = uz[c] = 0.0; continue; }
                        double rho = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
                        for (int i = 0; i < Q; i++) {
                            double fi = f[IDXF(x,y,z,i)];
                            rho += fi; mx += fi*LBM_CX[i]; my += fi*LBM_CY[i]; mz += fi*LBM_CZ[i];
                        }
                        ux[c] = mx/rho; uy[c] = my/rho; uz[c] = mz/rho;
                    }
                }
            }
            char fname[64];
            snprintf(fname, sizeof(fname), "slice_openmp_%05d.pgm", step + 1);
            write_velocity_slice_pgm(fname, ux, uy, uz, NX, NY, NZ, U_IN * 1.5);
            printf("passo %5d: slice salvo em %s\n", step + 1, fname);
        }
    }

    double t1 = omp_get_wtime();
    double total_s = t1 - t0;
    double ms_per_step = (total_s * 1000.0) / N_STEPS;
    printf("Tempo total: %.3f s | %.4f ms/passo | %.2f FPS (equivalente)\n",
           total_s, ms_per_step, 1000.0 / ms_per_step);

    free(f); free(fn); free(obstacle); free(ux); free(uy); free(uz);
    return 0;
}
