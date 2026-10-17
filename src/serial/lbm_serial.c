/* Tunel de Vento Virtual 3D - Lattice Boltzmann Method (D3Q19)
 * VERSAO SERIAL (baseline) - um unico nucleo de CPU, sem paralelismo.
 *
 * Serve de referencia de tempo para calcular o speedup das versoes
 * OpenMP e CUDA: as tres fazem exatamente a mesma conta (colisao +
 * propagacao do LBM sobre a grade), so muda como o trabalho e
 * distribuido. Nao tem janela nem interatividade - roda ate o fim e
 * salva cortes (slices) de velocidade em disco a cada OUTPUT_EVERY
 * passos, alem de imprimir o tempo total no final.
 *
 * A cada passo de tempo, o main() faz:
 *   1. Colisao: cada celula relaxa suas 19 funcoes de distribuicao
 *      para o equilibrio local (feq).
 *   2. Propagacao: cada funcao de distribuicao se desloca para a
 *      celula vizinha na direcao que ela representa (com bounce-back
 *      nas paredes/obstaculo).
 *   3. Condicoes de contorno: velocidade fixa na entrada (x=0),
 *      fluxo livre na saida (x=NX-1).
 */
#include <stdio.h>
#include <stdlib.h>

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

/* Aloca a grade, inicializa o escoamento em repouso, e roda N_STEPS
 * passos de tempo (colisao + propagacao + condicoes de contorno),
 * salvando cortes de velocidade em disco periodicamente e imprimindo o
 * tempo total no final - esse tempo e o que se compara com as versoes
 * OpenMP e CUDA pra calcular o speedup. */
int main(void) {
    size_t ncells = (size_t)NCELLS;
    double *f  = malloc(ncells * Q * sizeof(double));
    double *fn = malloc(ncells * Q * sizeof(double));
    unsigned char *obstacle = malloc(ncells * sizeof(unsigned char));
    double *ux = malloc(ncells * sizeof(double));
    double *uy = malloc(ncells * sizeof(double));
    double *uz = malloc(ncells * sizeof(double));
    if (!f || !fn || !obstacle || !ux || !uy || !uz) {
        fprintf(stderr, "lbm_serial: falha ao alocar memoria\n");
        return 1;
    }

    /* Inicializacao: fluido em repouso acelerado ate U_IN, obstaculo marcado */
    for (int z = 0; z < NZ; z++)
        for (int y = 0; y < NY; y++)
            for (int x = 0; x < NX; x++) {
                int obst = is_obstacle(x, y, z);
                obstacle[IDX3D(x,y,z)] = (unsigned char)obst;
                double ux0 = obst ? 0.0 : U_IN;
                for (int i = 0; i < Q; i++)
                    f[IDXF(x,y,z,i)] = feq(i, 1.0, ux0, 0.0, 0.0);
            }

    printf("Tunel de Vento LBM D3Q19 (Serial) - grade %dx%dx%d, %d passos\n", NX, NY, NZ, N_STEPS);

    double t0 = now_seconds();

    for (int step = 0; step < N_STEPS; step++) {
        /* 1. Colisao (BGK) em todas as celulas fluidas */
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

        /* 2. Propagacao com bounce-back em obstaculo e paredes y/z */
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
        for (int z = 0; z < NZ; z++)
            for (int y = 0; y < NY; y++) {
                if (obstacle[IDX3D(0,y,z)]) continue;
                for (int i = 0; i < Q; i++)
                    fn[IDXF(0,y,z,i)] = feq(i, 1.0, U_IN, 0.0, 0.0);
            }

        /* 3b. Saida (x=NX-1): fluxo livre (zero-gradient) */
        for (int z = 0; z < NZ; z++)
            for (int y = 0; y < NY; y++) {
                if (obstacle[IDX3D(NX-1,y,z)]) continue;
                for (int i = 0; i < Q; i++)
                    fn[IDXF(NX-1,y,z,i)] = fn[IDXF(NX-2,y,z,i)];
            }

        /* Troca os buffers (ping-pong) para o proximo passo */
        double *tmp = f; f = fn; fn = tmp;

        if ((step + 1) % OUTPUT_EVERY == 0 || step == N_STEPS - 1) {
            for (int z = 0; z < NZ; z++)
                for (int y = 0; y < NY; y++)
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
            char fname[64];
            snprintf(fname, sizeof(fname), "slice_serial_%05d.pgm", step + 1);
            write_velocity_slice_pgm(fname, ux, uy, uz, NX, NY, NZ, U_IN * 1.5);
            printf("passo %5d: slice salvo em %s\n", step + 1, fname);
        }
    }

    double t1 = now_seconds();
    double total_s = t1 - t0;
    double ms_per_step = (total_s * 1000.0) / N_STEPS;
    printf("Tempo total: %.3f s | %.4f ms/passo | %.2f FPS (equivalente)\n",
           total_s, ms_per_step, 1000.0 / ms_per_step);

    free(f); free(fn); free(obstacle); free(ux); free(uy); free(uz);
    return 0;
}
