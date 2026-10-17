#ifndef LBM_PARAMS_H
#define LBM_PARAMS_H

/* Modelo de velocidades D3Q19: 19 direções discretas por célula. */
#define Q 19

/* Tamanho da grade (células). Pode ser sobrescrito na compilação, ex.:
 *   gcc -DNX=64 -DNY=32 -DNZ=32 ...
 * para reproduzir as resoluções 32^3 / 64^3 / 128^3 citadas na proposta. */
#ifndef NX
#define NX 96
#endif
#ifndef NY
#define NY 48
#endif
#ifndef NZ
#define NZ 48
#endif

/* Parâmetros físicos/numéricos */
#ifndef TAU
#define TAU 0.6          /* tempo de relaxação BGK (viscosidade = (TAU-0.5)/3) */
#endif
#ifndef U_IN
#define U_IN 0.05         /* velocidade de entrada do vento (unidades de lattice) */
#endif
#ifndef N_STEPS
#define N_STEPS 500       /* número de passos de tempo simulados */
#endif
#ifndef OUTPUT_EVERY
#define OUTPUT_EVERY 50   /* intervalo (em passos) para salvar um slice de visualização */
#endif

/* Obstáculo: uma esfera sólida no interior do túnel de vento */
#define OBST_CX (NX/4)
#define OBST_CY (NY/2)
#define OBST_CZ (NZ/2)
#define OBST_R  (NY/6)

#define NCELLS (NX*NY*NZ)

/* Direções discretas do D3Q19: 1 repouso + 6 vizinhos de face + 12 de aresta */
static const int LBM_CX[Q] = {0,  1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
static const int LBM_CY[Q] = {0,  0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1};
static const int LBM_CZ[Q] = {0,  0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1};

static const double W[Q] = {
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

/* Direção oposta a cada i (usada na condição de bounce-back) */
static const int LBM_OPP[Q] = {0, 2,1, 4,3, 6,5, 10,9,8,7, 14,13,12,11, 18,17,16,15};

/* Índice linear de uma célula (x,y,z) e de sua i-ésima função de distribuição */
#define IDX3D(x,y,z)   ((x) + NX*((y) + NY*(z)))
#define IDXF(x,y,z,i)  (IDX3D((x),(y),(z))*Q + (i))

/* Testa se a célula (x,y,z) está dentro do obstáculo: uma esfera fixa,
 * usada pelas três versões de benchmark (serial/openmp/cuda). O
 * viewport interativo tem seu próprio sistema de formas trocáveis
 * (esfera, caixa, cilindro, etc.) e não usa esta função. */
static inline int is_obstacle(int x, int y, int z) {
    double dx = x - OBST_CX, dy = y - OBST_CY, dz = z - OBST_CZ;
    return (dx*dx + dy*dy + dz*dz) <= (double)(OBST_R*OBST_R);
}

#endif /* LBM_PARAMS_H */
