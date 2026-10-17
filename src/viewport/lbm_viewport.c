/* Tunel de Vento Virtual 3D - Lattice Boltzmann Method (D3Q19)
 * VIEWPORT INTERATIVO (raylib) - demonstracao visual, nao o benchmark.
 *
 * Roda a mesma fisica LBM da versao OpenMP (mesmo lbm_step(), com os
 * mesmos #pragma omp parallel for), um passo por quadro, numa janela
 * 3D onde da pra orbitar a camera, arrastar o obstaculo e trocar sua
 * forma. A fumaca que aparece na tela e so uma visualizacao: milhares
 * de particulas "traçadoras" que leem a velocidade ja calculada pela
 * fisica (nao participam da conta em si) e se movem de acordo, pra
 * deixar visivel algo que normalmente seria invisivel (o campo de
 * velocidade do ar).
 *
 * Controles:
 *   Clique+arraste no obstaculo  -> move ele (recalcula o escoamento)
 *   Clique+arraste fora dele     -> orbita a camera
 *   Setas do teclado             -> alternativa pra mover o obstaculo
 *   TAB                          -> troca a forma do obstaculo
 *   [ ]                          -> numero de lados do prisma poligonal
 *   Scroll                       -> zoom
 *   ESPACO                       -> pausa/retoma a simulacao
 *   ESC                          -> fecha a janela
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <raylib.h>

#include "../common/lbm_params.h"

#define SCALE 0.08f           /* unidades de mundo por celula da grade */
#define N_PARTICLES 6000
/* Cada particula agora e desenhada como 3 "billboards" (quadrados sempre
 * virados pra camera) com uma textura de gradiente radial suave - opaca
 * no centro, totalmente transparente na borda - em vez de esferas solidas
 * de borda dura. Com transparencia somada, varias particulas proximas se
 * fundem numa nuvem continua em vez de parecerem bolinhas separadas, que
 * e o que realmente faz algo "parecer fumaca" em renderizacao em tempo
 * real (a mesma tecnica usada em jogos). O tamanho pode parecer grande
 * perto de uma esfera solida, mas e proposital: e a sobreposicao das
 * bordas suaves que cria o efeito de neblina continua. */
#define PARTICLE_SIZE (0.3f * SCALE)
/* Multiplicador puramente visual: a velocidade fisica (U_IN ~ 0.05
 * celulas/passo) deixaria a fumaca cruzando o tunel muito devagar para
 * uma demonstracao. Nao afeta a fisica simulada, so o quanto a particula
 * anda na tela a cada quadro. */
#define PARTICLE_SPEED_MULT 12.0
/* Pequena perturbacao aleatoria a cada quadro, proporcional a velocidade
 * local - sem isso o escoamento (laminar, baixo numero de Reynolds) fica
 * com linhas de corrente perfeitamente lisas e "em trilhos", o que nao
 * parece fumaca de verdade (que sempre tem pequenos turbilhoes). Isso e
 * so um efeito visual, nao faz parte da fisica do solver LBM. */
#define TURBULENCE 0.10
/* Comprimento do rastro (em celulas), interpolado entre MIN (regiao
 * parada) e MAX (regiao rapida) conforme a velocidade local. */
#define STREAK_MIN 0.6
#define STREAK_MAX 3.5

typedef struct { float x, y, z; float tx, ty, tz; } Particle;

static double *f, *fn;
static double *ux, *uy, *uz;
static unsigned char *obstacle, *prev_obstacle;
static Particle particles[N_PARTICLES];
static Color particle_colors[N_PARTICLES];

/* Posicao/raio do obstaculo em tempo de execucao (comecam iguais as
 * constantes de lbm_params.h, mas podem ser alteradas arrastando o
 * obstaculo com o botao direito do mouse). */
static double obst_cx = OBST_CX, obst_cy = OBST_CY, obst_cz = OBST_CZ, obst_r = OBST_R;

/* Local fixo de onde a fumaca "sai" (como o bocal de um tunel de vento
 * de verdade) - NAO acompanha o obstaculo quando ele e arrastado. A
 * fumaca reage ao obstaculo porque o campo de velocidade do LBM muda
 * conforme ele se move, nao porque a fonte se move junto. */
static double smoke_src_y = OBST_CY, smoke_src_z = OBST_CZ;

#define PI_CONST 3.14159265358979323846

typedef enum {
    SHAPE_SPHERE = 0,
    SHAPE_BOX,
    SHAPE_CYLINDER,
    SHAPE_PRISM,     /* poligono regular extrudado (3 lados = triangulo, 6 = hexagono, etc.) */
    SHAPE_CONE,      /* funil/cone, afunilando ao longo de X */
    SHAPE_RAMP,      /* rampa/cunha, alta atras e baixa na frente, ao longo de X */
    SHAPE_HELIX,     /* parafuso/broca: eixo central solido + lamina helicoidal */
    SHAPE_COUNT
} ShapeType;

static ShapeType shape_type = SHAPE_SPHERE;
static int shape_sides = 6; /* usado so por SHAPE_PRISM */

/* Nome em portugues de cada forma, usado so pro texto exibido na tela. */
static const char *shape_name(ShapeType s) {
    switch (s) {
        case SHAPE_SPHERE:   return "Esfera";
        case SHAPE_BOX:      return "Caixa";
        case SHAPE_CYLINDER: return "Cilindro";
        case SHAPE_PRISM:    return "Prisma poligonal";
        case SHAPE_CONE:     return "Funil/Cone";
        case SHAPE_RAMP:     return "Rampa";
        case SHAPE_HELIX:    return "Parafuso/broca";
        default:             return "?";
    }
}

/* Funcao de distribuicao de equilibrio (Maxwell-Boltzmann discretizada)
 * para a direcao i, dada a densidade e a velocidade macroscopica de uma
 * celula. E para onde as 19 f_i da celula "relaxam" a cada colisao. */
static double feq(int i, double rho, double ux_, double uy_, double uz_) {
    double cu = LBM_CX[i]*ux_ + LBM_CY[i]*uy_ + LBM_CZ[i]*uz_;
    double uu = ux_*ux_ + uy_*uy_ + uz_*uz_;
    return W[i] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*uu);
}

/* Testa se a celula (x,y,z) esta dentro do obstaculo, de acordo com a
 * forma atualmente selecionada (shape_type). Todas as formas sao
 * centradas em (obst_cx,obst_cy,obst_cz) e dimensionadas por obst_r. */
static int is_obstacle_rt(int x, int y, int z) {
    double dx = x - obst_cx, dy = y - obst_cy, dz = z - obst_cz;

    switch (shape_type) {
    case SHAPE_BOX:
        return fabs(dx) <= obst_r && fabs(dy) <= obst_r && fabs(dz) <= obst_r;

    case SHAPE_CYLINDER:
        /* redondo em XY, extremidades planas em Z */
        return (dx*dx + dy*dy) <= (obst_r*obst_r) && fabs(dz) <= obst_r;

    case SHAPE_PRISM: {
        /* poligono regular de shape_sides lados, no plano XY, extrudado em Z */
        if (fabs(dz) > obst_r) return 0;
        double rp = sqrt(dx*dx + dy*dy);
        if (rp < 1e-9) return 1;
        double theta = atan2(dy, dx);
        double seg = 2.0 * PI_CONST / shape_sides;
        double phi = fmod(theta, seg);
        if (phi < 0.0) phi += seg;
        phi -= seg / 2.0;
        double boundary = obst_r * cos(PI_CONST / shape_sides) / cos(phi);
        return rp <= boundary;
    }

    case SHAPE_CONE: {
        /* afunila ao longo de X: boca larga (raio obst_r) em dx=-obst_r,
         * bico estreito (raio 0.15*obst_r) em dx=+obst_r */
        if (fabs(dx) > obst_r) return 0;
        double t = (dx + obst_r) / (2.0 * obst_r);
        double r = obst_r * (1.0 - t) + (obst_r * 0.15) * t;
        return (dy*dy + dz*dz) <= r*r;
    }

    case SHAPE_RAMP: {
        /* cunha: altura total (topo = +obst_r) na parte de tras (dx=-obst_r),
         * descendo ate o chao (topo = -obst_r) na frente (dx=+obst_r) */
        if (fabs(dy) > obst_r) return 0;
        if (dz < -obst_r) return 0;
        double t = (dx + obst_r) / (2.0 * obst_r);
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        double top = obst_r * (1.0 - 2.0 * t);
        return dz <= top;
    }

    case SHAPE_HELIX: {
        /* parafuso/broca: eixo ao longo de X. Eixo central solido (rp <=
         * r_core) + uma lamina helicoidal que gira em torno dele conforme
         * x avanca (um "fio de rosca" de r_core ate obst_r). */
        if (fabs(dx) > obst_r) return 0;
        double rp = sqrt(dy*dy + dz*dz);
        double r_core = obst_r * 0.30;
        if (rp <= r_core) return 1;
        if (rp > obst_r) return 0;
        double theta = atan2(dz, dy);
        if (theta < 0.0) theta += 2.0 * PI_CONST;
        double pitch = obst_r * 1.2; /* distancia ao longo de X por volta completa */
        double turns = (dx + obst_r) / pitch;
        double phase = fmod(turns * 2.0 * PI_CONST, 2.0 * PI_CONST);
        double rel = fmod(theta - phase + 4.0 * PI_CONST, 2.0 * PI_CONST);
        double thread_width = PI_CONST * 0.6; /* largura angular da lamina */
        return rel <= thread_width;
    }

    case SHAPE_SPHERE:
    default:
        return (dx*dx + dy*dy + dz*dz) <= (obst_r*obst_r);
    }
}

/* Remarca a mascara de celulas solidas a partir de obst_cx/obst_cy/obst_cz/obst_r.
 * Celulas que ACABAM de deixar de ser obstaculo (eram solidas no passo
 * anterior, agora sao fluidas) tem suas distribuicoes reconstruidas do
 * zero, em repouso local (rho=1, u=0) - sem isso, elas carregariam o
 * valor congelado de quando ficaram solidas pela ultima vez, o que pode
 * gerar transientes numericos estranhos logo apos mover o obstaculo. */
static void mark_obstacle(void) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < NZ; z++) {
        for (int y = 0; y < NY; y++) {
            for (int x = 0; x < NX; x++) {
                int idx = IDX3D(x,y,z);
                unsigned char now = (unsigned char)is_obstacle_rt(x,y,z);
                if (prev_obstacle[idx] && !now) {
                    for (int i = 0; i < Q; i++)
                        f[IDXF(x,y,z,i)] = feq(i, 1.0, 0.0, 0.0, 0.0);
                    ux[idx] = uy[idx] = uz[idx] = 0.0;
                }
                obstacle[idx] = now;
                prev_obstacle[idx] = now;
            }
        }
    }
}

static void lbm_init(void) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < NZ; z++) {
        for (int y = 0; y < NY; y++) {
            for (int x = 0; x < NX; x++) {
                int obst = is_obstacle_rt(x, y, z);
                int idx = IDX3D(x,y,z);
                obstacle[idx] = (unsigned char)obst;
                double ux0 = obst ? 0.0 : U_IN;
                ux[idx] = ux0; uy[idx] = 0.0; uz[idx] = 0.0;
                for (int i = 0; i < Q; i++)
                    f[IDXF(x,y,z,i)] = feq(i, 1.0, ux0, 0.0, 0.0);
            }
        }
    }
}

/* Um passo completo: colisao + propagacao + condicoes de contorno.
 * A colisao ja calcula a velocidade macroscopica de cada celula para
 * relaxar para o equilibrio, entao aproveitamos e guardamos esse valor
 * em ux/uy/uz (usado pelas particulas) sem precisar de uma passada extra. */
static void lbm_step(void) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < NZ; z++) {
        for (int y = 0; y < NY; y++) {
            for (int x = 0; x < NX; x++) {
                int idx = IDX3D(x,y,z);
                if (obstacle[idx]) { ux[idx] = uy[idx] = uz[idx] = 0.0; continue; }
                double rho = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
                for (int i = 0; i < Q; i++) {
                    double fi = f[IDXF(x,y,z,i)];
                    rho += fi; mx += fi*LBM_CX[i]; my += fi*LBM_CY[i]; mz += fi*LBM_CZ[i];
                }
                double vx = mx/rho, vy = my/rho, vz = mz/rho;
                ux[idx] = vx; uy[idx] = vy; uz[idx] = vz;
                for (int i = 0; i < Q; i++) {
                    double *fi = &f[IDXF(x,y,z,i)];
                    *fi -= (*fi - feq(i, rho, vx, vy, vz)) / TAU;
                }
            }
        }
    }

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
                        continue;
                    } else if (obstacle[IDX3D(nx,ny,nz)]) {
                        fn[IDXF(x,y,z,LBM_OPP[i])] = val;
                    } else {
                        fn[IDXF(nx,ny,nz,i)] = val;
                    }
                }
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int z = 0; z < NZ; z++)
        for (int y = 0; y < NY; y++) {
            if (obstacle[IDX3D(0,y,z)]) continue;
            for (int i = 0; i < Q; i++)
                fn[IDXF(0,y,z,i)] = feq(i, 1.0, U_IN, 0.0, 0.0);
        }

    #pragma omp parallel for schedule(static)
    for (int z = 0; z < NZ; z++)
        for (int y = 0; y < NY; y++) {
            if (obstacle[IDX3D(NX-1,y,z)]) continue;
            for (int i = 0; i < Q; i++)
                fn[IDXF(NX-1,y,z,i)] = fn[IDXF(NX-2,y,z,i)];
        }

    double *tmp = f; f = fn; fn = tmp;
}

/* Converte uma velocidade normalizada (0 = parado, 1 = velocidade de
 * referencia U_IN*1.6, usado como "rapido" para fins de cor) numa cor
 * de "fumaca": azul escuro (parado) -> ciano -> branco (rapido). */
static Color smoke_color(double norm) {
    if (norm < 0.0) norm = 0.0;
    if (norm > 1.0) norm = 1.0;
    Color a = (Color){ 40, 50, 80, 255 };
    Color b = (Color){ 100, 190, 235, 255 };
    Color c = (Color){ 255, 255, 255, 255 };
    if (norm < 0.5) {
        double t = norm / 0.5;
        return (Color){
            (unsigned char)(a.r + (b.r-a.r)*t),
            (unsigned char)(a.g + (b.g-a.g)*t),
            (unsigned char)(a.b + (b.b-a.b)*t),
            230 };
    } else {
        double t = (norm - 0.5) / 0.5;
        return (Color){
            (unsigned char)(b.r + (c.r-b.r)*t),
            (unsigned char)(b.g + (c.g-b.g)*t),
            (unsigned char)(b.b + (c.b-b.b)*t),
            230 };
    }
}

/* Reposiciona a particula i numa faixa estreita ao redor do local FIXO
 * de emissao (smoke_src_y/smoke_src_z) - nao da posicao atual do
 * obstaculo, que pode estar em outro lugar depois de arrastado. Isso
 * imita o bocal fixo de um tunel de vento de verdade: a fumaca sempre
 * sai do mesmo lugar, e e o escoamento ao redor do obstaculo (onde quer
 * que ele esteja) que muda o comportamento dela. anywhere=1: tambem
 * sorteia x em qualquer ponto ao longo do tunel (usado so na
 * inicializacao, para o tunel ja nascer com fumaca espalhada ao longo
 * do comprimento). anywhere=0: sempre na entrada (x=0), como uma nova
 * golfada de fumaca injetada no vento. */
static void spawn_particle(int i, int anywhere) {
    double band = OBST_R * 0.9;
    if (band < 1.0) band = 1.0;

    for (int tries = 0; tries < 10; tries++) {
        int sx = anywhere ? GetRandomValue(0, NX-1) : 0;
        /* Amostragem circular (bocal redondo), nao uma caixa Y x Z:
         * angulo uniforme + raio com sqrt(uniforme) para distribuicao
         * uniforme pela AREA do circulo (sem isso, os pontos se
         * concentrariam demais no centro). */
        double ang = (GetRandomValue(0, 35999) / 100.0) * (PI_CONST / 180.0);
        double rad = sqrt(GetRandomValue(0, 1000) / 1000.0) * band;
        int sy = (int)(smoke_src_y + rad * cos(ang));
        int sz = (int)(smoke_src_z + rad * sin(ang));
        if (sy < 1) sy = 1; if (sy > NY-2) sy = NY-2;
        if (sz < 1) sz = 1; if (sz > NZ-2) sz = NZ-2;
        if (!is_obstacle_rt(sx, sy, sz)) {
            particles[i].x = sx + 0.5f;
            particles[i].y = sy + 0.5f;
            particles[i].z = sz + 0.5f;
            particles[i].tx = particles[i].x;
            particles[i].ty = particles[i].y;
            particles[i].tz = particles[i].z;
            return;
        }
    }
    particles[i].x = 0.5f; particles[i].y = 1.5f; particles[i].z = 1.5f;
    particles[i].tx = particles[i].x; particles[i].ty = particles[i].y; particles[i].tz = particles[i].z;
}

/* Amostra a velocidade do escoamento por interpolacao trilinear entre as
 * 8 celulas vizinhas do ponto (px,py,pz) - em vez de so a celula mais
 * proxima, o que deixa a trajetoria das particulas suave ao atravessar
 * de uma celula para outra. Celulas solidas contam como velocidade zero
 * e NAO tem seu peso redistribuido, entao a velocidade interpolada cai
 * suavemente a zero conforme o ponto se aproxima da superficie do
 * obstaculo (em vez de um corte abrupto). */
static void sample_velocity(double px, double py, double pz, double *ovx, double *ovy, double *ovz) {
    double gx = px - 0.5, gy = py - 0.5, gz = pz - 0.5;
    if (gx < 0) gx = 0; if (gy < 0) gy = 0; if (gz < 0) gz = 0;
    if (gx > NX-1) gx = NX-1; if (gy > NY-1) gy = NY-1; if (gz > NZ-1) gz = NZ-1;
    int x0 = (int)gx, y0 = (int)gy, z0 = (int)gz;
    int x1 = (x0+1 < NX) ? x0+1 : x0;
    int y1 = (y0+1 < NY) ? y0+1 : y0;
    int z1 = (z0+1 < NZ) ? z0+1 : z0;
    double fx = gx - x0, fy = gy - y0, fz = gz - z0;

    double vx = 0.0, vy = 0.0, vz = 0.0;
    for (int dz = 0; dz <= 1; dz++) {
        int zi = dz ? z1 : z0;
        double wz = dz ? fz : (1.0 - fz);
        for (int dy = 0; dy <= 1; dy++) {
            int yi = dy ? y1 : y0;
            double wy = dy ? fy : (1.0 - fy);
            for (int dx = 0; dx <= 1; dx++) {
                int xi = dx ? x1 : x0;
                double wx = dx ? fx : (1.0 - fx);
                int idx = IDX3D(xi,yi,zi);
                if (!obstacle[idx]) {
                    double w = wx * wy * wz;
                    vx += w * ux[idx]; vy += w * uy[idx]; vz += w * uz[idx];
                }
            }
        }
    }
    *ovx = vx; *ovy = vy; *ovz = vz;
}

/* Move cada particula pela velocidade local do escoamento, usando
 * integracao RK2 (ponto medio): amostra a velocidade na posicao atual,
 * da um meio-passo para estimar um ponto intermediario, amostra de novo
 * ali, e so entao avanca a posicao completa com essa segunda amostra.
 * Isso segue a curvatura do escoamento (por exemplo, contornando o
 * obstaculo) bem melhor do que avancar direto com a velocidade inicial
 * (Euler simples). Recoloca a particula na entrada quando ela sai do
 * dominio ou atravessa o obstaculo.
 *
 * Nao e paralelizado com OpenMP porque spawn_particle() usa o gerador de
 * numeros aleatorios do raylib, que nao e garantido thread-safe - mas o
 * custo e irrelevante perto de um passo de simulacao (milhares de
 * particulas vs. a grade inteira). */
static void update_particles(void) {
    for (int i = 0; i < N_PARTICLES; i++) {
        double px = particles[i].x, py = particles[i].y, pz = particles[i].z;

        double k1x, k1y, k1z;
        sample_velocity(px, py, pz, &k1x, &k1y, &k1z);

        double midx = px + 0.5 * k1x * PARTICLE_SPEED_MULT;
        double midy = py + 0.5 * k1y * PARTICLE_SPEED_MULT;
        double midz = pz + 0.5 * k1z * PARTICLE_SPEED_MULT;

        double k2x, k2y, k2z;
        sample_velocity(midx, midy, midz, &k2x, &k2y, &k2z);

        double mag = sqrt(k2x*k2x + k2y*k2y + k2z*k2z);

        /* Turbulencia: pequeno chacoalho aleatorio, maior onde o
         * escoamento e mais rapido (perto/atras do obstaculo) - quebra o
         * aspecto de "trilhos perfeitamente lisos" do escoamento laminar. */
        double jit = TURBULENCE * mag;
        double jx = (GetRandomValue(-1000, 1000) / 1000.0) * jit;
        double jy = (GetRandomValue(-1000, 1000) / 1000.0) * jit;
        double jz = (GetRandomValue(-1000, 1000) / 1000.0) * jit;

        particles[i].x = (float)(px + k2x * PARTICLE_SPEED_MULT + jx);
        particles[i].y = (float)(py + k2y * PARTICLE_SPEED_MULT + jy);
        particles[i].z = (float)(pz + k2z * PARTICLE_SPEED_MULT + jz);

        double norm = mag / (U_IN * 1.6);
        particle_colors[i] = smoke_color(norm);

        /* Rastro: um segmento curto atras da particula, apontando para
         * onde ela veio, com comprimento proporcional a velocidade local. */
        double inv_mag = mag > 1e-6 ? 1.0/mag : 0.0;
        double streak_len = STREAK_MIN + (STREAK_MAX - STREAK_MIN) * (norm > 1.0 ? 1.0 : norm);
        particles[i].tx = particles[i].x - (float)(k2x * inv_mag * streak_len);
        particles[i].ty = particles[i].y - (float)(k2y * inv_mag * streak_len);
        particles[i].tz = particles[i].z - (float)(k2z * inv_mag * streak_len);

        bool outside = particles[i].x < 0 || particles[i].x >= NX ||
                       particles[i].y < 1 || particles[i].y >= NY-1 ||
                       particles[i].z < 1 || particles[i].z >= NZ-1;
        bool inside_obst = is_obstacle_rt((int)particles[i].x, (int)particles[i].y, (int)particles[i].z);
        if (outside || inside_obst) spawn_particle(i, 0);
    }
}

/* Aloca tudo, abre a janela, e roda o loop principal: a cada quadro,
 * le entrada do usuario (camera/arraste/forma), avanca 1 passo da
 * fisica (se nao estiver pausado), atualiza as particulas de fumaca, e
 * desenha a cena. */
int main(void) {
    size_t ncells = (size_t)NCELLS;
    f  = malloc(ncells * Q * sizeof(double));
    fn = malloc(ncells * Q * sizeof(double));
    ux = malloc(ncells * sizeof(double));
    uy = malloc(ncells * sizeof(double));
    uz = malloc(ncells * sizeof(double));
    obstacle = malloc(ncells * sizeof(unsigned char));
    prev_obstacle = malloc(ncells * sizeof(unsigned char));
    if (!f || !fn || !ux || !uy || !uz || !obstacle || !prev_obstacle) {
        fprintf(stderr, "lbm_viewport: falha ao alocar memoria\n");
        return 1;
    }
    lbm_init();
    memcpy(prev_obstacle, obstacle, ncells * sizeof(unsigned char));

    const int screenW = 1100, screenH = 750;
    InitWindow(screenW, screenH, "Tunel de Vento Virtual 3D - LBM (OpenMP, viewport interativo)");
    SetTargetFPS(60);

    /* Textura de "fumaca": um circulo com gradiente radial, opaco no
     * centro e transparente na borda, usado como billboard para cada
     * particula (ver comentario de PARTICLE_SIZE). */
    Image smokeImg = GenImageColor(64, 64, (Color){255,255,255,0});
    ImageDrawCircleGradient(&smokeImg, (Vector2){32,32}, 30,
                             (Color){255,255,255,235}, (Color){255,255,255,0});
    Texture2D smokeTex = LoadTextureFromImage(smokeImg);
    UnloadImage(smokeImg);
    SetTextureFilter(smokeTex, TEXTURE_FILTER_BILINEAR);

    for (int i = 0; i < N_PARTICLES; i++) {
        spawn_particle(i, 1);
        particle_colors[i] = smoke_color(0.3);
    }

    Vector3 domainCenter = { NX*SCALE/2.0f, NZ*SCALE/2.0f, NY*SCALE/2.0f };

    Camera3D camera = { 0 };
    camera.target = domainCenter;
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float yaw = 50.0f * DEG2RAD, pitch = 30.0f * DEG2RAD;
    float dist = (NX > NY ? NX : NY) * SCALE * 1.7f;

    bool paused = false;
    bool dragging_obstacle = false;
    double sim_time_total = 0.0;
    int sim_steps_done = 0;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;

        Vector3 obstPosNow = { (float)(obst_cx*SCALE), (float)(obst_cz*SCALE), (float)(obst_cy*SCALE) };

        /* Clicar o botao esquerdo EM CIMA da esfera comeca a arrasta-la;
         * clicar em qualquer outro ponto da cena orbita a camera, como
         * antes. Soltar o botao sempre encerra o arraste. */
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            Ray ray = GetScreenToWorldRay(GetMousePosition(), camera);
            RayCollision hit = GetRayCollisionSphere(ray, obstPosNow, (float)(obst_r*SCALE));
            dragging_obstacle = hit.hit;
        }
        if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) dragging_obstacle = false;

        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !dragging_obstacle) {
            Vector2 d = GetMouseDelta();
            yaw   -= d.x * 0.006f;
            pitch += d.y * 0.006f;
            if (pitch > 1.5f) pitch = 1.5f;
            if (pitch < -1.5f) pitch = -1.5f;
        }
        dist -= GetMouseWheelMove() * (NX * SCALE * 0.08f);
        if (dist < 1.0f) dist = 1.0f;

        camera.position.x = camera.target.x + dist * cosf(pitch) * sinf(yaw);
        camera.position.y = camera.target.y + dist * sinf(pitch);
        camera.position.z = camera.target.z + dist * cosf(pitch) * cosf(yaw);

        bool obst_moved = false;
        if ((dragging_obstacle && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) ||
            IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            Ray ray = GetScreenToWorldRay(GetMousePosition(), camera);
            float planeY = (float)(obst_cz * SCALE); /* profundidade atual do obstaculo */
            if (fabsf(ray.direction.y) > 1e-6f) {
                float t = (planeY - ray.position.y) / ray.direction.y;
                if (t > 0.0f) {
                    float hitX = ray.position.x + ray.direction.x * t;
                    float hitZ = ray.position.z + ray.direction.z * t;
                    obst_cx = hitX / SCALE;
                    obst_cy = hitZ / SCALE;
                    obst_moved = true;
                }
            }
        }
        /* Alternativa por teclado, garantida a funcionar independente de
         * camera/raycasting - util se o arraste do botao direito nao
         * responder no seu ambiente (driver, mouse, modo remoto etc.). */
        double key_speed = 15.0 * GetFrameTime();
        if (IsKeyDown(KEY_LEFT))  { obst_cx -= key_speed; obst_moved = true; }
        if (IsKeyDown(KEY_RIGHT)) { obst_cx += key_speed; obst_moved = true; }
        if (IsKeyDown(KEY_UP))    { obst_cy -= key_speed; obst_moved = true; }
        if (IsKeyDown(KEY_DOWN))  { obst_cy += key_speed; obst_moved = true; }

        if (obst_moved) {
            /* Margem extra na entrada/saida (alem do proprio raio): sem
             * isso, o obstaculo pode encostar em x=0 e bloquear a entrada
             * de vento inteira (a condicao de contorno pula celulas
             * obstaculo) - exatamente onde a fumaca nasce. Bloqueada a
             * entrada, nao entra fluido novo e a fumaca para de fluir. */
            double margin_x = NX / 6.0;
            if (obst_cx < obst_r + margin_x) obst_cx = obst_r + margin_x;
            if (obst_cx > NX - 1 - obst_r - margin_x) obst_cx = NX - 1 - obst_r - margin_x;
            if (obst_cy < obst_r) obst_cy = obst_r;
            if (obst_cy > NY - 1 - obst_r) obst_cy = NY - 1 - obst_r;
        }

        /* TAB troca a forma do obstaculo; '[' e ']' mudam o numero de
         * lados do prisma poligonal (3=triangulo, 6=hexagono, mais lados
         * se aproxima de um cilindro "com curva"). */
        bool shape_changed = false;
        if (IsKeyPressed(KEY_TAB)) {
            shape_type = (ShapeType)((shape_type + 1) % SHAPE_COUNT);
            shape_changed = true;
        }
        if (shape_type == SHAPE_PRISM) {
            if (IsKeyPressed(KEY_LEFT_BRACKET)  && shape_sides > 3)  { shape_sides--; shape_changed = true; }
            if (IsKeyPressed(KEY_RIGHT_BRACKET) && shape_sides < 20) { shape_sides++; shape_changed = true; }
        }

        if (obst_moved || shape_changed) mark_obstacle();

        if (!paused) {
            double t0 = omp_get_wtime();
            lbm_step();
            sim_time_total += omp_get_wtime() - t0;
            sim_steps_done++;
            update_particles();
        }

        BeginDrawing();
            ClearBackground((Color){10, 10, 14, 255});
            BeginMode3D(camera);
                DrawCubeWires(domainCenter, NX*SCALE, NZ*SCALE, NY*SCALE, (Color){80,80,92,255});

                /* Cada particula = 3 billboards (sempre virados pra camera)
                 * interpolados entre a cauda e a cabeca do rastro, com a
                 * opacidade crescendo em direcao a cabeca - isso da um
                 * efeito de "puff" alongado que se funde com os vizinhos
                 * (ver comentario de PARTICLE_SIZE no topo do arquivo). */
                BeginBlendMode(BLEND_ALPHA);
                for (int i = 0; i < N_PARTICLES; i++) {
                    Color c = particle_colors[i];
                    Vector3 headW = { particles[i].x*SCALE, particles[i].z*SCALE, particles[i].y*SCALE };
                    Vector3 tailW = { particles[i].tx*SCALE, particles[i].tz*SCALE, particles[i].ty*SCALE };
                    for (int s = 0; s < 3; s++) {
                        float t = s / 2.0f;
                        Vector3 pos = {
                            tailW.x + (headW.x - tailW.x) * t,
                            tailW.y + (headW.y - tailW.y) * t,
                            tailW.z + (headW.z - tailW.z) * t
                        };
                        Color tint = { c.r, c.g, c.b, (unsigned char)(c.a * (0.3f + 0.7f*t)) };
                        DrawBillboard(camera, smokeTex, pos, PARTICLE_SIZE, tint);
                    }
                }
                EndBlendMode();

                /* Obstaculo desenhado por cima da fumaca, em cor viva, para
                 * ficar claro onde ele esta e se ele se moveu. A malha
                 * desenhada e so uma referencia visual aproximada da forma
                 * escolhida - a fisica usa sempre a mascara de celulas
                 * exata calculada por is_obstacle_rt(), nao esta malha. */
                Vector3 obstPos = { (float)(obst_cx*SCALE), (float)(obst_cz*SCALE), (float)(obst_cy*SCALE) };
                float wr = (float)(obst_r*SCALE);
                Color fillColor = (Color){230,90,40,255};
                Color wireColor = (Color){255,160,90,255};
                switch (shape_type) {
                case SHAPE_BOX:
                    DrawCubeV(obstPos, (Vector3){wr*2, wr*2, wr*2}, fillColor);
                    DrawCubeWiresV(obstPos, (Vector3){wr*2, wr*2, wr*2}, wireColor);
                    break;
                case SHAPE_CYLINDER:
                case SHAPE_PRISM: {
                    int sides = (shape_type == SHAPE_PRISM) ? shape_sides : 24;
                    Vector3 top = { obstPos.x, obstPos.y + wr, obstPos.z };
                    Vector3 bot = { obstPos.x, obstPos.y - wr, obstPos.z };
                    DrawCylinderEx(bot, top, wr, wr, sides, fillColor);
                    DrawCylinderWiresEx(bot, top, wr, wr, sides, wireColor);
                    break;
                }
                case SHAPE_CONE: {
                    Vector3 mouth = { obstPos.x - wr, obstPos.y, obstPos.z };
                    Vector3 spout = { obstPos.x + wr, obstPos.y, obstPos.z };
                    DrawCylinderEx(mouth, spout, wr, wr*0.15f, 24, fillColor);
                    DrawCylinderWiresEx(mouth, spout, wr, wr*0.15f, 24, wireColor);
                    break;
                }
                case SHAPE_RAMP:
                    /* sem primitiva pronta pra cunha no raylib; a caixa
                     * delimitadora serve de referencia visual, mas a fisica
                     * usa a forma de rampa exata (ver is_obstacle_rt). */
                    DrawCubeV(obstPos, (Vector3){wr*2, wr*2, wr*2}, Fade(fillColor, 0.5f));
                    DrawCubeWiresV(obstPos, (Vector3){wr*2, wr*2, wr*2}, wireColor);
                    break;
                case SHAPE_HELIX: {
                    /* sem primitiva pronta pra helice no raylib: desenha o
                     * eixo central solido (que e exato) e um contorno
                     * cilindrico no raio total, so como referencia visual -
                     * a lamina helicoidal em si so existe na fisica
                     * (ver is_obstacle_rt). */
                    Vector3 axisA = { obstPos.x - wr, obstPos.y, obstPos.z };
                    Vector3 axisB = { obstPos.x + wr, obstPos.y, obstPos.z };
                    DrawCylinderEx(axisA, axisB, wr*0.30f, wr*0.30f, 16, fillColor);
                    DrawCylinderWiresEx(axisA, axisB, wr, wr, 16, Fade(wireColor, 0.5f));
                    break;
                }
                case SHAPE_SPHERE:
                default:
                    DrawSphere(obstPos, wr, fillColor);
                    DrawSphereWires(obstPos, wr, 12, 12, wireColor);
                    break;
                }
            EndMode3D();

            DrawText("Tunel de Vento Virtual 3D - LBM D3Q19 (OpenMP)", 14, 12, 20, RAYWHITE);
            DrawText(TextFormat("Grade: %dx%dx%d  |  %d particulas  |  passos: %d  |  %.3f ms/passo (so fisica)",
                     NX, NY, NZ, N_PARTICLES, sim_steps_done,
                     sim_steps_done ? (sim_time_total*1000.0/sim_steps_done) : 0.0),
                     14, 36, 16, (Color){180,180,190,255});
            DrawText(TextFormat("Forma: %s%s  |  Obstaculo: (%.1f, %.1f)  %s",
                     shape_name(shape_type),
                     shape_type == SHAPE_PRISM ? TextFormat(" (%d lados)", shape_sides) : "",
                     obst_cx, obst_cy, dragging_obstacle ? "ARRASTANDO" : ""),
                     14, 56, 16, (Color){150,200,255,255});
            DrawText("Clique e arraste o obstaculo: move | Arraste fora dele: orbita | TAB: troca a forma | [ ]: lados do prisma | Scroll: zoom | ESPACO: pausa",
                     14, screenH - 26, 16, (Color){150,150,160,255});
            if (paused) DrawText("PAUSADO", screenW - 120, 12, 20, (Color){230,180,60,255});
            DrawFPS(screenW - 90, screenH - 26);
        EndDrawing();
    }

    UnloadTexture(smokeTex);
    CloseWindow();

    free(f); free(fn); free(ux); free(uy); free(uz); free(obstacle); free(prev_obstacle);
    return 0;
}
