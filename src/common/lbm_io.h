#ifndef LBM_IO_H
#define LBM_IO_H

#include <stdio.h>
#include <math.h>

/* Cronômetro de alta resolução, portável entre Windows e Linux/WSL. */
#if defined(_WIN32)
#include <windows.h>
static double now_seconds(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    LARGE_INTEGER t;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

/* Salva um corte (slice) no plano XY no meio do eixo Z como imagem PGM
 * (escala de cinza), representando a magnitude da velocidade local.
 * Pode ser aberta em qualquer visualizador de imagens ou convertida com
 * ferramentas como ImageMagick/GIMP para inspecionar o escoamento. */
static void write_velocity_slice_pgm(const char *filename, const double *ux, const double *uy,
                                      const double *uz, int nx, int ny, int nz, double u_scale) {
    int zc = nz / 2;
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "lbm_io: nao foi possivel abrir %s para escrita\n", filename);
        return;
    }
    fprintf(fp, "P5\n%d %d\n255\n", nx, ny);
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            size_t idx = (size_t)x + (size_t)nx * ((size_t)y + (size_t)ny * (size_t)zc);
            double mag = sqrt(ux[idx]*ux[idx] + uy[idx]*uy[idx] + uz[idx]*uz[idx]);
            double norm = mag / u_scale;
            if (norm > 1.0) norm = 1.0;
            if (norm < 0.0) norm = 0.0;
            unsigned char v = (unsigned char)(norm * 255.0);
            fputc(v, fp);
        }
    }
    fclose(fp);
}

#endif /* LBM_IO_H */
