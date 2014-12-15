#include "Fluid.h"

#define BOTTOM_BOUND -25.0f
#define TOP_BOUND 25.0f
#define FRONT_BOUND 25.0f
#define BACK_BOUND -25.0f
#define LEFT_BOUND -25.0f
#define RIGHT_BOUND 25.0f

#define NUM_CELLS 132651
#define NUM_CELLS_X4 530604

void updateParticlesS(Particle *particles, int size, my_vec3 extForce);
void updateGridS(Particle *particles, int *gridCounter, int *gridCells);
void updateParticlesInCellsS(Particle *particles, int *gridCounter, int *gridCells);
void updateParticleKernelS(Particle *particles, my_vec3 extForce);
