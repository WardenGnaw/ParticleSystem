#ifndef PARTICLE_UPDATE_H
#define PARTICLE_UPDATE_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Fluid.h"

void updateParticles(Particle *particles, int size, my_vec3 localExtForce);
void updateGrid(Particle *particles, int *gridCounter, int *gridCells);
__global__ void updateParticleKernel(Particle *particles, my_vec3 *extForce,
                                     int *gridCounter, int *gridCells);
__global__ void updatePartVelocity(Particle *particles, int *gridCounter, int *gridCells);
__device__ void updateParticleInCells(Particle *particles, int *gridCounter,
                                      int *gridCells, int idx);
__device__ float circleDistance(my_vec3 *firstSphere, my_vec3 *secondSphere);
__device__ my_vec3 checkCells(int x, int y, int z, int idx,
                              Particle *particles,
                              int *gridCounter,
                              int *gridCells);

#endif
