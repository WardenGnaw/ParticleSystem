#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <map>
#include <cmath>
#include "ParticleUpdate.h"
#include "Fluid.h"

#define BOTTOM_BOUND -32
#define TOP_BOUND 32
#define FRONT_BOUND 32
#define BACK_BOUND -32
#define LEFT_BOUND -32
#define RIGHT_BOUND 32

#define NUM_CELLS 300763
#define NUM_CELLS_X4 1203052
#define CELL_DIM 67

#define DELTA_TIME 0.5
#define SPRING_FORCE 0.5
#define WALL_DAMP_FORCE 0.5
#define PART_DAMP_FORCE 0.01
#define SHEAR_FORCE 0.02

using namespace std;

void updateParticles(Particle *particles, int size, my_vec3 localExtForce) {
  int *gridCounter = (int *)calloc(sizeof(int), NUM_CELLS);
  int *gridCells = (int *)calloc(sizeof(int), NUM_CELLS_X4);

  updateGrid(particles, gridCounter, gridCells);

  int *d_gridCounter;
  if (cudaMalloc(&d_gridCounter, sizeof(int) * NUM_CELLS) != cudaSuccess) {
    printf("didn't malloc space for grid counters\n");
  }
  if (cudaMemcpy(d_gridCounter, gridCounter,
                 sizeof(int) * NUM_CELLS,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("didn't copy gridCounter\n");
  }

  int *d_gridCells;
  if (cudaMalloc(&d_gridCells, sizeof(int) * NUM_CELLS_X4) != cudaSuccess) {
    printf("didn't malloc space for grid cells\n");
  }
  if (cudaMemcpy(d_gridCells, gridCells,
                 sizeof(int) * NUM_CELLS_X4,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("didn't copy grid cells\n");
  }

  Particle *d_particles;
  if (cudaMalloc(&d_particles, sizeof(Particle) * size) != cudaSuccess) {
    printf("didn't malloc space for device particles\n");
  }
  if (cudaMemcpy(d_particles, particles,
                 sizeof(Particle) * size,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("didn't copy particles\n");
  }

  my_vec3 *d_localExtForce;
  if (cudaMalloc(&d_localExtForce, sizeof(my_vec3)) != cudaSuccess) {
    printf("didn't malloc space for force\n");
  }
  if (cudaMemcpy(d_localExtForce, &localExtForce,
                 sizeof(my_vec3),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("didn't copy force\n");
  }

  dim3 dimBlock(512);
  dim3 dimGrid(32);

  // Updating the particles with gravity
  updateParticleKernel<<<dimGrid, dimBlock>>>(d_particles, d_localExtForce,
                                              d_gridCounter, d_gridCells);

  if (cudaMemcpy(particles, d_particles,
                 sizeof(Particle) * size,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    printf("didn't memcpy back\n");
  }

  free(gridCounter);
  free(gridCells);
  cudaFree(d_gridCounter);
  cudaFree(d_gridCells);
  cudaFree(d_particles);
  cudaFree(d_localExtForce);
}

void updateGrid(Particle *particles, int *gridCounter, int *gridCells) {
  unsigned int idx;
  for (int x = 0; x < NUM_PARTICLES; x++) {

    particles[x].cell.x = floor(particles[x].position.x + 33);
    particles[x].cell.y = floor(particles[x].position.y + 33);
    particles[x].cell.z = floor(particles[x].position.z + 33);
    idx = particles[x].cell.z * CELL_DIM * CELL_DIM +
          particles[x].cell.y * CELL_DIM +
          particles[x].cell.x;
    if (gridCounter[idx] < 4) {
      gridCells[idx * 4 + gridCounter[idx]] = x;
      gridCounter[idx]++;
    }
  }
}

__global__ void updateParticleKernel(Particle *particles, my_vec3 *extForce,
                                     int *gridCounter, int *gridCells) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;



  particles[idx].velocity.x = particles[idx].newVelocity.x;
  particles[idx].velocity.y = particles[idx].newVelocity.y;
  particles[idx].velocity.z = particles[idx].newVelocity.z;

  updateParticleInCells(particles, gridCounter,
                        gridCells, idx);

  particles[idx].newVelocity.x += extForce->x * DELTA_TIME;
  particles[idx].newVelocity.y += extForce->y * DELTA_TIME;
  particles[idx].newVelocity.z += extForce->z * DELTA_TIME;

  particles[idx].position.x += particles[idx].newVelocity.x * DELTA_TIME;
  particles[idx].position.y += particles[idx].newVelocity.y * DELTA_TIME;
  particles[idx].position.z += particles[idx].newVelocity.z * DELTA_TIME;

  if (particles[idx].position.y != particles[idx].position.y) {
    particles[idx].position.y = 0;
    particles[idx].velocity.y = 0;
    printf("So sad\n");
  }
  if (particles[idx].position.x != particles[idx].position.x) {
    particles[idx].position.x = 0;
    particles[idx].velocity.x = 0;
    printf("So soooooo sad\n");
  }
  if (particles[idx].position.z != particles[idx].position.z) {
    particles[idx].position.z = 0;
    particles[idx].velocity.z = 0;
    printf("Much sad\n");
  }

  if (particles[idx].position.y <= BOTTOM_BOUND + RADIUS) {
    particles[idx].position.y = BOTTOM_BOUND + RADIUS;
    particles[idx].newVelocity.y *= -WALL_DAMP_FORCE;
  }
  else if (particles[idx].position.y > TOP_BOUND - RADIUS) {
    particles[idx].position.y = TOP_BOUND - RADIUS;
    particles[idx].newVelocity.y *= -WALL_DAMP_FORCE;

  }
  if (particles[idx].position.x < LEFT_BOUND + RADIUS) {
    particles[idx].position.x = LEFT_BOUND + RADIUS;
    particles[idx].newVelocity.x *= -WALL_DAMP_FORCE;
  }
  else if (particles[idx].position.x > RIGHT_BOUND - RADIUS) {
    particles[idx].position.x = RIGHT_BOUND - RADIUS;
    particles[idx].newVelocity.x *= -WALL_DAMP_FORCE;
  }
  if (particles[idx].position.z < BACK_BOUND + RADIUS) {
    particles[idx].position.z = BACK_BOUND + RADIUS;
    particles[idx].newVelocity.z *= -WALL_DAMP_FORCE;
  }
  else if (particles[idx].position.z > FRONT_BOUND - RADIUS) {
    particles[idx].position.z = FRONT_BOUND - RADIUS;
    particles[idx].newVelocity.z *= -WALL_DAMP_FORCE;
  }
}

__device__ void updateParticleInCells(Particle *particles, int *gridCounter,
                                      int *gridCells, int idx) {
  my_vec3 force;
  my_vec3 tmpForce;
  force.x = 0;
  force.y = 0;
  force.z = 0;

  for (int z = -1; z < 2; z++) {
    for (int y = -1; y < 2; y++) {
      for (int x = -1; x < 2; x++) {
        tmpForce = checkCells(particles[idx].cell.x + x,
                              particles[idx].cell.y + y,
                              particles[idx].cell.z + z,
                              idx,
                              particles,
                              gridCounter,
                              gridCells);
        force.x += tmpForce.x;
        force.y += tmpForce.y;
        force.z += tmpForce.z;
      }
    }
  }
  particles[idx].newVelocity.x = (particles[idx].velocity.x + force.x);
  particles[idx].newVelocity.y = (particles[idx].velocity.y + force.y);
  particles[idx].newVelocity.z = (particles[idx].velocity.z + force.z);
}

__device__ my_vec3 checkCells(int x, int y, int z, int idx,
                           Particle *particles,
                           int *gridCounter,
                           int *gridCells) {
  int cellId = z * CELL_DIM * CELL_DIM + y * CELL_DIM + x;
  int max = gridCounter[cellId];
  int comparePartIdx;
  float checkRadius;
  float firstDot;
  float dist;
  my_vec3 v_ab;
  my_vec3 col_normal;
  my_vec3 force;

  force.x = 0;
  force.y = 0;
  force.z = 0;

  cellId *= 4;
  for (int i = 0; i < max; i++) {
    comparePartIdx = gridCells[cellId + i];
    if (comparePartIdx != idx) {
      checkRadius = particles[idx].radius + particles[comparePartIdx].radius;
      dist = circleDistance(&particles[idx].position,
                            &particles[comparePartIdx].position);
      dist = sqrt(dist);
      if (checkRadius > dist) {
        v_ab.x = particles[comparePartIdx].velocity.x - particles[idx].velocity.x;
        v_ab.y = particles[comparePartIdx].velocity.y - particles[idx].velocity.y;
        v_ab.z = particles[comparePartIdx].velocity.z - particles[idx].velocity.z;

        col_normal.x = particles[idx].position.x - particles[comparePartIdx].position.x;
        col_normal.y = particles[idx].position.y - particles[comparePartIdx].position.y;
        col_normal.z = particles[idx].position.z - particles[comparePartIdx].position.z;

        if (dist != 0) {
          col_normal.x /= dist;
          col_normal.y /= dist;
          col_normal.z /= dist;
          firstDot = v_ab.x * col_normal.x +
                     v_ab.y * col_normal.y +
                     v_ab.z * col_normal.z;

          force.x = SPRING_FORCE * (checkRadius - dist) * col_normal.x;
          force.y = SPRING_FORCE * (checkRadius - dist) * col_normal.y;
          force.z = SPRING_FORCE * (checkRadius - dist) * col_normal.z;

          force.x += PART_DAMP_FORCE * v_ab.x;
          force.y += PART_DAMP_FORCE * v_ab.y;
          force.z += PART_DAMP_FORCE * v_ab.z;

          force.x += SHEAR_FORCE * (v_ab.x - firstDot * col_normal.x);
          force.y += SHEAR_FORCE * (v_ab.y - firstDot * col_normal.y);
          force.z += SHEAR_FORCE * (v_ab.z - firstDot * col_normal.z);
        }
      }
    }
  }

  return force;
}

__device__ float circleDistance(my_vec3 *firstSphere, my_vec3 *secondSphere) {
  float xDist = firstSphere->x - secondSphere->x;
  float yDist = firstSphere->y - secondSphere->y;
  float zDist = firstSphere->z - secondSphere->z;
  return xDist * xDist + yDist * yDist + zDist * zDist;
}