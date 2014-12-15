#include <stdio.h>
#include "Fluid.h"
#include "ParticleUpdate.h"
#include "SerialUpdate.h"
#define SIDES 32

// MEMBERS
//      Particle particles[MAX_PARTICLES]; /* Keeps track of all the particles */
//      unsigned int numParticles;         /* Current number of particles alive */
//      my_vec3 gravity;                      /* Particle System affects particles */
//      my_vec3 wind;
//      float milestone;                    /* How long a particle can live */
//      char* TextureFile;                 /* Name of the texture file */

//Particle System Functions/
ParticleSystem::ParticleSystem(float milestone, char* Texture) :
   milestone(milestone),
   TextureFile(Texture)
{}

void ParticleSystem::initalize() {
  srand(time(NULL));

  float particleYPos = SIDES - 5;
  float particleXPos = -SIDES;
  float particleZPos = SIDES;
  int particleCount = 1;

  for (int i = 0; i < NUM_PARTICLES; i++) {
    if (particleCount % (SIDES + 1) == 0) {
      particleXPos = -SIDES;
      if (particleZPos != -SIDES) {
        #ifdef ENABLE_CUDA
        particleZPos--;
        #else
        particleZPos -=2;
        #endif
      }
      else {
        particleZPos = SIDES;
        #ifdef ENABLE_CUDA
        particleYPos-=2;
        #else
        particleYPos-=5;
        #endif
      }
    }
    else {
      particleXPos++;
    }
    particles[i].position.x = particleXPos;
    particles[i].position.y = particleYPos;
    particles[i].position.z = particleZPos;
    #ifdef ENABLE_CUDA
    particles[i].velocity.x = 0;
    particles[i].velocity.y = 0;
    particles[i].velocity.z = 0;
    #else
    particles[i].velocity.x = rand()%2 - rand()/(float)RAND_MAX;
    particles[i].velocity.y = -rand()/(float)RAND_MAX;
    particles[i].velocity.z = rand()%2 - rand()/(float)RAND_MAX;
    #endif
    particles[i].newVelocity.x = 0;
    particles[i].newVelocity.y = 0;
    particles[i].newVelocity.z = 0;
    particles[i].radius = RADIUS;
    particleCount++;
  }
}

void ParticleSystem::update(float deltaTime) {
    my_vec3 gravity;
    gravity.x = 0.0f;
    gravity.y = -0.3f;
    gravity.z = 0.0f;

    my_vec3 wind;
    wind.x = 5.0f;
    wind.y = 0.0f;
    wind.z = 0.0f;

  #ifdef ENABLE_CUDA
    updateParticles(particles, NUM_PARTICLES, gravity);
  #else
    gravity.y = -2.3f;
    updateParticlesS(particles, NUM_PARTICLES, gravity);
  #endif
}
