#include <iostream>

#ifdef __APPLE__
#include "GLUT/glut.h"
#include <OPENGL/gl.h>
#endif

#ifdef __unix__
#include <GL/glut.h>
#endif

#ifdef _WIN32
#pragma comment(lib, "glew32.lib")

#include <GL\glew.h>
#include <GL\glut.h>
#endif

#include <time.h>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "GLSL_helper.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "GeometryCreator.h"
#include "MStackHelp.h"
#include "Fluid.h"
#include "ParticleUpdate.h"

using namespace std;
using namespace glm;

#define PI 3.14

GLuint triBuffObj;
GLuint colBuffObj;
GLuint normalBuffObj;

float currentTime;
float previousTime;
int frameCount = 0;
int particleCount = 1;
int pauseMode = 0;
int mode = 2;
int shade = 1;
int ShadeProg;
int mouseStartX;
int mouseStartY;
int mouseEndX;
int mouseEndY;
float g_width;
float g_height;
float fps = -1;
char fpsString[9];

ParticleSystem allParticles(-1, NULL);
int material[NUM_PARTICLES];
Mesh *particle;

GLint h_aPosition;
GLint h_aNormal;
GLint h_uModelMatrix;
GLint h_uViewMatrix;
GLint h_uProjMatrix;
GLint h_lightPos;
GLint h_cameraPos;
GLint h_uMatAmb;
GLint h_uMatDif;
GLint h_uMatSpec;
GLint h_uMatShine;
GLint h_uColor;

void SetProjectionMatrix() {
  mat4 Projection = perspective(90.0f, (float)g_width/g_height, 0.1f, 200.f);
  safe_glUniformMatrix4fv(h_uProjMatrix, value_ptr(Projection));
}

void SetView() {
  mat4 Trans = translate(mat4(1.0f), vec3(0.0f, 0, -80));
  mat4 RotateX = rotate(Trans, 0.0f, vec3(0.0f, 1, 0));
  safe_glUniformMatrix4fv(h_uViewMatrix, value_ptr(RotateX));
}

void SetModel(float transX, float transY, float transZ) {
  mat4 trans = translate(mat4(1.0f), vec3(transX, transY, transZ));
  safe_glUniformMatrix4fv(h_uModelMatrix, value_ptr(trans));
}

int InstallShader(const GLchar *vShaderName, const GLchar *fShaderName) {
  GLuint VS; //handles to shader object
  GLuint FS; //handles to frag shader object
  GLint vCompiled, fCompiled, linked; //status of shader

  VS = glCreateShader(GL_VERTEX_SHADER);
  FS = glCreateShader(GL_FRAGMENT_SHADER);

  //load the source
  glShaderSource(VS, 1, &vShaderName, NULL);
  glShaderSource(FS, 1, &fShaderName, NULL);

  //compile shader and print log
  glCompileShader(VS);
  /* check shader status requires helper functions */
  printOpenGLError();
  glGetShaderiv(VS, GL_COMPILE_STATUS, &vCompiled);
  printShaderInfoLog(VS);

  //compile shader and print log
  glCompileShader(FS);
  /* check shader status requires helper functions */
  printOpenGLError();
  glGetShaderiv(FS, GL_COMPILE_STATUS, &fCompiled);
  printShaderInfoLog(FS);

  if (!vCompiled || !fCompiled) {
    printf("Error compiling either shader %s or %s", vShaderName, fShaderName);
    return 0;
  }

  //create a program object and attach the compiled shader
  ShadeProg = glCreateProgram();
  glAttachShader(ShadeProg, VS);
  glAttachShader(ShadeProg, FS);

  glLinkProgram(ShadeProg);
  /* check shader status requires helper functions */
  printOpenGLError();
  glGetProgramiv(ShadeProg, GL_LINK_STATUS, &linked);
  printProgramInfoLog(ShadeProg);

  glUseProgram(ShadeProg);

  /* get handles to attribute data */
  h_aPosition = safe_glGetAttribLocation(ShadeProg, "aPosition");
  h_aNormal = safe_glGetAttribLocation(ShadeProg, "aNormal");
  h_uProjMatrix = safe_glGetUniformLocation(ShadeProg, "uProjMatrix");
  h_uViewMatrix = safe_glGetUniformLocation(ShadeProg, "uViewMatrix");
  h_uModelMatrix = safe_glGetUniformLocation(ShadeProg, "uModelMatrix");
  h_uMatAmb = safe_glGetUniformLocation(ShadeProg, "uMat.aColor");
  h_uMatDif = safe_glGetUniformLocation(ShadeProg, "uMat.dColor");
  h_uMatSpec = safe_glGetUniformLocation(ShadeProg, "uMat.sColor");
  h_uMatShine = safe_glGetUniformLocation(ShadeProg, "uMat.shine");
  h_lightPos = safe_glGetUniformLocation(ShadeProg, "lightPos");
  h_cameraPos = safe_glGetUniformLocation(ShadeProg, "cameraPos");
  return 1;
}

void initializeColors() {
  for(int i = 0; i < NUM_PARTICLES; i++) {
    material[i] = rand() % 3;
  }
}

void InitGeom() {
  // Make patient ZERO particle
  particle = GeometryCreator::CreateSphere(glm::vec3(RADIUS));

  // Fill HouseKeeping Array of Particle positions
  allParticles.initalize();
  initializeColors();

  fpsString[0] = 'F';
  fpsString[1] = 'P';
  fpsString[2] = 'S';
  fpsString[3] = ':';
  fpsString[4] = ' ';
}

void Initialize() {
  glClearColor(0, 0, 0, 1.0f);
  glEnable(GL_DEPTH_TEST);
}

void setMaterial(int type) {
  switch(type) {
    case 0:
      safe_glUniform3f(h_uMatAmb, 0.2, 0.3, 0.6);
      safe_glUniform3f(h_uMatDif, 0.0, 0.08, 0.5);
      safe_glUniform3f(h_uMatSpec, 0.4, 0.4, 0.4);
      safe_glUniform1f(h_uMatShine, 2.0);
      break;
    case 1:
      safe_glUniform3f(h_uMatAmb, 0.6, 0.3, 0.2);
      safe_glUniform3f(h_uMatDif, 0.5, 0.08, 0.0);
      safe_glUniform3f(h_uMatSpec, 0.4, 0.4, 0.4);
      safe_glUniform1f(h_uMatShine, 2.0);
      break;
    case 2:
      safe_glUniform3f(h_uMatAmb, 0.2, 0.6, 0.3);
      safe_glUniform3f(h_uMatDif, 0.0, 0.5, 0.08);
      safe_glUniform3f(h_uMatSpec, 0.4, 0.4, 0.4);
      safe_glUniform1f(h_uMatShine, 2.0);
      break;
  }
}

void Draw() {
  if (!pauseMode) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(ShadeProg);
    SetProjectionMatrix();
    SetView();
    glUniform4f(h_lightPos, 0, 30, 0, 1);
    glUniform4f(h_cameraPos, 0, 0, 80, 1);

    // Draw based on Array of Particle Positions
    for (int index = 0; index < NUM_PARTICLES; index++) {
      setMaterial(material[index]);
      safe_glEnableVertexAttribArray(h_aPosition);
      glBindBuffer(GL_ARRAY_BUFFER, particle->PositionHandle);
      safe_glVertexAttribPointer(h_aPosition, 3, GL_FLOAT, GL_FALSE, 0, 0);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, particle->IndexHandle);
      safe_glEnableVertexAttribArray(h_aNormal);
      glBindBuffer(GL_ARRAY_BUFFER, particle->NormalHandle);
      safe_glVertexAttribPointer(h_aNormal, 3, GL_FLOAT, GL_FALSE, 0, 0);
      SetModel(allParticles.particles[index].position.x,
               allParticles.particles[index].position.y,
               allParticles.particles[index].position.z);
      glDrawElements(GL_TRIANGLES, particle->IndexBufferLength, GL_UNSIGNED_SHORT, 0);
    }

    // Draw the wireframe cube
    SetModel(0, 0, 0);
    setMaterial(0);
    glutWireCube(64);
    allParticles.update(0.01);

    safe_glDisableVertexAttribArray(h_aPosition);

    glUseProgram(0);
    glutSwapBuffers();
  }
}

void ReshapeGL(int width, int height) {
  g_width = (float)width;
  g_height = (float)height;
  glViewport(0, 0, (GLsizei)(width), (GLsizei)(height));
}

void keyboard(unsigned char key, int x, int y ) {
  switch (key) {
    case ' ':
      pauseMode = !pauseMode;
      break;
    case 'q': case 'Q' :
      exit( EXIT_SUCCESS );
      break;
  }
  glutPostRedisplay();
}

void update(int val) {
  glutPostRedisplay();

  // http://mycodelog.com/2010/04/16/fps/
  frameCount++;
  currentTime = glutGet(GLUT_ELAPSED_TIME);

  float timeInterval = currentTime - previousTime;
  if(timeInterval > 1000) {
    fps = frameCount / (timeInterval / 1000);
    previousTime = currentTime;
    frameCount = 0;
  }

  if (fps != -1) {
    fpsString[5] = (int)fps / 100 % 10 + 48;
    fpsString[6] = (int)fps / 10 % 10 + 48;
    fpsString[7] = (int)fps / 1 % 10 + 48;

    glutSetWindowTitle(fpsString);
  }
  glutTimerFunc(10, update, 0);
}

int main(int argc, char *argv[]) {
  srand(time(NULL));
  glutInit(&argc, argv);
  glutInitWindowPosition(200, 200);
  glutInitWindowSize(1000, 1000);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutCreateWindow("Fluid");
  glutReshapeFunc(ReshapeGL);
  glutDisplayFunc(Draw);
  glutKeyboardFunc(keyboard);
  glutTimerFunc(10, update, 0);
  g_width = g_height = 200;
  #ifdef _WIN32
    GLenum err = glewInit();
    if (GLEW_OK != err) {
      cerr << "Error initializing glew! " << glewGetErrorString(err) << endl;
      return 1;
    }
  #endif
  Initialize();
  getGLversion();
  if (!InstallShader(textFileRead((char *)"Fluid_Vert.glsl"),
                     textFileRead((char *)"Fluid_Frag.glsl"))) {
    printf("Error installing shader!\n");
    return 0;
  }
  InitGeom();
  glutMainLoop();
  return 0;
}
