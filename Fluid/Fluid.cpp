#include <iostream>
#include <fstream>
#include <cmath>
#include <CL/opencl.h>

#include <GL/freeglut.h>

#include "Common.h"

// *****************
// OpenCL processing
// *****************

// OpenCL infrastructure
cl_platform_id platformID;
cl_device_id deviceID;
cl_context context;
cl_command_queue queue;

cl_program simulationProgram;

// simulation
int gridResolution = 512;
int inputVelocityBuffer = 0;
cl_mem velocityBuffer[2];

int inputDensityBuffer = 0;
cl_mem densityBuffer[2];
cl_float4 densityColor;

int inputPressureBuffer = 0;
cl_mem pressureBuffer[2];
cl_mem divergenceBuffer;

cl_mem vorticityBuffer;

cl_kernel advectionKernel;
cl_kernel advectionDensityKernel;
cl_kernel diffusionKernel;
cl_kernel divergenceKernel;
cl_kernel pressureJacobiKernel;
cl_kernel projectionKernel;
cl_kernel vorticityKernel;
cl_kernel addVorticityForceKernel;
cl_kernel addForceKernel;
cl_kernel resetSimulationKernel;

size_t problemSize[2];

cl_float2 force;

// visualization
int width = 512;
int height = 512;

cl_mem visualizationBufferGPU;
cl_float4* visualizationBufferCPU;

int visualizationMethod = 1;

size_t visualizationSize[2];
cl_kernel visualizationDensityKernel;
cl_kernel visualizationVelocityKernel;
cl_kernel visualizationPressureKernel;

void initSimulation(){
	clGetPlatformIDs(1, &platformID, NULL);
	clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, NULL);
	cl_context_properties contextProperties[] =
	{ CL_CONTEXT_PLATFORM, (cl_context_properties)platformID, 0 };
	context = clCreateContext(contextProperties, 1, &deviceID, NULL, NULL, NULL);
	queue = clCreateCommandQueue(context, deviceID, CL_QUEUE_PROFILING_ENABLE, NULL);

	std::string source = FileToString("../kernels/programs.cl");
	const char* csource = source.c_str();
	simulationProgram = clCreateProgramWithSource(context, 1, &csource, NULL, NULL);
	cl_int err = clBuildProgram(simulationProgram, 1, &deviceID, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		cl_uint logLength;
		clGetProgramBuildInfo(simulationProgram, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &logLength);
		char* log = new char[logLength];
		clGetProgramBuildInfo(simulationProgram, deviceID, CL_PROGRAM_BUILD_LOG, logLength, log, 0);
		std::cout << log << std::endl;
		delete[] log;
		exit(-1);
	}

	// simulation
	problemSize[0] = gridResolution;
	problemSize[1] = gridResolution;


	advectionKernel = clCreateKernel(simulationProgram, "advection", &err);
	velocityBuffer[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float2) * gridResolution * gridResolution, NULL, NULL);
	velocityBuffer[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float2) * gridResolution * gridResolution, NULL, NULL);

	advectionDensityKernel = clCreateKernel(simulationProgram, "advectionDensity", &err);
	densityBuffer[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4) * gridResolution * gridResolution, NULL, NULL);
	densityBuffer[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4) * gridResolution * gridResolution, NULL, NULL);

	diffusionKernel = clCreateKernel(simulationProgram, "diffusion", &err);

	divergenceKernel = clCreateKernel(simulationProgram, "divergence", &err);
	pressureJacobiKernel = clCreateKernel(simulationProgram, "pressureJacobi", &err);
	pressureBuffer[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * gridResolution * gridResolution, NULL, NULL);
	pressureBuffer[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * gridResolution * gridResolution, NULL, NULL);
	divergenceBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * gridResolution * gridResolution, NULL, NULL);
	projectionKernel = clCreateKernel(simulationProgram, "projection", &err);

	vorticityKernel = clCreateKernel(simulationProgram, "vorticity", &err);
	addVorticityForceKernel = clCreateKernel(simulationProgram, "addVorticity", &err);
	vorticityBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * gridResolution * gridResolution, NULL, NULL);

	densityColor.s[0] = densityColor.s[1] = densityColor.s[2] = densityColor.s[3] = 1.0f;
	addForceKernel = clCreateKernel(simulationProgram, "addForce", &err);

	resetSimulationKernel = clCreateKernel(simulationProgram, "resetSimulation", &err);

	// visualization
	visualizationSize[0] = width;
	visualizationSize[1] = height;

	visualizationBufferCPU = new cl_float4[width * height];
	visualizationBufferGPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4) * width * height, NULL, NULL);
	visualizationDensityKernel = clCreateKernel(simulationProgram, "visualizationDensity", &err);
	visualizationVelocityKernel = clCreateKernel(simulationProgram, "visualizationVelocity", &err);
	visualizationPressureKernel = clCreateKernel(simulationProgram, "visualizationPressure", &err);
}

void resetSimulation() {
	clSetKernelArg(resetSimulationKernel, 0, sizeof(int), &gridResolution);
	clSetKernelArg(resetSimulationKernel, 1, sizeof(cl_mem), &velocityBuffer[inputVelocityBuffer]);
	clSetKernelArg(resetSimulationKernel, 2, sizeof(cl_mem), &pressureBuffer[inputPressureBuffer]);
	clSetKernelArg(resetSimulationKernel, 3, sizeof(cl_mem), &densityBuffer[inputDensityBuffer]);

		clEnqueueNDRangeKernel(queue, resetSimulationKernel,
			2, NULL, problemSize, NULL,
			0, NULL, NULL);
	clFinish(queue);
}

void resetPressure() {
	clSetKernelArg(resetSimulationKernel, 0, sizeof(int), &gridResolution);
	clSetKernelArg(resetSimulationKernel, 1, sizeof(cl_mem), &velocityBuffer[(inputVelocityBuffer + 1) % 2]);
	clSetKernelArg(resetSimulationKernel, 2, sizeof(cl_mem), &pressureBuffer[inputPressureBuffer]);
	clSetKernelArg(resetSimulationKernel, 3, sizeof(cl_mem), &densityBuffer[(inputDensityBuffer + 1) % 2]);

	clEnqueueNDRangeKernel(queue, resetSimulationKernel,
		2, NULL, problemSize, NULL,
		0, NULL, NULL);
	clFinish(queue);
}

void simulateAdvection() {
	clSetKernelArg(advectionKernel, 0, sizeof(int), &gridResolution);
	clSetKernelArg(advectionKernel, 1, sizeof(cl_mem), &velocityBuffer[inputVelocityBuffer]);
	clSetKernelArg(advectionKernel, 2, sizeof(cl_mem), &velocityBuffer[(inputVelocityBuffer + 1) % 2]);

	clEnqueueNDRangeKernel(queue, advectionKernel,
		2, NULL, problemSize, NULL,
		0, NULL, NULL);
	clFinish(queue);

	inputVelocityBuffer = (inputVelocityBuffer + 1) % 2;
}

void simulateVorticity() {
	clSetKernelArg(vorticityKernel, 0, sizeof(int), &gridResolution);
	clSetKernelArg(vorticityKernel, 1, sizeof(cl_mem), &velocityBuffer[inputVelocityBuffer]);
	clSetKernelArg(vorticityKernel, 2, sizeof(cl_mem), &vorticityBuffer);

	clEnqueueNDRangeKernel(queue, vorticityKernel,
		2, NULL, problemSize, NULL,
		0, NULL, NULL);
	clFinish(queue);

	clSetKernelArg(addVorticityForceKernel, 0, sizeof(int), &gridResolution);
	clSetKernelArg(addVorticityForceKernel, 1, sizeof(cl_mem), &vorticityBuffer);
	clSetKernelArg(addVorticityForceKernel, 2, sizeof(cl_mem), &velocityBuffer[inputVelocityBuffer]);

	clEnqueueNDRangeKernel(queue, addVorticityForceKernel,
		2, NULL, problemSize, NULL,
		0, NULL, NULL);
	clFinish(queue);
}

void simulateDiffusion() {
	for (int i = 0; i < 10; ++i) {
		clSetKernelArg(diffusionKernel, 0, sizeof(int), &gridResolution);
		clSetKernelArg(diffusionKernel, 1, sizeof(cl_mem), &velocityBuffer[inputVelocityBuffer]);
		clSetKernelArg(diffusionKernel, 2, sizeof(cl_mem), &velocityBuffer[(inputVelocityBuffer + 1) % 2]);

		clEnqueueNDRangeKernel(queue, diffusionKernel,
			2, NULL, problemSize, NULL,
			0, NULL, NULL);
		clFinish(queue);

		inputVelocityBuffer = (inputVelocityBuffer + 1) % 2;
	}
}

void projection() {
	clSetKernelArg(divergenceKernel, 0, sizeof(int), &gridResolution);
	clSetKernelArg(divergenceKernel, 1, sizeof(cl_mem), &velocityBuffer[inputVelocityBuffer]);
	clSetKernelArg(divergenceKernel, 2, sizeof(cl_mem), &divergenceBuffer);

	clEnqueueNDRangeKernel(queue, divergenceKernel,
		2, NULL, problemSize, NULL,
		0, NULL, NULL);
	clFinish(queue);

	resetPressure();

	for (int i = 0; i < 10; ++i) {
		clSetKernelArg(pressureJacobiKernel, 0, sizeof(int), &gridResolution);
		clSetKernelArg(pressureJacobiKernel, 1, sizeof(cl_mem), &pressureBuffer[inputPressureBuffer]);
		clSetKernelArg(pressureJacobiKernel, 2, sizeof(cl_mem), &pressureBuffer[(inputPressureBuffer + 1) % 2]);
		clSetKernelArg(pressureJacobiKernel, 3, sizeof(cl_mem), &divergenceBuffer);

		clEnqueueNDRangeKernel(queue, pressureJacobiKernel,
			2, NULL, problemSize, NULL,
			0, NULL, NULL);
		clFinish(queue);

		inputPressureBuffer = (inputPressureBuffer + 1) % 2;
	}

	clSetKernelArg(projectionKernel, 0, sizeof(int), &gridResolution);
	clSetKernelArg(projectionKernel, 1, sizeof(cl_mem), &velocityBuffer[inputVelocityBuffer]);
	clSetKernelArg(projectionKernel, 2, sizeof(cl_mem), &pressureBuffer[inputPressureBuffer]);
	clSetKernelArg(projectionKernel, 3, sizeof(cl_mem), &velocityBuffer[(inputVelocityBuffer + 1) % 2]);

	clEnqueueNDRangeKernel(queue, projectionKernel,
		2, NULL, problemSize, NULL,
		0, NULL, NULL);
	clFinish(queue);

	inputVelocityBuffer = (inputVelocityBuffer + 1) % 2;
}

void simulateDensityAdvection() {
	clSetKernelArg(advectionDensityKernel, 0, sizeof(int), &gridResolution);
	clSetKernelArg(advectionDensityKernel, 1, sizeof(cl_mem), &velocityBuffer[inputVelocityBuffer]);
	clSetKernelArg(advectionDensityKernel, 2, sizeof(cl_mem), &densityBuffer[inputDensityBuffer]);
	clSetKernelArg(advectionDensityKernel, 3, sizeof(cl_mem), &densityBuffer[(inputDensityBuffer + 1) % 2]);

	clEnqueueNDRangeKernel(queue, advectionDensityKernel,
		2, NULL, problemSize, NULL,
		0, NULL, NULL);
	clFinish(queue);

	inputDensityBuffer = (inputDensityBuffer + 1) % 2;
}

void addForce(int x, int y, cl_float2 force) {
	float fx = (float)x / width;
	float fy = (float)y / height;

	clSetKernelArg(addForceKernel, 0, sizeof(float), &fx);
	clSetKernelArg(addForceKernel, 1, sizeof(float), &fy);
	clSetKernelArg(addForceKernel, 2, sizeof(cl_float2), &force);
	clSetKernelArg(addForceKernel, 3, sizeof(int), &gridResolution);
	clSetKernelArg(addForceKernel, 4, sizeof(cl_mem), &velocityBuffer[inputVelocityBuffer]);
	clSetKernelArg(addForceKernel, 5, sizeof(cl_float4), &densityColor);
	clSetKernelArg(addForceKernel, 6, sizeof(cl_mem), &densityBuffer[inputDensityBuffer]);

	clEnqueueNDRangeKernel(queue, addForceKernel,
		2, NULL, problemSize, NULL,
		0, NULL, NULL);
	clFinish(queue);
}

void simulationStep() {
	simulateAdvection();
	simulateDiffusion();
	simulateVorticity();
	projection();
	simulateDensityAdvection();
}

void visualizationStep() {
	switch (visualizationMethod) {
	case 0:
		clSetKernelArg(visualizationDensityKernel, 0, sizeof(int), &width);
		clSetKernelArg(visualizationDensityKernel, 1, sizeof(int), &height);
		clSetKernelArg(visualizationDensityKernel, 2, sizeof(cl_mem), &visualizationBufferGPU);
		clSetKernelArg(visualizationDensityKernel, 3, sizeof(int), &gridResolution);
		clSetKernelArg(visualizationDensityKernel, 4, sizeof(cl_mem), &densityBuffer[inputDensityBuffer]);
		clEnqueueNDRangeKernel(queue, visualizationDensityKernel,
			2, NULL, visualizationSize, NULL,
			0, NULL, NULL);
		break;

	case 1:
		clSetKernelArg(visualizationVelocityKernel, 0, sizeof(int), &width);
		clSetKernelArg(visualizationVelocityKernel, 1, sizeof(int), &height);
		clSetKernelArg(visualizationVelocityKernel, 2, sizeof(cl_mem), &visualizationBufferGPU);
		clSetKernelArg(visualizationVelocityKernel, 3, sizeof(int), &gridResolution);
		clSetKernelArg(visualizationVelocityKernel, 4, sizeof(cl_mem), &velocityBuffer[inputVelocityBuffer]);
		clEnqueueNDRangeKernel(queue, visualizationVelocityKernel,
			2, NULL, visualizationSize, NULL,
			0, NULL, NULL);
		break;

	case 2:
		clSetKernelArg(visualizationPressureKernel, 0, sizeof(int), &width);
		clSetKernelArg(visualizationPressureKernel, 1, sizeof(int), &height);
		clSetKernelArg(visualizationPressureKernel, 2, sizeof(cl_mem), &visualizationBufferGPU);
		clSetKernelArg(visualizationPressureKernel, 3, sizeof(int), &gridResolution);
		clSetKernelArg(visualizationPressureKernel, 4, sizeof(cl_mem), &pressureBuffer[inputPressureBuffer]);

		clEnqueueNDRangeKernel(queue, visualizationPressureKernel,
			2, NULL, visualizationSize, NULL,
			0, NULL, NULL);
		break;

	}
	clFinish(queue);

	clEnqueueReadBuffer(queue, visualizationBufferGPU, CL_TRUE,
		0, sizeof(cl_float4) * width * height, visualizationBufferCPU,
		0, NULL, NULL);
	glDrawPixels(width, height, GL_RGBA, GL_FLOAT, visualizationBufferCPU);
}

// OpenGL
int method = 1;
bool keysPressed[256];

void initOpenGL() {
	glClearColor(0.17f, 0.4f, 0.6f, 1.0f);
}

void display() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	simulationStep();
	visualizationStep();

	glEnable(GL_DEPTH_TEST);
	glutSwapBuffers();
}

void idle() {
	glutPostRedisplay();
}

void keyDown(unsigned char key, int x, int y) {
	keysPressed[key] = true;
}

void keyUp(unsigned char key, int x, int y) {
	keysPressed[key] = false;
	switch (key) {
	case 'r':
		resetSimulation();
		break;

	case 'd':
		visualizationMethod = 0;
		break;
	case 'v':
		visualizationMethod = 1;
		break;
	case 'p':
		visualizationMethod = 2;
		break;

	case '1':
		densityColor.s[0] = densityColor.s[1] = densityColor.s[2] = densityColor.s[3] = 1.0f;
		break;

	case '2':
		densityColor.s[0] = 1.0f;
		densityColor.s[1] = densityColor.s[2] = densityColor.s[3] = 0.0f;
		break;

	case '3':
		densityColor.s[1] = 1.0f;
		densityColor.s[0] = densityColor.s[2] = densityColor.s[3] = 0.0f;
		break;

	case '4':
		densityColor.s[2] = 1.0f;
		densityColor.s[0] = densityColor.s[1] = densityColor.s[3] = 0.0f;
		break;

	case 27:
		exit(0);
		break;
	}
}

int mX, mY;

void mouseClick(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON)
		if (state == GLUT_DOWN) {
			mX = x;
			mY = y;
		}
}

void mouseMove(int x, int y) {
	force.s[0] = (float)(x - mX);
	force.s[1] = -(float)(y - mY);
	//addForce(mX, height - mY, force);
	addForce(256, 256, force);
	mX = x;
	mY = y;
}

void reshape(int newWidth, int newHeight) {
	width = newWidth;
	height = newHeight;
	glViewport(0, 0, width, height);
}

int main(int argc, char* argv[]) {
	glutInit(&argc, argv);
	glutInitContextVersion(3, 0);
	glutInitContextFlags(GLUT_CORE_PROFILE | GLUT_DEBUG);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("GPGPU 13. labor: Incompressible fluid simulation");

	initOpenGL();

	glutDisplayFunc(display);
	glutIdleFunc(idle);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyDown);
	glutKeyboardUpFunc(keyUp);
	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseMove);

	// OpenCL processing
	initSimulation();

	glutMainLoop();
	return(0);
}
