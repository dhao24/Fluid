/* -*- mode: c++ -*- */

__constant float dt = 0.1f;
__constant float viscosity = 2.2f;

__kernel
void resetSimulation(const int gridResolution,
		     __global float2* velocityBuffer,
		     __global float* pressureBuffer,
		     __global float4* densityBuffer){
  int2 id = (int2)(get_global_id(0), get_global_id(1));

  // if (id.x == 1 && id.y == 1) {
  // printf("Here! ");
  // }

  if( id.x < gridResolution && id.y < gridResolution){
    velocityBuffer[id.x + id.y * gridResolution] = (float2)(0.0f);
    pressureBuffer[id.x + id.y * gridResolution] = 0.0f;
    densityBuffer[id.x + id.y * gridResolution] = (float4)(0.0f);
  }
}

// bilinear interpolation
float2 getBil(float2 p, int gridResolution, __global float2* buffer){
  p = clamp(p, (float2)(0.0f), (float2)(gridResolution));

  float2 p00 = buffer[(int)(p.x) + (int)(p.y) * gridResolution];
  float2 p10 = buffer[(int)(p.x) + 1 + (int)(p.y) * gridResolution];
  float2 p11 = buffer[(int)(p.x) + 1 + (int)(p.y + 1.0f) * gridResolution];
  float2 p01 = buffer[(int)(p.x) + (int)(p.y + 1.0f) * gridResolution];

  __private float flr;
  float t0 = fract(p.x, &flr);
  float t1 = fract(p.y, &flr);

  float2 v0 = mix(p00, p10, t0);
  float2 v1 = mix(p01, p11, t0);

  return mix(v0, v1, t1);
}

float4 getBil4(float2 p, int gridResolution, __global float4* buffer){
  p = clamp(p, (float2)(0.0f), (float2)(gridResolution));

  float4 p00 = buffer[(int)(p.x) + (int)(p.y) * gridResolution];
  float4 p10 = buffer[(int)(p.x) + 1 + (int)(p.y) * gridResolution];
  float4 p11 = buffer[(int)(p.x) + 1 + (int)(p.y + 1.0f) * gridResolution];
  float4 p01 = buffer[(int)(p.x) + (int)(p.y + 1.0f) * gridResolution];

  __private float flr;
  float t0 = fract(p.x, &flr);
  float t1 = fract(p.y, &flr);

  float4 v0 = mix(p00, p10, t0);
  float4 v1 = mix(p01, p11, t0);

  return mix(v0, v1, t1);
}

// TODO
//
// Inner parts:
//  velocity := inputVelocityBuffer[x,y]
//  p := id.xy - velocity.xy * dt
//  outputVelocity[x,y] = getBil(p, gridResolution, inputVelocityBuffer)
//
// Border:
//  outputVelocity = -inputVelocity (nearest neighbour)
__kernel
void advection(const int gridResolution,
	       __global float2* inputVelocityBuffer,
	       __global float2* outputVelocityBuffer){
  int2 id = (int2)(get_global_id(0), get_global_id(1));

  // TODO
  if (id.x > 0 && id.x < gridResolution-1 && id.y>0 && id.y < gridResolution-1)
  {
      float2 velocity= inputVelocityBuffer[id.x + id.y * gridResolution];
      float2 p= (float2)(id.x - velocity.x*dt, id.y - velocity.y*dt);

      outputVelocityBuffer[id.x + id.y * gridResolution]=getBil(p, gridResolution, inputVelocityBuffer);
  }else if(id.x==0 || id.y ==0 || id.x >= gridResolution-1 || id.y >= gridResolution-1)
  {
    if (id.x==0){outputVelocityBuffer[id.x + id.y * gridResolution]=-inputVelocityBuffer[id.x + 1 + id.y * gridResolution];}
    if (id.x >= gridResolution-1){outputVelocityBuffer[gridResolution-1 + id.y * gridResolution]=-inputVelocityBuffer[gridResolution - 2 + id.y * gridResolution];}
    if (id.y==0){outputVelocityBuffer[id.x + id.y * gridResolution]=-inputVelocityBuffer[id.x + (id.y+1) * gridResolution];}
    if (id.y >= gridResolution-1){outputVelocityBuffer[id.x + (gridResolution-1) * gridResolution]=-inputVelocityBuffer[id.x + (gridResolution-2) * gridResolution];}

  }
}

// TODO
//
// Inner parts:
//  velocity := velocityBuffer[x,y]
//  p := id.xy - velocity.xy * dt
//  outputDensityBuffer[x,y] := getBil4(p, gridResolution, inputDensityBuffer)
__kernel
void advectionDensity(const int gridResolution,
		      __global float2* velocityBuffer,
		      __global float4* inputDensityBuffer,
		      __global float4* outputDensityBuffer){
  int2 id = (int2)(get_global_id(0), get_global_id(1));

  // TODO
}

// TODO
//
// Inner parts:
//  alpha := 1.0 / (viscousity * dt)
//  beta := 1.0 / (4.0 + alpha)
//
//  vL := inputVelocityBuffer[x - 1, y]
//  vR := inputVelocityBuffer[x + 1, y]
//  vB := inputVelocityBuffer[x, y - 1]
//  vT := inputVelocityBuffer[x, y + 1]
//
//  velocity := inputVelocityBuffer[x,y]
//
//  outputVelocityBuffer[x,y] := (vL + vR + vB + vT + alpha * velocity) * beta
// 
// Border:
//  outputVelocityBuffer[x,y] := inputVelocityBuffer[x,y]
__kernel
void diffusion(const int gridResolution,
	       __global float2* inputVelocityBuffer,
	       __global float2* outputVelocityBuffer){
  int2 id = (int2)(get_global_id(0), get_global_id(1));
  
  // TODO
  float alpha = 1.0 / (viscosity * dt);
  float beta = 1.0 / (4.0 + alpha);

  // if (id.x == 1 && id.y == 1) {
  //   printf("Diffusion Here! alpha=%f beta=%f\n", alpha, beta);
  // }
  
  if(id.x > 0 && id.x < gridResolution - 1 &&
     id.y > 0 && id.y < gridResolution - 1){
      float2 vL = inputVelocityBuffer[id.x - 1 + id.y*gridResolution];
      float2 vR = inputVelocityBuffer[id.x + 1 + id.y*gridResolution];
      float2 vB = inputVelocityBuffer[id.x + (id.y-1)*gridResolution];
      float2 vT = inputVelocityBuffer[id.x + (id.y+1)*gridResolution];

      float2 velocity = inputVelocityBuffer[id.x + id.y*gridResolution];

      outputVelocityBuffer[id.x + id.y*gridResolution] = (vL + vR + vB + vT + alpha * velocity) * beta;
    }
  }

// TODO
//
// Inner parts:
//  vL := velocityBuffer[x - 1, y]
//  vR := velocityBuffer[x + 1, y]
//  vB := velocityBuffer[x, y - 1]
//  vT := velocityBuffer[x, y + 1]
//  divergenceBuffer[x,y] := 0.5 * ((vR.x - vL.x) + (vT.y - vB.y))
//
// Border:
//  divergenceBuffer[x,y] := 0.0

__kernel
void divergence(const int gridResolution, __global float2* velocityBuffer,
		__global float* divergenceBuffer){
  int2 id = (int2)(get_global_id(0), get_global_id(1));

 // TODO
 if(id.x > 0 && id.x < gridResolution - 1 &&
     id.y > 0 && id.y < gridResolution - 1){
  float2 vL = velocityBuffer[id.x - 1 + id.y*gridResolution];
  float2 vR = velocityBuffer[id.x + 1 + id.y*gridResolution];
  float2 vB = velocityBuffer[id.x + (id.y-1)*gridResolution];
  float2 vT = velocityBuffer[id.x + (id.y+1)*gridResolution];
  divergenceBuffer[id.x + id.y*gridResolution]=0.5 * ((vR.x - vL.x) + (vT.y - vB.y));
  }
}

// TODO
//
// Inner parts:
//	alpha := -1.0
//	beta  := 0.25
//	vL := inputPressureBuffer[x - 1, y]
//	vR := inputPressureBuffer[x + 1, y]
//	vB := inputPressureBuffer[x, y - 1]
//	vT := inputPressureBuffer[x, y + 1]
//	divergence := divergenceBuffer[x,y]
//
//	ouputPressure := (vL + vR + vB + vT + divergence * alpha) * beta
//
// Border:
//  ouputPressure[x,y] = inputPressureBuffer (nearest neighbour)
__kernel
void pressureJacobi(const int gridResolution,
		    __global float* inputPressureBuffer,
		    __global float* outputPressureBuffer,
		    __global float* divergenceBuffer){
  int2 id = (int2)(get_global_id(0), get_global_id(1));

  // TODO
  float alpha = -1.0f;
	float beta = 0.25f;

  if(id.x > 0 && id.x < gridResolution - 1 &&
     id.y > 0 && id.y < gridResolution - 1){
  float vL = inputPressureBuffer[id.x - 1 + id.y*gridResolution];
  float vR = inputPressureBuffer[id.x + 1 + id.y*gridResolution];
  float vB = inputPressureBuffer[id.x + (id.y-1)*gridResolution];
  float vT = inputPressureBuffer[id.x + (id.y+1)*gridResolution];
  float divergence = divergenceBuffer[id.x + id.y*gridResolution];

  outputPressureBuffer[id.x + id.y*gridResolution] = (vL + vR + vB + vT + divergence * alpha) * beta;
  }
}

// TODO
//
// Inner parts:
//  pL := pressureBuffer[x - 1, y]
//  pR := pressureBuffer[x + 1, y]
//  pB := pressureBuffer[x, y - 1]
//  pT := pressureBuffer[x, y + 1]
//  velocity := inputVelocityBuffer[x,y]
//
//  outputVelocity[x,y] = velocity - (pR - pL, pT - pB)
//
// Border:
//  outputVelocity = -inputVelocity (nearest neighbour)
__kernel
void projection(const int gridResolution,
		__global float2* inputVelocityBuffer,
		__global float* pressureBuffer,
		__global float2* outputVelocityBuffer){
  int2 id = (int2)(get_global_id(0), get_global_id(1));

  // TODO
  if(id.x > 0 && id.x < gridResolution - 1 &&
     id.y > 0 && id.y < gridResolution - 1){
  float pL = pressureBuffer[id.x - 1 + id.y * gridResolution];
  float pR = pressureBuffer[id.x + 1 + id.y * gridResolution];
  float pB = pressureBuffer[id.x + (id.y - 1) * gridResolution];
  float pT = pressureBuffer[id.x + (id.y + 1) * gridResolution];
  float2 velocity = inputVelocityBuffer[id.x + id.y * gridResolution];

  outputVelocityBuffer[id.x + id.y * gridResolution] = velocity - (pR - pL, pT - pB);
  }
}

// TODO
//
// Add force to velocityBuffer
// Add matter to densityBuffer
//
// Example:
//  dx := global_id(0) / gridResolution - x
//  dy := global_id(1) / gridResolution - y
//  radius := 0.001
//  c := e^(-(dx^2 + dy^2) / radius) * dt
//
//  velocityBuffer[x,y] += c * force
//  densityBuffer[x,y]  += c * density
__kernel
void addForce(const float x, const float y, const float2 force,
	      const int gridResolution, __global float2* velocityBuffer,
	      const float4 density, __global float4* densityBuffer){
  int2 id = (int2)(get_global_id(0), get_global_id(1));

  // TODO
  float dx= id.x / (float)gridResolution - x;
  float dy= id.y / (float)gridResolution - y;
  float radius=0.001f;
  float c= exp(-(dx*dx + dy*dy) / radius) * dt;
  velocityBuffer[id.x + id.y * gridResolution] += c * force;
  densityBuffer[id.x + id.y * gridResolution] += c * density;
}

__kernel
void vorticity(const int gridResolution, __global float2* velocityBuffer,
	       __global float* vorticityBuffer){
  int2 id = (int2)(get_global_id(0), get_global_id(1));

  if(id.x > 0 && id.x < gridResolution - 1 &&
     id.y > 0 && id.y < gridResolution - 1){
    float2 vL = velocityBuffer[id.x - 1 + id.y * gridResolution];
    float2 vR = velocityBuffer[id.x + 1 + id.y * gridResolution];
    float2 vB = velocityBuffer[id.x + (id.y - 1) * gridResolution];
    float2 vT = velocityBuffer[id.x + (id.y + 1) * gridResolution];

    vorticityBuffer[id.x + id.y * gridResolution] = (vR.y - vL.y) - (vT.x - vB.x);
  } else {
    vorticityBuffer[id.x + id.y * gridResolution] = 0.0f;
  }
}

__kernel
void addVorticity(const int gridResolution, __global float* vorticityBuffer,
		  __global float2* velocityBuffer){
  int2 id = (int2)(get_global_id(0), get_global_id(1));

  const float scale = 0.2f;

  if(id.x > 0 && id.x < gridResolution - 1 &&
     id.y > 0 && id.y < gridResolution - 1){
    float vL = vorticityBuffer[id.x - 1 + id.y * gridResolution];
    float vR = vorticityBuffer[id.x + 1 + id.y * gridResolution];
    float vB = vorticityBuffer[id.x + (id.y - 1) * gridResolution];
    float vT = vorticityBuffer[id.x + (id.y + 1) * gridResolution];

    float4 gradV = (float4)(vR - vL, vT - vB, 0.0f, 0.0f);
    float4 z = (float4)(0.0f, 0.0f, 1.0f, 0.0f);

    if(dot(gradV, gradV)){
      float4 vorticityForce = scale * cross(gradV, z);
      velocityBuffer[id.x + id.y * gridResolution] += vorticityForce.xy * dt;
    }
  }
}

// *************
// Visualization
// *************

// TODO
// MAP densityBuffer TO visualizationBuffer
//     visualizationBuffer[x,y] := density
__kernel
void visualizationDensity(const int width, const int height, __global float4* visualizationBuffer,
			  const int gridResolution, __global float4* densityBuffer){
  int2 id = (int2)(get_global_id(0), get_global_id(1));

  if( id.x < width && id.y < height){
    // TODO
    float4 density = densityBuffer[id.x + id.y * width];
    visualizationBuffer[id.x + id.y * width] = density;
  }
}

// TODO
// MAP velocityBuffer TO visualizationBuffer
//     visualizationBuffer[x,y] := (1.0 + velocity) / 2.0
__kernel
void visualizationVelocity(const int width, const int height, __global float4* visualizationBuffer,
			   const int gridResolution, __global float2* velocityBuffer){
  int2 id = (int2)(get_global_id(0), get_global_id(1));

  if( id.x < width && id.y < height){
    // TODO
    float2 velocity = velocityBuffer[id.x + id.y * width];
    visualizationBuffer[id.x + id.y * width] = clamp((float4)((1.0f + velocity.x) / 2.0f, (1.0f + velocity.y) / 2.0f, 0.0f, 0.0f), 0.0f, 1.0f);
  }
}

// TODO
// MAP pressureBuffer TO visualizationBuffer
//     visualizationBuffer[x,y] := (1.0 + pressure) / 2.0
__kernel
void visualizationPressure(const int width, const int height, __global float4* visualizationBuffer,
			   const int gridResolution, __global float* pressureBuffer){
  int2 id = (int2)(get_global_id(0), get_global_id(1));

  if( id.x < width && id.y < height){
    float pressure = pressureBuffer[id.x + id.y * width];
    visualizationBuffer[id.x + id.y * width] = (float4)((1.0f + pressure)/2.0f);
  }
}
