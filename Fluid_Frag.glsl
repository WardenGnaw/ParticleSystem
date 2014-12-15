struct Material {
  vec3 aColor;
  vec3 dColor;
  vec3 sColor;
  float shine;
};

uniform Material uMat;
uniform vec4 cameraPos;

varying vec3 vColor;
varying vec4 normal;
varying vec4 light;
varying vec4 spec;

void main() {
  float tempSpec;
  vec3 Refl;
  vec3 lColor = vec3(1, 1, 1);
  vec4 fNormal = normalize(normal);
  vec4 fLight = normalize(light);

  Refl = lColor * max(0.0, dot(fLight, fNormal)) * uMat.dColor + lColor * uMat.aColor;
  tempSpec = dot(normalize(spec), -fLight + 2.0 * dot(fLight, fNormal) * fNormal);
  Refl += lColor * clamp(pow(tempSpec, uMat.shine), 0.0, 1.0) * uMat.sColor;
  gl_FragColor = vec4(Refl.x, Refl.y, Refl.z, 1.0);
}
