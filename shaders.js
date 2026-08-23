// GLSL shader source for Orrery. Loaded before app.js; as top-level consts
// in a classic script they are visible to app.js via the shared global scope.

const NOISE = `
float hash(vec3 p){ p=fract(p*0.3183099+vec3(0.1,0.2,0.3)); p*=17.0;
  return fract(p.x*p.y*p.z*(p.x+p.y+p.z)); }
float vnoise(vec3 x){ vec3 i=floor(x),f=fract(x); f=f*f*(3.0-2.0*f);
  return mix(mix(mix(hash(i+vec3(0,0,0)),hash(i+vec3(1,0,0)),f.x),
                 mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),
             mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),
                 mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x),f.y),f.z);}
float fbm(vec3 p,int oct){ float s=0.0,a=0.5,t=0.0;
  for(int i=0;i<8;i++){ if(i>=oct)break; s+=a*vnoise(p); t+=a; p*=2.02; a*=0.5;} return s/t; }
float ridge(vec3 p,int oct){ float s=0.0,a=0.5,t=0.0;
  for(int i=0;i<8;i++){ if(i>=oct)break; float n=1.0-abs(2.0*vnoise(p)-1.0); s+=a*n*n; t+=a; p*=2.03; a*=0.5;} return s/t; }
`;

const VERT = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUv;
uniform mat4 uMVP, uModel; uniform mat3 uNorm;
out vec3 vWorld; out vec3 vNormal; out vec2 vUv; out vec3 vLocal;
void main(){ vec4 w=uModel*vec4(aPos,1.0); vWorld=w.xyz; vLocal=aPos;
  vNormal=normalize(uNorm*aPos); vUv=aUv; gl_Position=uMVP*vec4(aPos,1.0); }`;

/* ---- planets & moons ---- */
const PLANET_FS = `#version 300 es
precision highp float;
in vec3 vWorld; in vec3 vNormal; in vec2 vUv; in vec3 vLocal;
out vec4 frag;
uniform vec3 uSun, uCam;
uniform float uTime, uSeed;
uniform int uType;
uniform vec3 uColA, uColB, uColC;
${NOISE}
vec3 aces(vec3 x){return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14),0.0,1.0);}
void main(){
  vec3 N=normalize(vNormal);
  vec3 P=normalize(vLocal);
  vec3 L=normalize(uSun-vWorld);
  vec3 Vd=normalize(uCam-vWorld);
  float ndl=dot(N,L);
  float day=clamp(ndl,0.0,1.0);
  float seed=uSeed;
  vec3 sp=P*2.0+seed;

  vec3 alb; float spec=0.0; float shin=24.0; vec3 emit=vec3(0.0);

  if(uType==1){ // Mercury — cratered rock
    float c=fbm(sp*3.0,6); float cr=ridge(sp*7.0,5);
    alb=mix(uColA,uColB,c); alb*=0.75+0.5*cr; alb*=0.9+0.2*fbm(sp*22.0,4);
  } else if(uType==2){ // Venus — thick sulphur clouds
    vec3 w=sp; w.x+=uTime*0.02; float b=fbm(w*2.2+vec3(0.0,fbm(w*3.0,4),0.0),6);
    float sw=fbm(w*5.0+b,5); alb=mix(uColA,uColB,b*0.7+sw*0.4); alb=mix(alb,uColC,pow(sw,3.0)*0.5);
  } else if(uType==3){ // Earth
    float cont=fbm(sp*1.7,7)+0.12*ridge(sp*4.0,5);
    float lat=abs(P.y);
    float sea=smoothstep(0.49,0.5,cont);       // 1 land, 0 ocean
    vec3 ocean=mix(vec3(0.015,0.05,0.14), vec3(0.04,0.16,0.32), smoothstep(0.5,0.42,cont));
    float veg=fbm(sp*3.4+11.0,5);
    vec3 land=mix(vec3(0.09,0.16,0.05), vec3(0.32,0.28,0.14), smoothstep(0.3,0.75,veg));
    land=mix(land, vec3(0.22,0.17,0.10), smoothstep(0.55,0.62,cont)); // mountains/desert
    float ice=smoothstep(0.62,0.78,lat)+smoothstep(0.7,0.5,cont)*smoothstep(0.55,0.7,lat);
    land=mix(land, vec3(0.9,0.94,0.98), clamp(ice,0.0,1.0));
    alb=mix(ocean,land,sea);
    spec=(1.0-sea)*0.6; shin=60.0;
    // city lights on night side
    float pop=smoothstep(0.6,0.95,fbm(sp*9.0+3.0,5))*sea*(1.0-ice);
    emit=vec3(1.0,0.82,0.5)*pop*smoothstep(0.05,-0.25,ndl)*1.6;
  } else if(uType==4){ // Mars
    float d=fbm(sp*2.4,6); float cr=ridge(sp*6.0,5);
    alb=mix(uColA,uColB,d); alb*=0.8+0.4*cr;
    alb=mix(alb, vec3(0.35,0.14,0.08), smoothstep(0.6,0.8,fbm(sp*3.0+7.0,4))); // dark basalt
    float lat=abs(P.y); alb=mix(alb, vec3(0.95,0.96,0.99), smoothstep(0.82,0.9,lat)); // caps
  } else if(uType==5){ // Jupiter — turbulent bands + Great Red Spot
    float lat=P.y;
    float warp=fbm(sp*vec3(3.0,7.0,3.0)+vec3(uTime*0.05,0.0,0.0),6);
    float bands=sin((lat*13.0)+warp*3.2)+0.42*sin(lat*33.0+warp*5.0);
    float t=clamp(bands*0.55+0.5,0.0,1.0);
    t=clamp((t-0.5)*1.4+0.5,0.0,1.0);                 // crisper band edges
    alb=mix(uColA,uColB,t);
    alb=mix(alb,uColC,smoothstep(0.62,0.95,fbm(sp*9.0+warp,5))*0.7);
    // GRS
    vec3 gp=P; float lon=atan(gp.z,gp.x);
    vec2 d=vec2((lon+2.0)*1.0,(gp.y+0.28)*2.6);
    float grs=smoothstep(1.0,0.0,length(d));
    alb=mix(alb, vec3(0.7,0.24,0.14), grs*0.9);
    alb*=0.92+0.16*fbm(sp*20.0,4);
  } else if(uType==6){ // Saturn — soft bands
    float lat=P.y; float warp=fbm(sp*vec3(2.0,6.0,2.0)+vec3(uTime*0.03,0.0,0.0),5);
    float bands=sin((lat*9.0)+warp*2.0)*0.5+0.5;
    alb=mix(uColA,uColB,bands); alb=mix(alb,uColC,smoothstep(0.65,0.95,bands));
    alb*=0.95+0.1*fbm(sp*16.0,4);
  } else if(uType==7){ // Uranus — smooth pale cyan
    float lat=P.y; float bands=sin(lat*7.0+fbm(sp*3.0,4))*0.5+0.5;
    alb=mix(uColA,uColB,bands*0.4); alb=mix(alb,uColC,pow(bands,4.0)*0.3);
  } else if(uType==8){ // Neptune — deep blue + storms
    vec3 w=sp; w.x+=uTime*0.04; float warp=fbm(w*vec3(2.5,6.0,2.5),6);
    float bands=sin(P.y*8.0+warp*2.0)*0.5+0.5;
    alb=mix(uColA,uColB,bands);
    float storm=smoothstep(0.72,0.95,fbm(w*4.0+2.0,5)); alb=mix(alb,uColC,storm*0.8);
    alb*=0.94+0.12*fbm(sp*14.0,4);
  } else { // moons — cratered, tinted by uColA; icy sheen via uColB.x
    float c=fbm(sp*3.5,6), cr=ridge(sp*8.0,5);
    alb=mix(uColA*0.7,uColA,c); alb*=0.7+0.6*cr; alb*=0.9+0.2*fbm(sp*24.0,4);
    spec=uColB.x; shin=40.0;
    // Io-style sulphur mottling / Europa cracks encoded via uColC
    alb=mix(alb,uColC,smoothstep(0.6,0.85,fbm(sp*5.0+seed,5))*uColB.y);
  }

  // lighting
  float amb=0.035;
  vec3 col=alb*(day+amb);
  // specular (oceans / ice)
  if(spec>0.0){ vec3 H=normalize(L+Vd); col+=vec3(1.0,0.98,0.92)*pow(max(dot(N,H),0.0),shin)*spec*day; }
  // soft terminator warm scatter
  float term=smoothstep(0.0,0.25,ndl)*smoothstep(0.5,0.0,ndl);
  col+=alb*vec3(0.4,0.18,0.06)*term*0.5;
  col+=emit;

  frag=vec4(col,1.0);
}`;

/* ---- sun ---- */
const SUN_FS = `#version 300 es
precision highp float;
in vec3 vWorld; in vec3 vNormal; in vec3 vLocal; in vec2 vUv;
out vec4 frag; uniform float uTime; uniform vec3 uCam;
${NOISE}
void main(){
  vec3 P=normalize(vLocal);
  vec3 g=P*3.0; g+=vec3(uTime*0.06,uTime*0.03,-uTime*0.05);
  float gran=fbm(g*3.0,6);
  float turb=ridge(g*6.0+gran,5);
  float f=gran*0.6+turb*0.6;
  vec3 hot=vec3(1.0,0.95,0.7), mid=vec3(1.0,0.6,0.15), cool=vec3(0.85,0.22,0.05);
  vec3 col=mix(cool,mid,smoothstep(0.25,0.55,f)); col=mix(col,hot,smoothstep(0.55,0.85,f));
  // limb darkening (brighter to camera-facing center)
  vec3 Vd=normalize(uCam-vWorld); float limb=pow(max(dot(normalize(vNormal),Vd),0.0),0.4);
  col*=0.6+0.9*limb;
  col*=3.4; // HDR punch for bloom
  frag=vec4(col,1.0);
}`;

/* ---- atmosphere shell (additive rim) ---- */
const ATM_FS = `#version 300 es
precision highp float;
in vec3 vWorld; in vec3 vNormal;
out vec4 frag; uniform vec3 uSun,uCam,uColA; uniform float uPow,uStr;
void main(){ vec3 N=normalize(vNormal); vec3 Vd=normalize(uCam-vWorld);
  vec3 L=normalize(uSun-vWorld);
  float rim=pow(1.0-max(dot(N,Vd),0.0),uPow);
  float lit=clamp(dot(N,L)+0.3,0.0,1.0);
  frag=vec4(uColA*rim*uStr*lit, 1.0); }`;

/* ---- rings ---- */
const RING_FS = `#version 300 es
precision highp float;
in vec3 vWorld; in vec2 vUv;
out vec4 frag; uniform vec3 uSun,uCam,uColA,uColB,uPlanet; uniform float uPr;
${NOISE}
void main(){
  float r=vUv.x;                          // 0 inner .. 1 outer
  float band=fbm(vec3(r*140.0,0.0,0.0),6);
  float fine=fbm(vec3(r*520.0,3.0,0.0),5);
  float dens=0.35+0.65*band; dens*=0.6+0.5*fine;
  // Cassini division
  dens*=smoothstep(0.02,0.05,abs(r-0.62));
  dens*=smoothstep(0.0,0.03,r)*smoothstep(1.0,0.97,r);
  vec3 col=mix(uColB,uColA,band);
  // planet shadow across the ring
  vec3 L=normalize(uSun-vWorld); vec3 toC=uPlanet-vWorld;
  float tproj=dot(toC,L); vec3 closest=vWorld+L*max(tproj,0.0);
  float dcl=length(closest-uPlanet); float sh=(tproj>0.0)?smoothstep(uPr*0.9,uPr*1.1,dcl):1.0;
  col*=0.25+0.75*sh;
  float a=clamp(dens,0.0,1.0)*0.92;
  frag=vec4(col*a, a);
}`;

/* ---- orbit line ---- */
const ORBIT_VS = `#version 300 es
layout(location=0) in vec3 aPos; uniform mat4 uMVP;
void main(){ gl_Position=uMVP*vec4(aPos,1.0); }`;
const ORBIT_FS = `#version 300 es
precision highp float; out vec4 frag; uniform vec3 uCol; uniform float uA;
void main(){ frag=vec4(uCol*uA,1.0); }`;

/* ---- asteroids (instanced) ---- */
const AST_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=2) in vec4 iPosScale;  // xyz pos, w scale
layout(location=3) in vec3 iRot;       // euler
uniform mat4 uVP; uniform vec3 uSun;
out vec3 vN; out vec3 vW; out float vSeed;
mat3 rot(vec3 e){ float cx=cos(e.x),sx=sin(e.x),cy=cos(e.y),sy=sin(e.y),cz=cos(e.z),sz=sin(e.z);
  return mat3(cy*cz,cy*sz,-sy, sx*sy*cz-cx*sz,sx*sy*sz+cx*cz,sx*cy, cx*sy*cz+sx*sz,cx*sy*sz-sx*cz,cx*cy); }
void main(){ mat3 R=rot(iRot); vec3 p=R*(aPos*iPosScale.w)+iPosScale.xyz;
  vN=normalize(R*aPos); vW=p; vSeed=iPosScale.w;
  gl_Position=uVP*vec4(p,1.0); }`;
const AST_FS = `#version 300 es
precision highp float; in vec3 vN; in vec3 vW; in float vSeed; out vec4 frag;
uniform vec3 uSun;
void main(){ vec3 L=normalize(uSun-vW); float d=max(dot(normalize(vN),L),0.0);
  vec3 c=vec3(0.32,0.29,0.26)*(0.15+d); frag=vec4(c,1.0); }`;

/* ---- stars (points) ---- */
const STAR_VS = `#version 300 es
layout(location=0) in vec3 aPos; layout(location=1) in vec2 aBS; // brightness,size
uniform mat4 uVP; uniform float uPx; out float vB; out vec3 vC;
void main(){ gl_Position=uVP*vec4(aPos,1.0); gl_PointSize=aBS.y*uPx; vB=aBS.x;
  float t=fract(aPos.x*13.1+aPos.y*7.7); vC=mix(vec3(0.7,0.8,1.0),vec3(1.0,0.85,0.7),t); }`;
const STAR_FS = `#version 300 es
precision highp float; in float vB; in vec3 vC; out vec4 frag;
void main(){ vec2 d=gl_PointCoord-0.5; float r=length(d);
  float a=smoothstep(0.5,0.0,r); a=pow(a,1.6);
  frag=vec4(vC*vB, a); }`;

/* ---- background sky (milky way) ---- */
const SKY_VS = `#version 300 es
layout(location=0) in vec3 aPos; uniform mat4 uVP; out vec3 vD;
void main(){ vD=aPos; vec4 p=uVP*vec4(aPos,1.0); gl_Position=p.xyww; }`;
const SKY_FS = `#version 300 es
precision highp float; in vec3 vD; out vec4 frag;
${NOISE}
void main(){ vec3 d=normalize(vD);
  float band=exp(-pow((d.y-0.02)*3.2,2.0)*1.6);       // milky way plane
  float clouds=fbm(d*6.0,6)*fbm(d*2.0+5.0,4);
  float dust=fbm(d*10.0+2.0,5);
  vec3 mw=mix(vec3(0.03,0.04,0.07), vec3(0.14,0.13,0.17), clouds);
  mw=mix(mw, vec3(0.20,0.16,0.20), smoothstep(0.4,0.9,clouds));
  mw*=band*(0.6+0.8*clouds); mw*=1.0-0.6*smoothstep(0.4,0.8,dust)*band;
  vec3 base=vec3(0.006,0.008,0.016)*(1.0-0.3*d.y);
  frag=vec4(base+mw*0.5,1.0);
}`;

/* ---- fullscreen passes ---- */
const FS_VS = `#version 300 es
layout(location=0) in vec2 aPos; out vec2 vUv;
void main(){ vUv=aPos*0.5+0.5; gl_Position=vec4(aPos,0.0,1.0); }`;
const BRIGHT_FS = `#version 300 es
precision highp float; in vec2 vUv; out vec4 frag; uniform sampler2D uTex;
void main(){ vec3 c=texture(uTex,vUv).rgb; float l=dot(c,vec3(0.299,0.587,0.114));
  float k=smoothstep(0.9,1.6,l); frag=vec4(c*k,1.0); }`;
const BLUR_FS = `#version 300 es
precision highp float; in vec2 vUv; out vec4 frag;
uniform sampler2D uTex; uniform vec2 uDir;
void main(){ vec2 t=uDir;
  vec3 c=texture(uTex,vUv).rgb*0.227;
  c+=texture(uTex,vUv+t*1.384).rgb*0.316;
  c+=texture(uTex,vUv-t*1.384).rgb*0.316;
  c+=texture(uTex,vUv+t*3.230).rgb*0.070;
  c+=texture(uTex,vUv-t*3.230).rgb*0.070;
  frag=vec4(c,1.0); }`;
const COMP_FS = `#version 300 es
precision highp float; in vec2 vUv; out vec4 frag;
uniform sampler2D uScene,uBloom; uniform vec2 uPx;
vec3 aces(vec3 x){return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14),0.0,1.0);}
void main(){
  // FXAA-lite on the scene
  vec3 c =texture(uScene,vUv).rgb;
  vec3 nw=texture(uScene,vUv+vec2(-uPx.x,-uPx.y)).rgb;
  vec3 ne=texture(uScene,vUv+vec2( uPx.x,-uPx.y)).rgb;
  vec3 sw=texture(uScene,vUv+vec2(-uPx.x, uPx.y)).rgb;
  vec3 se=texture(uScene,vUv+vec2( uPx.x, uPx.y)).rgb;
  vec3 blur=(nw+ne+sw+se+c)*0.2;
  vec3 lum=vec3(0.299,0.587,0.114);
  float e=abs(dot(nw,lum)-dot(se,lum))+abs(dot(ne,lum)-dot(sw,lum));
  vec3 scene=mix(c,blur,clamp(e*2.2,0.0,1.0));
  vec3 bloom=texture(uBloom,vUv).rgb;
  vec3 col=scene+bloom*0.75;
  col=aces(col*1.05);
  col=pow(col,vec3(0.9));                 // gentle contrast
  // subtle vignette
  vec2 q=vUv-0.5; col*=1.0-dot(q,q)*0.5;
  frag=vec4(col,1.0);
}`;
