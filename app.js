"use strict";
(function(){
const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl2", {antialias:false, alpha:false, powerPreference:"high-performance", preserveDrawingBuffer:false});
if(!gl){ document.getElementById("boot").style.display="none";
  const e=document.getElementById("err"); e.style.display="flex";
  e.textContent="This experience needs WebGL2, which your browser or device did not provide."; return; }

/* ============================ math ============================ */
const V3={
  add:(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]],
  sub:(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]],
  scale:(a,s)=>[a[0]*s,a[1]*s,a[2]*s],
  dot:(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],
  len:a=>Math.hypot(a[0],a[1],a[2]),
  norm:a=>{const l=Math.hypot(a[0],a[1],a[2])||1;return[a[0]/l,a[1]/l,a[2]/l];},
  cross:(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],
};
const M4={
  ident:()=>[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
  mul:(a,b)=>{const o=new Array(16);
    for(let r=0;r<4;r++)for(let c=0;c<4;c++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s;}return o;},
  persp:(fovy,asp,n,f)=>{const t=1/Math.tan(fovy/2);return[t/asp,0,0,0, 0,t,0,0, 0,0,(f+n)/(n-f),-1, 0,0,2*f*n/(n-f),0];},
  lookAt:(eye,ctr,up)=>{const z=V3.norm(V3.sub(eye,ctr)),x=V3.norm(V3.cross(up,z)),y=V3.cross(z,x);
    return[x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
      -V3.dot(x,eye),-V3.dot(y,eye),-V3.dot(z,eye),1];},
  trans:(m,v)=>M4.mul(m,[1,0,0,0,0,1,0,0,0,0,1,0,v[0],v[1],v[2],1]),
  scale:(m,s)=>M4.mul(m,[s,0,0,0,0,s,0,0,0,0,s,0,0,0,0,1]),
  rotX:a=>{const c=Math.cos(a),s=Math.sin(a);return[1,0,0,0,0,c,s,0,0,-s,c,0,0,0,0,1];},
  rotY:a=>{const c=Math.cos(a),s=Math.sin(a);return[c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1];},
  rotZ:a=>{const c=Math.cos(a),s=Math.sin(a);return[c,s,0,0,-s,c,0,0,0,0,1,0,0,0,0,1];},
  // normal matrix (upper-left inverse-transpose) as mat3 in vec-friendly layout
  norm3:m=>{ // returns 9 floats
    const a=m[0],b=m[1],c=m[2],d=m[4],e=m[5],f=m[6],g=m[8],h=m[9],i=m[10];
    const A=e*i-f*h, B=f*g-d*i, C=d*h-e*g;
    const det=a*A+b*B+c*C || 1, id=1/det;
    return [A*id,B*id,C*id,
            (c*h-b*i)*id,(a*i-c*g)*id,(b*g-a*h)*id,
            (b*f-c*e)*id,(c*d-a*f)*id,(a*e-b*d)*id];
  }
};

/* ============================ gl helpers ============================ */
function sh(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)+"\n"+src.split("\n").map((l,i)=>(i+1)+": "+l).join("\n"));
  return s;}
function prog(vs,fs){const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,vs));gl.attachShader(p,sh(gl.FRAGMENT_SHADER,fs));
  gl.linkProgram(p); if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p;}
function uni(p){const n=gl.getProgramParameter(p,gl.ACTIVE_UNIFORMS),o={};
  for(let i=0;i<n;i++){const u=gl.getActiveUniform(p,i);o[u.name.replace(/\[0\]$/,"")]=gl.getUniformLocation(p,u.name);}return o;}

/* float render targets for HDR bloom (graceful fallback) */
const cbf = gl.getExtension("EXT_color_buffer_float");
gl.getExtension("OES_texture_float_linear");
const HDR_IF = cbf ? gl.RGBA16F : gl.RGBA8;
const HDR_T  = cbf ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;

function tex(w,h,internal,type,filter){
  const t=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,t);
  gl.texImage2D(gl.TEXTURE_2D,0,internal,w,h,0,gl.RGBA,type,null);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,filter);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,filter);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  return t;
}
function fbo(t,depth){const f=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,f);
  gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,t,0);
  let d=null; if(depth){d=gl.createRenderbuffer();gl.bindRenderbuffer(gl.RENDERBUFFER,d);
    gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT24,t.w||1,t.h||1);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,d);}
  gl.bindFramebuffer(gl.FRAMEBUFFER,null); return {f,d};}

/* ============================ geometry ============================ */
function sphere(seg,ring){
  const pos=[],uv=[],idx=[];
  for(let y=0;y<=ring;y++){const v=y/ring,th=v*Math.PI,st=Math.sin(th),ct=Math.cos(th);
    for(let x=0;x<=seg;x++){const u=x/seg,ph=u*Math.PI*2,sp=Math.sin(ph),cp=Math.cos(ph);
      pos.push(sp*st,ct,cp*st); uv.push(u,v);}}
  for(let y=0;y<ring;y++)for(let x=0;x<seg;x++){const a=y*(seg+1)+x,b=a+seg+1;
    idx.push(a,b,a+1, b,b+1,a+1);}
  return mesh(pos,uv,idx);
}
function mesh(pos,uv,idx){
  const va=gl.createVertexArray(); gl.bindVertexArray(va);
  const pb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,pb);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(pos),gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
  if(uv){const ub=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,ub);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(uv),gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1,2,gl.FLOAT,false,0,0);}
  const ib=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint32Array(idx),gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  return {va,count:idx.length};
}
// flat annulus in XZ plane, radial coordinate in uv.x (0=inner,1=outer)
function ringGeo(inner,outer,seg){
  const pos=[],uv=[],idx=[];
  for(let i=0;i<=seg;i++){const a=i/seg*Math.PI*2,c=Math.cos(a),s=Math.sin(a);
    pos.push(c*inner,0,s*inner); uv.push(0,i/seg);
    pos.push(c*outer,0,s*outer); uv.push(1,i/seg);}
  for(let i=0;i<seg;i++){const b=i*2; idx.push(b,b+1,b+2, b+1,b+3,b+2);}
  return mesh(pos,uv,idx);
}
/* shaders: GLSL source lives in shaders.js (loaded before this file) */

let P;
try{
P={
  planet:{p:prog(VERT,PLANET_FS)}, sun:{p:prog(VERT,SUN_FS)}, atm:{p:prog(VERT,ATM_FS)},
  ring:{p:prog(VERT,RING_FS)}, orbit:{p:prog(ORBIT_VS,ORBIT_FS)}, ast:{p:prog(AST_VS,AST_FS)},
  star:{p:prog(STAR_VS,STAR_FS)}, sky:{p:prog(SKY_VS,SKY_FS)},
  bright:{p:prog(FS_VS,BRIGHT_FS)}, blur:{p:prog(FS_VS,BLUR_FS)}, comp:{p:prog(FS_VS,COMP_FS)}
};
for(const k in P) P[k].u=uni(P[k].p);
}catch(err){ boot.style.display="none"; const e=document.getElementById("err");
  e.style.display="flex"; e.textContent="Shader error: "+err.message; console.error(err); return; }

/* ============================ scene data ============================ */
// visual scales are compressed for framing; facts in the panel are real.
const SUN_R=13;
const BODIES=[
  {name:"Mercury",type:1,r:0.85,orbit:24,period:0.42,rot:0.018,tilt:0.001,
   colA:[0.42,0.38,0.34],colB:[0.62,0.58,0.52],
   sub:"Terrestrial planet",
   desc:"The smallest planet and the closest to the Sun — a scorched, airless world of craters, its surface locked between blistering days and frozen nights.",
   facts:{Diameter:"4,879 km","From Sun":"57.9M km","Day":"58.6 days",Year:"88 days",Moons:"0"}},
  {name:"Venus",type:2,r:1.9,orbit:33,period:0.62,rot:0.006,tilt:3.096,
   colA:[0.85,0.72,0.42],colB:[0.95,0.86,0.6],colC:[0.7,0.55,0.28],
   atm:[0.9,0.75,0.4,3.0,0.5],
   sub:"Terrestrial planet",
   desc:"Earth's blistering twin, wrapped in crushing clouds of sulphuric acid. A runaway greenhouse makes it the hottest planet — hotter even than Mercury.",
   facts:{Diameter:"12,104 km","From Sun":"108.2M km",Day:"243 days",Year:"225 days",Moons:"0"}},
  {name:"Earth",type:3,r:2.0,orbit:44,period:1.0,rot:0.09,tilt:0.409,
   colA:[0.1,0.3,0.5],colB:[0.2,0.5,0.3],
   atm:[0.35,0.6,1.0,3.2,0.85], cloud:true,
   sub:"Our home world",
   desc:"The only known world with liquid oceans, breathable air and life. A pale blue dot cloaked in weather, its single Moon steadies its seasons.",
   facts:{Diameter:"12,742 km","From Sun":"149.6M km",Day:"24 hours",Year:"365.25 days",Moons:"1"}},
  {name:"Mars",type:4,r:1.15,orbit:56,period:1.88,rot:0.088,tilt:0.44,
   colA:[0.55,0.26,0.14],colB:[0.75,0.4,0.22],
   atm:[0.8,0.5,0.35,4.0,0.25],
   sub:"The red planet",
   desc:"A cold desert world of rust-red dust, dead volcanoes and polar ice. It once had rivers and lakes — and remains our most likely second home.",
   facts:{Diameter:"6,779 km","From Sun":"227.9M km",Day:"24.6 hours",Year:"687 days",Moons:"2"}},
  {name:"Jupiter",type:5,r:6.4,orbit:88,period:11.86,rot:0.22,tilt:0.055,
   colA:[0.72,0.58,0.42],colB:[0.9,0.82,0.68],colC:[0.6,0.42,0.3],
   sub:"Gas giant",
   desc:"The giant of the system — more massive than all other planets combined. Its Great Red Spot is a storm wider than Earth that has raged for centuries.",
   facts:{Diameter:"139,820 km","From Sun":"778.5M km",Day:"9.9 hours",Year:"11.9 years",Moons:"95"}},
  {name:"Saturn",type:6,r:5.4,orbit:122,period:29.4,rot:0.2,tilt:0.466,
   colA:[0.82,0.72,0.5],colB:[0.92,0.85,0.66],colC:[0.98,0.92,0.78],
   ring:{inner:6.6,outer:11.5,colA:[0.86,0.78,0.62],colB:[0.5,0.44,0.34]},
   sub:"The ringed giant",
   desc:"Crowned by the most spectacular rings in the system — countless icy particles spanning a width greater than the Earth–Moon distance, yet only metres thick.",
   facts:{Diameter:"116,460 km","From Sun":"1.43B km",Day:"10.7 hours",Year:"29.4 years",Moons:"146"}},
  {name:"Uranus",type:7,r:3.3,orbit:152,period:84,rot:0.14,tilt:1.706,
   colA:[0.5,0.78,0.82],colB:[0.66,0.86,0.88],colC:[0.8,0.94,0.94],
   atm:[0.6,0.85,0.9,3.4,0.5],
   ring:{inner:4.2,outer:5.6,colA:[0.4,0.5,0.55],colB:[0.2,0.28,0.32],thin:true},
   sub:"Ice giant",
   desc:"Tipped fully onto its side by an ancient collision, Uranus rolls around the Sun like a barrel. Its serene blue-green comes from methane in a frigid atmosphere.",
   facts:{Diameter:"50,724 km","From Sun":"2.87B km",Day:"17.2 hours",Year:"84 years",Moons:"28"}},
  {name:"Neptune",type:8,r:3.2,orbit:180,period:164.8,rot:0.15,tilt:0.494,
   colA:[0.14,0.3,0.7],colB:[0.2,0.42,0.85],colC:[0.5,0.7,0.95],
   atm:[0.3,0.45,0.95,3.3,0.7],
   sub:"Ice giant",
   desc:"The farthest planet — a deep-blue world of supersonic winds, the fastest in the system. Dark storms the size of Earth appear and vanish in its methane skies.",
   facts:{Diameter:"49,244 km","From Sun":"4.50B km",Day:"16 hours",Year:"164.8 years",Moons:"16"}},
];
const MOONS=[
  {p:"Earth",name:"Moon",r:0.55,orbit:3.6,period:0.25,colA:[0.5,0.5,0.52],ic:[0.6,0.6,0.62]},
  {p:"Mars",name:"Phobos",r:0.16,orbit:1.9,period:0.05,colA:[0.34,0.3,0.27]},
  {p:"Mars",name:"Deimos",r:0.12,orbit:2.6,period:0.09,colA:[0.36,0.32,0.28]},
  {p:"Jupiter",name:"Io",r:0.5,orbit:9.5,period:0.12,colA:[0.85,0.78,0.35],ic:[0.9,0.55,0.2],sheen:[0.0,0.7]},
  {p:"Jupiter",name:"Europa",r:0.45,orbit:11.5,period:0.2,colA:[0.82,0.8,0.75],ic:[0.6,0.72,0.85],sheen:[0.5,0.6]},
  {p:"Jupiter",name:"Ganymede",r:0.62,orbit:13.8,period:0.29,colA:[0.55,0.52,0.5],ic:[0.7,0.68,0.66],sheen:[0.2,0.4]},
  {p:"Jupiter",name:"Callisto",r:0.58,orbit:16.5,period:0.42,colA:[0.4,0.37,0.35]},
  {p:"Saturn",name:"Titan",r:0.6,orbit:14.5,period:0.3,colA:[0.78,0.55,0.2],ic:[0.85,0.62,0.25],sheen:[0.1,0.5]},
  {p:"Saturn",name:"Rhea",r:0.3,orbit:17.5,period:0.22,colA:[0.6,0.6,0.62]},
  {p:"Saturn",name:"Enceladus",r:0.24,orbit:12.8,period:0.16,colA:[0.9,0.92,0.95],sheen:[0.7,0.3]},
  {p:"Uranus",name:"Titania",r:0.34,orbit:6.4,period:0.18,colA:[0.5,0.5,0.52]},
  {p:"Uranus",name:"Oberon",r:0.32,orbit:7.8,period:0.26,colA:[0.46,0.44,0.44]},
  {p:"Neptune",name:"Triton",r:0.42,orbit:6.6,period:-0.24,colA:[0.7,0.72,0.75],ic:[0.8,0.82,0.85],sheen:[0.3,0.4]},
];

/* meshes */
const meshPlanet=sphere(140,80), meshMoon=sphere(64,36);
const quad=(()=>{const va=gl.createVertexArray();gl.bindVertexArray(va);
  const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);gl.bindVertexArray(null);return va;})();

/* orbit line buffers */
function orbitBuf(radius){const n=256,pos=[];
  for(let i=0;i<=n;i++){const a=i/n*Math.PI*2;pos.push(Math.cos(a)*radius,0,Math.sin(a)*radius);}
  const va=gl.createVertexArray();gl.bindVertexArray(va);
  const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(pos),gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);gl.bindVertexArray(null);
  return {va,count:n+1};}
BODIES.forEach(b=>b._orbit=orbitBuf(b.orbit));

/* rings */
BODIES.forEach(b=>{ if(b.ring) b._ring=ringGeo(b.ring.inner,b.ring.outer,220); });

/* asteroid belt + kuiper (instanced) — rock geometry + per-instance transform in one VAO */
function makeBelt(count,rIn,rOut,yspread){
  // rock geometry data
  let v=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
  let f=[[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]];
  for(let s=0;s<1;s++){const nf=[];for(const t of f){const [a,b,c]=t.map(i=>v[i]);
    const ab=V3.norm(V3.add(a,b)),bc=V3.norm(V3.add(b,c)),ca=V3.norm(V3.add(c,a));
    const ia=v.push(ab)-1,ib=v.push(bc)-1,ic=v.push(ca)-1;
    nf.push([t[0],ia,ic],[ia,t[1],ib],[ic,ib,t[2]],[ia,ib,ic]);} f=nf;}
  const pos=[],idx=[];
  v=v.map(p=>{const n=V3.norm(p);const h=Math.abs((Math.sin(n[0]*91.7+n[1]*47.3+n[2]*33.1)*43758.5)%1);
    return V3.scale(n,0.7+h*0.5);});
  for(const p of v)pos.push(p[0],p[1],p[2]); for(const t of f)idx.push(t[0],t[1],t[2]);
  const inst=new Float32Array(count*7);
  for(let i=0;i<count;i++){const a=Math.random()*Math.PI*2, rr=rIn+Math.random()*(rOut-rIn);
    inst.set([Math.cos(a)*rr,(Math.random()-0.5)*yspread,Math.sin(a)*rr,
      0.05+Math.random()*Math.random()*0.4, Math.random()*6.28,Math.random()*6.28,Math.random()*6.28],i*7);}
  const va=gl.createVertexArray(); gl.bindVertexArray(va);
  const pb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,pb);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(pos),gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
  const ibuf=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,ibuf);
  gl.bufferData(gl.ARRAY_BUFFER,inst,gl.STATIC_DRAW);
  gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2,4,gl.FLOAT,false,28,0); gl.vertexAttribDivisor(2,1);
  gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3,3,gl.FLOAT,false,28,16); gl.vertexAttribDivisor(3,1);
  const eb=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,eb);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint32Array(idx),gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  return {va,index:idx.length,count};
}
const asteroids=makeBelt(1600,66,80,2.2);
const kuiper=makeBelt(1400,200,250,6.0);

/* stars */
function makeStars(n){
  const pos=new Float32Array(n*3), bs=new Float32Array(n*2);
  for(let i=0;i<n;i++){const u=Math.random()*2-1,th=Math.random()*Math.PI*2,r=Math.sqrt(1-u*u);
    const R=1800+Math.random()*400;
    pos.set([Math.cos(th)*r*R,u*R,Math.sin(th)*r*R],i*3);
    const b=Math.pow(Math.random(),3.0);
    bs.set([0.4+b*1.6, 1.0+b*3.2],i*2);}
  const va=gl.createVertexArray();gl.bindVertexArray(va);
  const pb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pb);
  gl.bufferData(gl.ARRAY_BUFFER,pos,gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
  const bb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,bb);
  gl.bufferData(gl.ARRAY_BUFFER,bs,gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,2,gl.FLOAT,false,0,0);
  gl.bindVertexArray(null); return {va,count:n};
}
const stars=makeStars(4200);
/* sky cube */
const skyMesh=(()=>{const s=1; const p=[-s,-s,-s,s,-s,-s,s,s,-s,-s,s,-s,-s,-s,s,s,-s,s,s,s,s,-s,s,s];
  const idx=[0,1,2,0,2,3,5,4,7,5,7,6,4,0,3,4,3,7,1,5,6,1,6,2,3,2,6,3,6,7,4,5,1,4,1,0];
  const va=gl.createVertexArray();gl.bindVertexArray(va);
  const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(p),gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
  const e=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,e);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(idx),gl.STATIC_DRAW);
  gl.bindVertexArray(null);return {va,count:idx.length};})();

/* ============================ camera ============================ */
const cam={theta:0.9,phi:1.15,tTheta:0.9,tPhi:1.15,dist:230,tDist:230,target:[0,0,0],tTarget:[0,0,0],follow:null};
let simT=0, speed=1, paused=false, showOrbits=true, showLabels=true, selected=null;
const SPEEDS=[0,0.25,0.5,1,2,4,8,20]; let spIdx=3;

/* ============================ FBOs ============================ */
let W,H, sceneTex,sceneFBO, briTex,briFBO, b1Tex,b1FBO, b2Tex,b2FBO, BW,BH;
function resize(){
  const dpr=Math.min(window.devicePixelRatio||1,2);
  W=Math.floor(innerWidth*dpr); H=Math.floor(innerHeight*dpr);
  canvas.width=W; canvas.height=H; canvas.style.width=innerWidth+"px"; canvas.style.height=innerHeight+"px";
  BW=Math.max(1,W>>1); BH=Math.max(1,H>>1);
  [sceneTex,briTex,b1Tex,b2Tex].forEach(t=>t&&gl.deleteTexture(t));
  sceneTex=tex(W,H,HDR_IF,HDR_T,gl.LINEAR); sceneTex.w=W; sceneTex.h=H; sceneFBO=fbo(sceneTex,true);
  briTex=tex(BW,BH,HDR_IF,HDR_T,gl.LINEAR); briTex.w=BW; briTex.h=BH; briFBO=fbo(briTex,false);
  b1Tex=tex(BW,BH,HDR_IF,HDR_T,gl.LINEAR); b1Tex.w=BW; b1Tex.h=BH; b1FBO=fbo(b1Tex,false);
  b2Tex=tex(BW,BH,HDR_IF,HDR_T,gl.LINEAR); b2Tex.w=BW; b2Tex.h=BH; b2FBO=fbo(b2Tex,false);
}
window.addEventListener("resize",resize); resize();

/* ============================ interaction ============================ */
let drag=false,px=0,py=0,moved=0;
canvas.addEventListener("pointerdown",e=>{drag=true;moved=0;px=e.clientX;py=e.clientY;canvas.classList.add("grabbing");canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener("pointermove",e=>{
  if(drag){const dx=e.clientX-px,dy=e.clientY-py;px=e.clientX;py=e.clientY;moved+=Math.abs(dx)+Math.abs(dy);
    cam.theta-=dx*0.005; cam.phi=Math.max(0.08,Math.min(Math.PI-0.08,cam.phi-dy*0.005));
    cam.tTheta=cam.theta; cam.tPhi=cam.phi;}
  else{ hoverPick(e.clientX,e.clientY); }
});
canvas.addEventListener("pointerup",e=>{drag=false;canvas.classList.remove("grabbing");
  if(moved<6) clickPick(e.clientX,e.clientY);});
canvas.addEventListener("wheel",e=>{e.preventDefault();
  cam.tDist*=Math.exp(e.deltaY*0.0011); cam.tDist=Math.max(minDist(),Math.min(900,cam.tDist));},{passive:false});

function minDist(){ return cam.follow? cam.follow._r*1.6+0.3 : 20; }

/* body screen positions (updated each frame) */
const bodyScreen=[]; // {body,x,y,depth,rpix}
function hoverPick(mx,my){ let hit=null;
  for(const b of bodyScreen){ if(b.depth<0)continue; const d=Math.hypot(b.x-mx,b.y-my);
    if(d<Math.max(b.rpix+8,16)){hit=b;break;} }
  canvas.classList.toggle("pointing",!!hit);
}
function clickPick(mx,my){ let hit=null,best=1e9;
  for(const b of bodyScreen){ if(b.depth<0)continue; const d=Math.hypot(b.x-mx,b.y-my);
    if(d<Math.max(b.rpix+10,18)&&d<best){best=d;hit=b;} }
  if(hit) select(hit.body); else deselect();
}

/* ============================ selection / UI ============================ */
const infoEl=document.getElementById("info");
function select(b){
  selected=b; cam.follow=b;
  cam.tDist=Math.max(b._r*3.4, b.ring? b.ring.outer*2.2 : b._r*3.4);
  // frame the sunlit face: place the camera on the sun side, at a 3/4 angle
  cam.tTheta=b._ang+Math.PI+0.6; cam.tPhi=1.06;
  document.getElementById("iName").textContent=b.name;
  document.getElementById("iType").textContent=b.sub||(b._moon?("Moon of "+b._moon):"");
  document.getElementById("iDesc").textContent=b.desc||"";
  const g=document.getElementById("iGrid"); g.innerHTML="";
  const facts=b.facts|| (b._moon? {Parent:b._moon}:{});
  for(const k in facts){const c=document.createElement("div");c.className="cell";
    c.innerHTML=`<div class="n">${k}</div><div class="v">${facts[k]}</div>`; g.appendChild(c);}
  infoEl.classList.add("show");
  [...document.querySelectorAll(".chip")].forEach(c=>c.classList.toggle("on",c.dataset.n===b.name));
  buildLabels();
}
function deselect(){ selected=null; cam.follow=null; infoEl.classList.remove("show");
  [...document.querySelectorAll(".chip")].forEach(c=>c.classList.remove("on"));
  cam.tTarget=[0,0,0]; cam.tDist=Math.max(cam.tDist,120); buildLabels(); }
document.getElementById("infoX").onclick=deselect;

/* rail chips */
const rail=document.getElementById("rail");
{ const sun=document.createElement("button"); sun.className="chip"; sun.dataset.n="Sun";
  sun.innerHTML='<b style="background:#ffb454;box-shadow:0 0 6px #ffb454"></b>Sun';
  sun.onclick=()=>focusSun(); rail.appendChild(sun);
  BODIES.forEach(b=>{const c=document.createElement("button");c.className="chip";c.dataset.n=b.name;
    const col=`rgb(${b.colA.map(x=>Math.round(x*255*1.3)).join(",")})`;
    c.innerHTML=`<b style="background:${col};box-shadow:0 0 6px ${col}"></b>${b.name}`;
    c.onclick=()=>select(b); rail.appendChild(c);});
}
function focusSun(){ selected=null;cam.follow={name:"Sun",_r:SUN_R,pos:[0,0,0]};
  cam.tDist=SUN_R*3.2; infoEl.classList.add("show");
  document.getElementById("iName").textContent="The Sun";
  document.getElementById("iType").textContent="G-type main-sequence star";
  document.getElementById("iDesc").textContent="The star at the heart of it all — a vast sphere of plasma holding 99.86% of the system's mass. Its fusion furnace lights and warms every world that circles it.";
  const g=document.getElementById("iGrid");g.innerHTML="";
  const f={Diameter:"1,391,000 km",Surface:"5,500 °C",Core:"15M °C",Age:"4.6B years",Mass:"333,000 Earths"};
  for(const k in f){const c=document.createElement("div");c.className="cell";
    c.innerHTML=`<div class="n">${k}</div><div class="v">${f[k]}</div>`;g.appendChild(c);}
  [...document.querySelectorAll(".chip")].forEach(c=>c.classList.toggle("on",c.dataset.n==="Sun"));
  buildLabels();
}

/* controls */
const speedBtn=document.getElementById("speed");
function setSpeed(){speed=SPEEDS[spIdx];speedBtn.textContent=speed===0?"paused":speed+"×";}
document.getElementById("bFaster").onclick=()=>{spIdx=Math.min(SPEEDS.length-1,spIdx+1);setSpeed();};
document.getElementById("bSlower").onclick=()=>{spIdx=Math.max(0,spIdx-1);setSpeed();};
speedBtn.onclick=()=>{spIdx=(spIdx===0?3:0);setSpeed();};
setSpeed();
const bO=document.getElementById("bOrbits"),bL=document.getElementById("bLabels");
bO.classList.toggle("on",showOrbits); bL.classList.toggle("on",showLabels);
bO.onclick=()=>{showOrbits=!showOrbits;bO.classList.toggle("on",showOrbits);};
bL.onclick=()=>{showLabels=!showLabels;bL.classList.toggle("on",showLabels);document.getElementById("labels").style.display=showLabels?"":"none";};

/* labels DOM */
const labelWrap=document.getElementById("labels"); let labelEls=[];
function buildLabels(){ labelWrap.innerHTML=""; labelEls=[];
  const add=(body,name,cls)=>{const el=document.createElement("div");el.className="lab "+cls;
    el.innerHTML=`<i></i>${name}`; el.onclick=()=>{ if(body.name==="Sun")focusSun(); else select(body);};
    labelWrap.appendChild(el); labelEls.push({el,body});};
  add({name:"Sun",_r:SUN_R,pos:[0,0,0],_sun:true},"Sun","");
  BODIES.forEach(b=>add(b,b.name,""));
  // moons only when their planet is the focus
  if(cam.follow){ const host=BODIES.find(b=>b===cam.follow);
    if(host) MOONS.filter(m=>m.p===host.name).forEach(m=>add(m,m.name,"moon")); }
}
buildLabels();

/* ============================ world positions ============================ */
function bodyPos(b){ const a=simT*b.period*0 + b._ang; return [Math.cos(a)*b.orbit,0,Math.sin(a)*b.orbit]; }
BODIES.forEach((b,i)=>{ b._ang=i*1.7; b._rot=0; b._r=b.r; });
MOONS.forEach((m,i)=>{ m._ang=i*2.1; m._r=m.r; });

/* ============================ draw ============================ */
function setMats(u,model,VP){ const mvp=M4.mul(VP,model);
  gl.uniformMatrix4fv(u.uMVP,false,new Float32Array(mvp));
  gl.uniformMatrix4fv(u.uModel,false,new Float32Array(model));
  gl.uniformMatrix3fv(u.uNorm,false,new Float32Array(M4.norm3(model)));}

let eye=[0,0,1];
function render(dt){
  if(speed>0) simT+=dt*speed*0.06;
  // advance angles
  BODIES.forEach(b=>{ b._ang+=dt*speed*0.06*(1.0/b.period); b._rot+=dt*speed*b.rot; });
  MOONS.forEach(m=>{ m._ang+=dt*speed*0.06*(1.0/m.period)*4.0; });

  // world positions
  BODIES.forEach(b=>{ b.pos=[Math.cos(b._ang)*b.orbit,0,Math.sin(b._ang)*b.orbit]; });
  MOONS.forEach(m=>{ const host=BODIES.find(b=>b.name===m.p);
    m.pos=[host.pos[0]+Math.cos(m._ang)*m.orbit, host.pos[1]+Math.sin(m._ang*0.6)*m.orbit*0.12, host.pos[2]+Math.sin(m._ang)*m.orbit]; });

  // camera target follow
  let tgt=[0,0,0];
  if(cam.follow){ tgt = cam.follow.pos || [0,0,0]; }
  for(let i=0;i<3;i++) cam.target[i]+=(tgt[i]-cam.target[i])*Math.min(1,dt*4);
  cam.dist+=(cam.tDist-cam.dist)*Math.min(1,dt*4);
  // smoothly swing azimuth/elevation toward framed angle (shortest path)
  let da=cam.tTheta-cam.theta; da=Math.atan2(Math.sin(da),Math.cos(da));
  cam.theta+=da*Math.min(1,dt*3.2);
  cam.phi+=(cam.tPhi-cam.phi)*Math.min(1,dt*3.2);
  cam.phi=Math.max(0.08,Math.min(Math.PI-0.08,cam.phi));

  const st=Math.sin(cam.phi),ct=Math.cos(cam.phi);
  eye=[cam.target[0]+cam.dist*st*Math.cos(cam.theta),
       cam.target[1]+cam.dist*ct,
       cam.target[2]+cam.dist*st*Math.sin(cam.theta)];
  const near=Math.max(0.02,cam.dist*0.02), far=cam.dist*4+6000;
  const proj=M4.persp(1.05,W/H,near,far);
  const view=M4.lookAt(eye,cam.target,[0,1,0]);
  const VP=M4.mul(proj,view);
  const viewNoT=M4.lookAt([0,0,0],V3.sub(cam.target,eye),[0,1,0]);
  const skyVP=M4.mul(proj,viewNoT);

  /* ---- scene into HDR fbo ---- */
  gl.bindFramebuffer(gl.FRAMEBUFFER,sceneFBO.f);
  gl.viewport(0,0,W,H); gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
  gl.disable(gl.BLEND); gl.enable(gl.CULL_FACE); gl.cullFace(gl.BACK);

  // sky
  gl.depthMask(false); gl.disable(gl.CULL_FACE);
  gl.useProgram(P.sky.p); gl.bindVertexArray(skyMesh.va);
  gl.uniformMatrix4fv(P.sky.u.uVP,false,new Float32Array(skyVP));
  gl.drawElements(gl.TRIANGLES,skyMesh.count,gl.UNSIGNED_SHORT,0);
  // stars
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE);
  gl.useProgram(P.star.p); gl.bindVertexArray(stars.va);
  gl.uniformMatrix4fv(P.star.u.uVP,false,new Float32Array(skyVP));
  gl.uniform1f(P.star.u.uPx,Math.min(devicePixelRatio||1,2));
  gl.drawArrays(gl.POINTS,0,stars.count);
  gl.disable(gl.BLEND); gl.depthMask(true); gl.enable(gl.CULL_FACE);

  // orbit lines
  if(showOrbits){ gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(P.orbit.p);
    BODIES.forEach(b=>{ gl.bindVertexArray(b._orbit.va);
      gl.uniformMatrix4fv(P.orbit.u.uMVP,false,new Float32Array(VP));
      gl.uniform3f(P.orbit.u.uCol,0.28,0.36,0.55);
      const fade=(selected===b||!selected)?0.5:0.22;
      gl.uniform1f(P.orbit.u.uA,fade);
      gl.drawArrays(gl.LINE_STRIP,0,b._orbit.count); });
    gl.disable(gl.BLEND);
  }

  // sun
  gl.useProgram(P.sun.p); gl.bindVertexArray(meshPlanet.va);
  { const m=M4.scale(M4.ident(),SUN_R);
    setMats(P.sun.u,m,VP);
    gl.uniform1f(P.sun.u.uTime,simT); gl.uniform3fv(P.sun.u.uCam,new Float32Array(eye));
    gl.drawElements(gl.TRIANGLES,meshPlanet.count,gl.UNSIGNED_INT,0);
  }

  // asteroid belts
  gl.useProgram(P.ast.p);
  gl.uniformMatrix4fv(P.ast.u.uVP,false,new Float32Array(VP));
  gl.uniform3f(P.ast.u.uSun,0,0,0);
  gl.bindVertexArray(asteroids.va);
  gl.drawElementsInstanced(gl.TRIANGLES,asteroids.index,gl.UNSIGNED_INT,0,asteroids.count);
  gl.bindVertexArray(kuiper.va);
  gl.drawElementsInstanced(gl.TRIANGLES,kuiper.index,gl.UNSIGNED_INT,0,kuiper.count);

  // planets
  gl.useProgram(P.planet.p);
  gl.uniform3f(P.planet.u.uSun,0,0,0); gl.uniform3fv(P.planet.u.uCam,new Float32Array(eye));
  gl.uniform1f(P.planet.u.uTime,simT);
  BODIES.forEach(b=>{
    let m=M4.trans(M4.ident(),b.pos);
    m=M4.mul(m,M4.rotZ(b.tilt||0));
    m=M4.mul(m,M4.rotY(b._rot));
    m=M4.mul(m,[b._r,0,0,0,0,b._r,0,0,0,0,b._r,0,0,0,0,1]);
    gl.bindVertexArray(meshPlanet.va);
    setMats(P.planet.u,m,VP);
    gl.uniform1i(P.planet.u.uType,b.type);
    gl.uniform1f(P.planet.u.uSeed,b._ang*0.0+b.type*3.1);
    gl.uniform3fv(P.planet.u.uColA,new Float32Array(b.colA));
    gl.uniform3fv(P.planet.u.uColB,new Float32Array(b.colB||b.colA));
    gl.uniform3fv(P.planet.u.uColC,new Float32Array(b.colC||b.colB||b.colA));
    gl.drawElements(gl.TRIANGLES,meshPlanet.count,gl.UNSIGNED_INT,0);
  });

  // moons
  MOONS.forEach(m=>{
    let mm=M4.trans(M4.ident(),m.pos);
    mm=M4.mul(mm,M4.rotY(simT*2.0));
    mm=M4.mul(mm,[m._r,0,0,0,0,m._r,0,0,0,0,m._r,0,0,0,0,1]);
    gl.bindVertexArray(meshMoon.va);
    setMats(P.planet.u,mm,VP);
    gl.uniform1i(P.planet.u.uType,9);
    gl.uniform1f(P.planet.u.uSeed,m._ang*0.0+m.name.length*2.0);
    gl.uniform3fv(P.planet.u.uColA,new Float32Array(m.colA));
    gl.uniform3fv(P.planet.u.uColB,new Float32Array([(m.sheen?m.sheen[0]:0.0),(m.sheen?m.sheen[1]:0.0),0]));
    gl.uniform3fv(P.planet.u.uColC,new Float32Array(m.ic||m.colA));
    gl.drawElements(gl.TRIANGLES,meshMoon.count,gl.UNSIGNED_INT,0);
  });

  // rings (transparent, after opaque)
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA); gl.disable(gl.CULL_FACE); gl.depthMask(false);
  gl.useProgram(P.ring.p);
  BODIES.forEach(b=>{ if(!b._ring)return;
    let m=M4.trans(M4.ident(),b.pos); m=M4.mul(m,M4.rotZ(b.tilt||0));
    setMats(P.ring.u,m,VP);
    gl.uniform3f(P.ring.u.uSun,0,0,0); gl.uniform3fv(P.ring.u.uCam,new Float32Array(eye));
    gl.uniform3fv(P.ring.u.uColA,new Float32Array(b.ring.colA));
    gl.uniform3fv(P.ring.u.uColB,new Float32Array(b.ring.colB));
    gl.uniform3fv(P.ring.u.uPlanet,new Float32Array(b.pos));
    gl.uniform1f(P.ring.u.uPr,b._r);
    gl.bindVertexArray(b._ring.va); gl.drawElements(gl.TRIANGLES,b._ring.count,gl.UNSIGNED_INT,0);
  });
  gl.depthMask(true);

  // atmospheres (additive shell)
  gl.blendFunc(gl.SRC_ALPHA,gl.ONE); gl.enable(gl.CULL_FACE); gl.cullFace(gl.FRONT); gl.depthMask(false);
  gl.useProgram(P.atm.p);
  BODIES.forEach(b=>{ if(!b.atm)return;
    const s=b._r*1.14; let m=M4.trans(M4.ident(),b.pos); m=M4.mul(m,[s,0,0,0,0,s,0,0,0,0,s,0,0,0,0,1]);
    gl.bindVertexArray(meshPlanet.va); setMats(P.atm.u,m,VP);
    gl.uniform3f(P.atm.u.uSun,0,0,0); gl.uniform3fv(P.atm.u.uCam,new Float32Array(eye));
    gl.uniform3fv(P.atm.u.uColA,new Float32Array([b.atm[0],b.atm[1],b.atm[2]]));
    gl.uniform1f(P.atm.u.uPow,b.atm[3]); gl.uniform1f(P.atm.u.uStr,b.atm[4]);
    gl.drawElements(gl.TRIANGLES,meshPlanet.count,gl.UNSIGNED_INT,0);
  });
  gl.cullFace(gl.BACK); gl.depthMask(true); gl.disable(gl.BLEND);

  /* ---- bloom ---- */
  gl.disable(gl.DEPTH_TEST); gl.bindVertexArray(quad);
  gl.bindFramebuffer(gl.FRAMEBUFFER,briFBO.f); gl.viewport(0,0,BW,BH);
  gl.useProgram(P.bright.p); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,sceneTex);
  gl.uniform1i(P.bright.u.uTex,0); gl.drawArrays(gl.TRIANGLES,0,3);
  // blur ping-pong
  let src=briTex,passes=[[b1FBO,b1Tex,[1/BW,0]],[b2FBO,b2Tex,[0,1/BH]],[b1FBO,b1Tex,[1/BW,0]],[b2FBO,b2Tex,[0,1/BH]]];
  gl.useProgram(P.blur.p);
  for(const [f,t,dir] of passes){ gl.bindFramebuffer(gl.FRAMEBUFFER,f.f); gl.viewport(0,0,BW,BH);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,src); gl.uniform1i(P.blur.u.uTex,0);
    gl.uniform2f(P.blur.u.uDir,dir[0],dir[1]); gl.drawArrays(gl.TRIANGLES,0,3); src=t; }

  /* ---- composite to screen ---- */
  gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.viewport(0,0,W,H);
  gl.useProgram(P.comp.p);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,sceneTex); gl.uniform1i(P.comp.u.uScene,0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D,src); gl.uniform1i(P.comp.u.uBloom,1);
  gl.uniform2f(P.comp.u.uPx,1/W,1/H);
  gl.drawArrays(gl.TRIANGLES,0,3);

  /* ---- labels ---- */
  bodyScreen.length=0;
  const project=(p)=>{const c=M4.mul(VP,M4.ident()); // p is world
    const x=VP[0]*p[0]+VP[4]*p[1]+VP[8]*p[2]+VP[12];
    const y=VP[1]*p[0]+VP[5]*p[1]+VP[9]*p[2]+VP[13];
    const z=VP[2]*p[0]+VP[6]*p[1]+VP[10]*p[2]+VP[14];
    const w=VP[3]*p[0]+VP[7]*p[1]+VP[11]*p[2]+VP[15];
    return [x,y,z,w];};
  const rpixOf=(p,r)=>{ const d=V3.len(V3.sub(p,eye)); const f=1/Math.tan(1.05/2);
    return (r/Math.max(d,0.001))*f*(H/2); };
  // register clickable screen positions for all bodies
  const regs=[{body:{name:"Sun",_r:SUN_R,pos:[0,0,0],_sun:true},pos:[0,0,0],r:SUN_R}];
  BODIES.forEach(b=>regs.push({body:b,pos:b.pos,r:b._r}));
  if(cam.follow){const host=BODIES.find(b=>b===cam.follow);
    if(host)MOONS.filter(m=>m.p===host.name).forEach(m=>regs.push({body:m,pos:m.pos,r:m._r}));}
  regs.forEach(rg=>{const c=project(rg.pos);
    if(c[3]<=0){bodyScreen.push({body:rg.body,depth:-1});return;}
    const sx=(c[0]/c[3]*0.5+0.5)*innerWidth, sy=(-c[1]/c[3]*0.5+0.5)*innerHeight;
    bodyScreen.push({body:rg.body,x:sx,y:sy,depth:c[2]/c[3],rpix:rpixOf(rg.pos,rg.r)});});

  if(showLabels){ labelEls.forEach(le=>{ const b=le.body; const pos=b.pos||[0,0,0];
    const c=project(pos); if(c[3]<=0){le.el.style.opacity=0;return;}
    const sx=(c[0]/c[3]*0.5+0.5)*innerWidth, sy=(-c[1]/c[3]*0.5+0.5)*innerHeight;
    const rp=rpixOf(pos,b._r||1);
    const distE=V3.len(V3.sub(pos,eye));
    const isFocus = selected===b || cam.follow===b || (b._sun&&cam.follow&&cam.follow.name==="Sun");
    const isMoon = le.el.classList.contains("moon");
    let op=1;
    if(rp>innerHeight*0.55) op=0;                       // too close / huge
    else if(!isFocus && !isMoon && distE>cam.dist*2.6) op=0;  // far background clutter
    else if(rp<1.2 && !isFocus) op=0;                   // sub-pixel
    le.el.style.left=sx+"px"; le.el.style.top=(sy - Math.max(rp,6) - 12)+"px";
    le.el.style.opacity=op; le.el.classList.toggle("sel",isFocus);
  }); }
}

/* ============================ loop ============================ */
let last=performance.now(), booted=false;
function frame(now){ const dt=Math.min(0.05,(now-last)/1000); last=now;
  try{ render(dt); }catch(err){ console.error(err);
    document.getElementById("err").style.display="flex";
    document.getElementById("err").textContent="Runtime error: "+err.message; return; }
  if(!booted){ booted=true; const b=document.getElementById("boot"); b.classList.add("gone"); setTimeout(()=>b.style.display="none",800);
    setTimeout(()=>focusSun(),300); }
  requestAnimationFrame(frame);
}
const boot=document.getElementById("boot");
requestAnimationFrame(frame);

/* expose a tiny hook for headless verification */
window.__orrery={gl,glError:()=>gl.getError(),bodies:BODIES.length};
})();
