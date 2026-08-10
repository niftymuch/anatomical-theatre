/*
  Interior view.

  The exterior runs on model-viewer, which gives us AR on both platforms but
  lights everything with a uniform environment -- nothing casts onto anything,
  so the inside of the building reads as flat coloured card.  This view loads
  the same theatre_cut.glb into plain three.js so the sun can throw the tiers
  onto each other, the stair against the wall behind it, and a warm light can
  sit under the skylight where one actually would have.

  Nothing is baked into theatre_cut.glb -- theatre.py exports it with bake=False
  -- so all of the shading you see here is computed live.
*/
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const INK = 0x101820;

export function start(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(INK);
  scene.fog = new THREE.Fog(INK, 60, 190);

  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 400);
  camera.position.set(15, 11, 16);

  // --- light ------------------------------------------------------------
  scene.add(new THREE.HemisphereLight(0xbfd3e0, 0x2a2b26, 0.55));

  const sun = new THREE.DirectionalLight(0xfff1dc, 2.1);
  sun.position.set(-19, 29, 23);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0006;
  sun.shadow.normalBias = 0.02;
  Object.assign(sun.shadow.camera,
    { left: -26, right: 26, top: 26, bottom: -26, near: 1, far: 90 });
  sun.shadow.camera.updateProjectionMatrix();
  scene.add(sun);

  const bounce = new THREE.DirectionalLight(0x8fa6b8, 0.32);
  bounce.position.set(18, 9, -21);
  scene.add(bounce);

  // daylight dropping through the skylight onto the dissecting floor
  const skylight = new THREE.PointLight(0xfff4de, 26, 26, 2);
  skylight.position.set(0, 6.6, 0);
  scene.add(skylight);

  // the charnel had no daylight to speak of
  const lamp = new THREE.PointLight(0xffd9a6, 9, 15, 2);
  lamp.position.set(-2, -2.6, -2);
  scene.add(lamp);

  // --- controls ---------------------------------------------------------
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 2.2, 0);
  controls.minDistance = 4;
  controls.maxDistance = 70;
  controls.maxPolarAngle = Math.PI * 0.52;
  controls.update();

  // --- model ------------------------------------------------------------
  new GLTFLoader().load('theatre_cut.glb', (gltf) => {
    gltf.scene.traverse((o) => {
      if (!o.isMesh) return;
      o.castShadow = true;
      o.receiveShadow = true;
      const m = o.material;
      if (m) {
        m.roughness = m.transparent ? 0.25 : 0.92;
        m.metalness = 0;
        m.envMapIntensity = 0.5;
        if (m.map) m.map.anisotropy = renderer.capabilities.getMaxAnisotropy();
      }
    });
    scene.add(gltf.scene);
  });

  // --- loop -------------------------------------------------------------
  let active = false, raf = null;

  function resize() {
    const w = innerWidth, h = innerHeight;
    if (canvas.width === w * renderer.getPixelRatio() && canvas.height === h * renderer.getPixelRatio()) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }
  addEventListener('resize', resize);

  function tick() {
    raf = requestAnimationFrame(tick);
    if (!active) return;
    resize();
    controls.update();
    renderer.render(scene, camera);
  }

  return {
    setActive(on) {
      active = on;
      if (on) { resize(); if (raf === null) tick(); }
    },
    resume() { resize(); if (raf === null) tick(); }
  };
}
