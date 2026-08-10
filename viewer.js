/*
  The whole web viewer.

  model-viewer stays in the page, but only to launch AR -- it lights everything
  with a uniform environment, which is why the building looked like flat card.
  Everything you actually see on screen is rendered here in three.js, with a
  sun that casts, a bounce light, and lamps inside the building.

  Nothing is baked into the .glb files (theatre.py exports with bake=False), so
  all shading is live and every knob below does something visible.
*/
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const INK = 0x101820;

const VIEWS = {
  out:  { file: 'theatre.glb',      az:  38, el: 20, pad: 1.30, ground: true },
  hill: { file: 'theatre_hill.glb', az: 205, el: 17, pad: 1.20, ground: false },
  in:   { file: 'theatre_cut.glb',  az:  45, el: 26, pad: 1.05, ground: false }
};

export function start(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(INK);

  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 600);

  // --- light ------------------------------------------------------------
  scene.add(new THREE.HemisphereLight(0xc3d6e4, 0x33352c, 0.85));

  const sun = new THREE.DirectionalLight(0xfff2e0, 2.4);
  sun.position.set(-20, 30, 24);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.03;
  const sc = sun.shadow.camera;
  sc.left = -30; sc.right = 30; sc.top = 30; sc.bottom = -30; sc.near = 1; sc.far = 110;
  sc.updateProjectionMatrix();
  scene.add(sun);

  const bounce = new THREE.DirectionalLight(0x93aabc, 0.45);
  bounce.position.set(22, 10, -24);
  scene.add(bounce);

  // daylight down the skylight onto the dissecting floor, and a lamp below
  const skylight = new THREE.PointLight(0xfff4de, 30, 30, 2);
  skylight.position.set(0, 6.8, 0);
  const lamp = new THREE.PointLight(0xffd9a6, 12, 17, 2);
  lamp.position.set(-1, -2.4, -1);
  scene.add(skylight, lamp);

  // catches the sun's shadow under the flat exterior model
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(160, 160),
    new THREE.ShadowMaterial({ opacity: 0.42 }));
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  ground.visible = false;
  scene.add(ground);

  // --- controls ---------------------------------------------------------
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.075;
  controls.minDistance = 4;
  controls.maxDistance = 140;
  controls.maxPolarAngle = Math.PI * 0.502;   // never go under the ground

  // --- models, loaded once each and kept ---------------------------------
  const loader = new GLTFLoader();
  const cache = {};
  let current = null, active = false, raf = null;

  function dress(root) {
    root.traverse((o) => {
      if (!o.isMesh) return;
      o.castShadow = true;
      o.receiveShadow = true;
      const m = o.material;
      if (!m) return;
      m.roughness = m.transparent ? 0.25 : 0.94;
      m.metalness = 0;
      if (m.map) m.map.anisotropy = renderer.capabilities.getMaxAnisotropy();
    });
  }

  function frameTo(obj, v) {
    const box = new THREE.Box3().setFromObject(obj);
    const c = box.getCenter(new THREE.Vector3());
    const r = box.getSize(new THREE.Vector3()).length() / 2;
    const d = (r * v.pad) / Math.sin(THREE.MathUtils.degToRad(camera.fov) / 2);
    const a = THREE.MathUtils.degToRad(v.az), e = THREE.MathUtils.degToRad(v.el);
    camera.position.set(c.x + d * Math.cos(e) * Math.sin(a),
                        c.y + d * Math.sin(e),
                        c.z + d * Math.cos(e) * Math.cos(a));
    controls.target.copy(c);
    controls.update();
  }

  function show(name, onReady) {
    const v = VIEWS[name];
    const apply = (obj) => {
      if (current && current !== obj) current.visible = false;
      obj.visible = true;
      current = obj;
      ground.visible = v.ground;
      const inside = name === 'in';
      skylight.visible = inside;
      lamp.visible = inside;
      frameTo(obj, v);
      if (onReady) onReady();
    };
    if (cache[name]) return apply(cache[name]);
    loader.load(v.file, (gltf) => {
      dress(gltf.scene);
      scene.add(gltf.scene);
      cache[name] = gltf.scene;
      apply(gltf.scene);
    });
  }

  function resize() {
    const w = innerWidth, h = innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }
  addEventListener('resize', resize);
  resize();

  function tick() {
    raf = requestAnimationFrame(tick);
    if (!active) return;
    controls.update();
    renderer.render(scene, camera);
  }

  return {
    show,
    setActive(on) { active = on; if (on && raf === null) tick(); }
  };
}
