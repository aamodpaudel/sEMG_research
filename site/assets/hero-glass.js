/* Hero: the φ as a glass object. The real muscle-activity envelope (drawn by site.js into #emg-bg)
   runs behind it and bends through the glass: the signal passing through. Falls back to the flat
   SVG mark when WebGL or the CDN is unavailable. */
import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const host = document.getElementById("hero-3d");
const hero = document.getElementById("top");
const emg = document.getElementById("emg-bg");
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

function start() {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  } catch (e) {
    return;
  }
  const small = () => innerWidth < 700;
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, small() ? 1.5 : 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#000000");
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 200);

  // The mark on its 64 x 96 grid, 1 unit = 10 px: ring radius 22, stroke 9, bar from y 8 to 88.
  const glass = (tint, distance) =>
    new THREE.MeshPhysicalMaterial({
      color: 0xffffff, transmission: 1, thickness: 1.1, roughness: 0.05, ior: 1.5, dispersion: 0.35,
      attenuationColor: new THREE.Color(tint), attenuationDistance: distance,
      clearcoat: 1, clearcoatRoughness: 0.04, envMapIntensity: 1.15, specularIntensity: 1,
    });
  const phi = new THREE.Group();
  const ring = new THREE.Mesh(new THREE.TorusGeometry(2.2, 0.45, 64, 220), glass("#5DB5AE", 1.6));
  const bar = new THREE.Mesh(new THREE.CapsuleGeometry(0.45, 7.1, 16, 48), glass("#F0794A", 0.75));
  bar.position.z = 0.12;
  phi.add(ring, bar);
  scene.add(phi);

  // Behind the glass, redrawn every frame on an opaque canvas (the transmission pass ignores
  // transparent objects): a soft teal light, a warm streak behind the bar, then the live signal.
  const back = document.createElement("canvas");
  const bctx = back.getContext("2d");
  const backTex = new THREE.CanvasTexture(back);
  backTex.colorSpace = THREE.SRGBColorSpace;
  const backdrop = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: backTex, toneMapped: false }));
  backdrop.position.z = -3;
  scene.add(backdrop);

  function paintBack() {
    const W = back.width, H = back.height, cx = W / 2, cy = H / 2, r = Math.min(W, H) * 0.62;
    bctx.globalCompositeOperation = "source-over";
    bctx.fillStyle = "#000";
    bctx.fillRect(0, 0, W, H);
    let g = bctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    g.addColorStop(0, "rgb(52, 128, 124)");
    g.addColorStop(0.35, "rgb(20, 62, 64)");
    g.addColorStop(1, "rgb(0, 0, 0)");
    bctx.fillStyle = g;
    bctx.fillRect(0, 0, W, H);
    bctx.save();
    bctx.translate(cx, cy);
    bctx.scale(0.16, 1);
    g = bctx.createRadialGradient(0, 0, 0, 0, 0, r * 0.95);
    g.addColorStop(0, "rgba(240, 121, 74, 0.55)");
    g.addColorStop(1, "rgba(240, 121, 74, 0)");
    bctx.fillStyle = g;
    bctx.fillRect(-r, -r, 2 * r, 2 * r);
    bctx.restore();
    if (emg && emg.width > 1) {
      bctx.globalCompositeOperation = "lighter";
      const eh = (emg.height / emg.width) * W;
      bctx.drawImage(emg, 0, cy - eh / 2, W, eh);
    }
    backTex.needsUpdate = true;
  }

  // The mark fills most of its block, centred.
  function layout() {
    const w = host.clientWidth, h = host.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    const share = w / h < 0.8 ? 0.7 : 0.78;
    const viewH = 8.9 / share;
    camera.position.z = viewH / (2 * Math.tan(THREE.MathUtils.degToRad(15)));
    camera.updateProjectionMatrix();
    const bh = 2 * (camera.position.z - backdrop.position.z) * Math.tan(THREE.MathUtils.degToRad(15));
    backdrop.scale.set(bh * camera.aspect, bh, 1);
    back.width = Math.max(2, Math.round(w));
    back.height = Math.max(2, Math.round(h));
  }
  layout();
  addEventListener("resize", layout);

  const pointer = { x: 0, y: 0 };
  addEventListener("pointermove", (e) => {
    pointer.x = (e.clientX / innerWidth) * 2 - 1;
    pointer.y = (e.clientY / innerHeight) * 2 - 1;
  }, { passive: true });

  let visible = true;
  new IntersectionObserver((es) => (visible = es[0].isIntersecting)).observe(host);

  const t0 = performance.now();
  let last = t0;
  const ease = (k) => 1 - Math.pow(1 - k, 4);
  function frame(now) {
    const t = (now - t0) / 1000;
    const follow = 1 - Math.pow(0.94, Math.min(0.1, (now - last) / 1000) * 60);   // same feel at any frame rate
    last = now;
    if (visible && !document.hidden) {
      const k = reduce ? 1 : ease(Math.min(1, t / 2.2));
      const sway = reduce ? 0.35 : Math.sin(t * 0.32) * 0.42 + pointer.x * 0.28;
      phi.rotation.y += ((-1.4 * (1 - k) + sway) - phi.rotation.y) * (reduce ? 1 : follow);
      phi.rotation.x += ((reduce ? 0 : pointer.y * 0.12) - phi.rotation.x) * follow;
      phi.scale.setScalar(0.86 + 0.14 * k);
      phi.position.y = reduce ? 0 : Math.sin(t * 0.7) * 0.06;
      paintBack();

      // Scrolling away: the mark lifts and the hero fades to black.
      const p = Math.min(1, Math.max(0, scrollY / (hero.offsetHeight * 0.8)));
      host.style.opacity = String(1 - p * 0.85);
      phi.position.y += p * 1.5;
      renderer.render(scene, camera);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  host.classList.add("live");
}

if (host && hero) start();
