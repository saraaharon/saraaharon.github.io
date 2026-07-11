import * as THREE from "three";

// Container
const container = document.getElementById("earth-container");

// Scene
const scene = new THREE.Scene();

// Camera
const camera = new THREE.PerspectiveCamera(
    45,
    container.clientWidth / container.clientHeight,
    0.1,
    1000
);

camera.position.z = 3;

// Renderer
const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true
});

renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);

container.appendChild(renderer.domElement);

// Texture Loader
const loader = new THREE.TextureLoader();

const earthTexture = loader.load("assets/images/earth_day.jpg");

// Geometry
const geometry = new THREE.SphereGeometry(1, 64, 64);

// Material
const material = new THREE.MeshStandardMaterial({
    map: earthTexture
});

// Earth
const earth = new THREE.Mesh(geometry, material);

scene.add(earth);

// Lights

const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xffffff, 2);

sunLight.position.set(5, 3, 5);

scene.add(sunLight);

// Animation

function animate() {

    requestAnimationFrame(animate);

    earth.rotation.y += 0.0015;

    renderer.render(scene, camera);

}

animate();

// Resize

window.addEventListener("resize", () => {

    camera.aspect =
        container.clientWidth / container.clientHeight;

    camera.updateProjectionMatrix();

    renderer.setSize(
        container.clientWidth,
        container.clientHeight
    );

});