import * as THREE from "https://esm.sh/three@0.150.1";
import { OrbitControls } from "https://esm.sh/three@0.150.1/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "https://esm.sh/three@0.150.1/examples/jsm/loaders/GLTFLoader.js";
import { PLYLoader } from "https://esm.sh/three@0.150.1/examples/jsm/loaders/PLYLoader.js";
import { OBJLoader } from "https://esm.sh/three@0.150.1/examples/jsm/loaders/OBJLoader.js";
import { MTLLoader } from "https://esm.sh/three@0.150.1/examples/jsm/loaders/MTLLoader.js";
import { Line2 } from "https://esm.sh/three@0.150.1/examples/jsm/lines/Line2.js";
import { LineMaterial } from "https://esm.sh/three@0.150.1/examples/jsm/lines/LineMaterial.js";
import { LineSegmentsGeometry } from "https://esm.sh/three@0.150.1/examples/jsm/lines/LineSegmentsGeometry.js";

// Viewer configuration with mesh options
const meshOptions = [
  { value: "2dts.glb", label: "2DTS (Ours)" },
  { value: "2dgs.ply", label: "2DGS" },
  { value: "nvdiffrec.obj", label: "Nvdiffrec" }
];

const viewers = {
  ship: {
    scene: new THREE.Scene(),
    camera: new THREE.PerspectiveCamera(60, 1, 0.1, 1000),
    cameraPosition: new THREE.Vector3(0, -2, 1.0),
    cameraTarget: new THREE.Vector3(0, 0, -0.5),
    cameraUp: new THREE.Vector3(0, 0, 1),
    autoRotate: true,
    autoRotateSpeed: 0.01,
    rotationAxis: new THREE.Vector3(0, 0, 1),
  },
  ficus: {
    scene: new THREE.Scene(),
    camera: new THREE.PerspectiveCamera(60, 1, 0.1, 1000),
    cameraPosition: new THREE.Vector3(0, -2, 1.0),
    cameraTarget: new THREE.Vector3(0, 0, 0.2),
    cameraUp: new THREE.Vector3(0, 0, 1),
    autoRotate: true,
    autoRotateSpeed: 0.01,
    rotationAxis: new THREE.Vector3(0, 0, 1),
  }
};

// Function to create controls for a viewer
function createViewerControls(sceneName) {
  const viewerSection = document.querySelector(`[data-scene="${sceneName}"]`);
  const controlsContainer = viewerSection.querySelector('.viewer-controls');
  const viewerContainer = viewerSection.querySelector('.viewer');
  
  // Set the viewer element ID
  viewerContainer.id = `viewer-${sceneName}`;
  
  // Create mesh selector
  const meshSelector = document.createElement('select');
  meshSelector.id = `meshSelector-${sceneName}`;
  meshOptions.forEach(option => {
    const optionElement = document.createElement('option');
    optionElement.value = option.value;
    optionElement.textContent = option.label;
    meshSelector.appendChild(optionElement);
  });
  
  // Create wireframe toggle
  const wireframeLabel = document.createElement('label');
  const wireframeCheckbox = document.createElement('input');
  wireframeCheckbox.type = 'checkbox';
  wireframeCheckbox.id = `wireframeToggle-${sceneName}`;
  wireframeLabel.appendChild(wireframeCheckbox);
  wireframeLabel.appendChild(document.createTextNode(' Wireframe'));
  
  // Create auto-rotation toggle
  const autoRotateLabel = document.createElement('label');
  const autoRotateCheckbox = document.createElement('input');
  autoRotateCheckbox.type = 'checkbox';
  autoRotateCheckbox.id = `autoRotateToggle-${sceneName}`;
  autoRotateCheckbox.checked = true;
  autoRotateLabel.appendChild(autoRotateCheckbox);
  autoRotateLabel.appendChild(document.createTextNode(' Auto Rotate'));
  
  // Create backface culling toggle
  const backfaceCullingLabel = document.createElement('label');
  const backfaceCullingCheckbox = document.createElement('input');
  backfaceCullingCheckbox.type = 'checkbox';
  backfaceCullingCheckbox.id = `backfaceCullingToggle-${sceneName}`;
  backfaceCullingLabel.appendChild(backfaceCullingCheckbox);
  backfaceCullingLabel.appendChild(document.createTextNode(' Backface Culling'));
  
  // Create face count display
  const faceCountDisplay = document.createElement('span');
  faceCountDisplay.id = `faceCount-${sceneName}`;
  faceCountDisplay.className = 'face-count';
  faceCountDisplay.textContent = 'Faces: -';
  
  // Create ambient light intensity slider
  const lightLabel = document.createElement('label');
  lightLabel.textContent = 'Light: ';
  lightLabel.className = 'slider-label';
  
  const lightSlider = document.createElement('input');
  lightSlider.type = 'range';
  lightSlider.id = `lightSlider-${sceneName}`;
  lightSlider.min = '0';
  lightSlider.max = '3';
  lightSlider.step = '0.1';
  lightSlider.value = '1.0';
  lightSlider.className = 'light-slider';
  
  const lightValue = document.createElement('span');
  lightValue.id = `lightValue-${sceneName}`;
  lightValue.className = 'slider-value';
  lightValue.textContent = '1.0';
  
  lightLabel.appendChild(lightSlider);
  lightLabel.appendChild(lightValue);
  
  // Add controls to container
  controlsContainer.appendChild(meshSelector);
  controlsContainer.appendChild(wireframeLabel);
  controlsContainer.appendChild(autoRotateLabel);
  controlsContainer.appendChild(backfaceCullingLabel);
  controlsContainer.appendChild(lightLabel);
  controlsContainer.appendChild(faceCountDisplay);
  
  // Store references in viewer config
  viewers[sceneName].element = viewerContainer;
  viewers[sceneName].meshSelector = meshSelector;
  viewers[sceneName].wireframeToggle = wireframeCheckbox;
  viewers[sceneName].autoRotateToggle = autoRotateCheckbox;
  viewers[sceneName].backfaceCullingToggle = backfaceCullingCheckbox;
  viewers[sceneName].lightSlider = lightSlider;
  viewers[sceneName].lightValue = lightValue;
  viewers[sceneName].faceCountDisplay = faceCountDisplay;
  
  // Add event listeners
  meshSelector.addEventListener('change', (e) => {
    loadMeshForViewer(sceneName, e.target.value);
  });
  
  wireframeCheckbox.addEventListener('change', (e) => {
    toggleWireframeForViewer(sceneName, e.target.checked);
  });
  
  autoRotateCheckbox.addEventListener('change', (e) => {
    viewers[sceneName].autoRotate = e.target.checked;
    if (!e.target.checked) {
      viewers[sceneName].isAutoRotating = false; // Stop auto-rotation immediately
    }
  });
  
  backfaceCullingCheckbox.addEventListener('change', (e) => {
    toggleBackfaceCullingForViewer(sceneName, e.target.checked);
  });
  
  lightSlider.addEventListener('input', (e) => {
    const intensity = parseFloat(e.target.value);
    lightValue.textContent = intensity.toFixed(1);
    if (viewers[sceneName].ambientLight) {
      viewers[sceneName].ambientLight.intensity = intensity;
    }
  });
}

function setupViewer(sceneName) {
  const viewer = viewers[sceneName];
  
  // Setup camera
  viewer.camera.position.set(viewer.cameraPosition.x, viewer.cameraPosition.y, viewer.cameraPosition.z);
  viewer.camera.up.copy(viewer.cameraUp); // Set the camera's up vector
  
  // Setup renderer
  const viewerWidth = viewer.element.clientWidth;
  const viewerHeight = viewer.element.clientHeight;
  viewer.camera.aspect = viewerWidth / viewerHeight;
  viewer.camera.updateProjectionMatrix();

  viewer.renderer = new THREE.WebGLRenderer({ antialias: true });
  viewer.renderer.setSize(viewerWidth, viewerHeight);
  viewer.renderer.setClearColor(0x000000, 1.0);
  viewer.element.appendChild(viewer.renderer.domElement);
  
  // Setup controls
  viewer.controls = new OrbitControls(viewer.camera, viewer.renderer.domElement);
  viewer.controls.target.set(viewer.cameraTarget.x, viewer.cameraTarget.y, viewer.cameraTarget.z);
  viewer.controls.update();
  
  // Add interaction listeners to detect when user is controlling the camera
  viewer.controls.addEventListener('start', () => {
    viewer.lastInteraction = Date.now();
    viewer.isAutoRotating = false; // User started interacting, stop auto-rotation
  });
  
  viewer.controls.addEventListener('change', () => {
    // Only update lastInteraction if we're not in auto-rotation mode
    if (!viewer.isAutoRotating) {
      viewer.lastInteraction = Date.now();
    }
  });
  
  viewer.controls.addEventListener('end', () => {
    viewer.lastInteraction = Date.now();
    viewer.isAutoRotating = false; // User finished interacting
  });
  
  // Setup lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
  viewer.scene.add(ambientLight);
  viewer.ambientLight = ambientLight; // Store reference for intensity control

  // initialize auto-rotation state
  viewer.isAutoRotating = false;
  viewer.lastInteraction = Date.now() - 3000
}

function updateFaceCount(sceneName, geometry) {
  const viewer = viewers[sceneName];
  let faceCount = 0;
  
  if (geometry.index) {
    // Indexed geometry - face count is indices divided by 3
    faceCount = geometry.index.count / 3;
  } else {
    // Non-indexed geometry - face count is positions divided by 9 (3 vertices * 3 coordinates)
    faceCount = geometry.attributes.position.count / 3;
  }
  
  viewer.faceCountDisplay.textContent = `Faces: ${faceCount.toLocaleString()}`;
}

// Initialize viewers
Object.keys(viewers).forEach(sceneName => {
  createViewerControls(sceneName);
  setupViewer(sceneName);
});

function loadMeshFromFile(url) {
  return new Promise((resolve, reject) => {
    if (url.endsWith(".glb")) {
      const loader = new GLTFLoader();
      loader.load(
        url, 
        (gltf) => {
          const geometry = gltf.scene.children[0].children[0].geometry;
          const material = new THREE.MeshStandardMaterial({
            vertexColors: true,
            flatShading: true,
            metalness: 0.0,
            roughness: 1.0,
          });
          const mesh = new THREE.Mesh(geometry, material);
          console.log("GLB Mesh:", mesh);
          resolve(mesh);
        },
        undefined,
        (error) => reject(error)
      );
    } else if (url.endsWith(".ply")) {
      const loader = new PLYLoader();
      loader.load(
        url,
        (geometry) => {
          // const colors = geometry.getAttribute("color");
          // if (colors) {
          //   for (let i = 0; i < colors.array.length; i++) {
          //     colors.array[i] = colors.array[i] * 1;
          //   }
          // }
          const material = new THREE.MeshStandardMaterial({
            vertexColors: true,
            flatShading: true,
            metalness: 0.0,
            roughness: 1.0,
          });
          const mesh = new THREE.Mesh(geometry, material);
          console.log("PLY Mesh:", mesh);
          resolve(mesh);
        },
        undefined,
        (error) => reject(error)
      );
    } else if (url.endsWith(".obj")) {
      const basePath = url.substring(0, url.lastIndexOf("/") + 1);
      const fileName = url.substring(
        url.lastIndexOf("/") + 1,
        url.lastIndexOf(".")
      );

      const mtlLoader = new MTLLoader();
      mtlLoader.setPath(basePath);

      mtlLoader.load(fileName + ".mtl", (materials) => {
        materials.preload();

        const objLoader = new OBJLoader();
        objLoader.setMaterials(materials);
        objLoader.load(
          url, 
          (object) => {
            const mesh = object.children[0];
            const rotationMatrix = new THREE.Matrix4().makeRotationX(Math.PI / 2);
            mesh.geometry.applyMatrix4(rotationMatrix);
            mesh.geometry.computeBoundingBox();
            mesh.geometry.computeBoundingSphere();
            console.log("OBJ Mesh:", mesh);
            resolve(mesh);
          },
          undefined,
          (error) => reject(error)
        );
      });
    } else {
      reject(new Error("Unsupported file format"));
    }
  });
}

async function loadMeshForViewer(sceneName, selectedMesh) {
  const viewer = viewers[sceneName];
  const meshUrl = `assets/${sceneName}/${selectedMesh}`;

  if (viewer.currentMesh) viewer.scene.remove(viewer.currentMesh);
  if (viewer.wireframeMesh) {
    viewer.scene.remove(viewer.wireframeMesh);
    viewer.wireframeMesh = null;
  }
  
  try {
    viewer.currentMesh = await loadMeshFromFile(meshUrl);
    viewer.scene.add(viewer.currentMesh);
    
    // Update face count display
    updateFaceCount(sceneName, viewer.currentMesh.geometry);
    
    // Apply current toggle states
    toggleWireframeForViewer(sceneName, viewer.wireframeToggle.checked);
    toggleBackfaceCullingForViewer(sceneName, viewer.backfaceCullingToggle.checked);
  } catch (error) {
    console.error(`Failed to load mesh for ${sceneName}:`, error);
    viewer.faceCountDisplay.textContent = 'Faces: Error';
  }
}

async function loadMeshForAllViewers() {
  // Load mesh for each viewer with their individual selections
  await Promise.all(
    Object.keys(viewers).map(sceneName => 
      loadMeshForViewer(sceneName, viewers[sceneName].meshSelector.value)
    )
  );
}

function toggleWireframeForViewer(sceneName, enabled) {
  const viewer = viewers[sceneName];
  
  if (!viewer.currentMesh) return;
  
  if (!enabled) {
    if (viewer.wireframeMesh) {
      viewer.scene.remove(viewer.wireframeMesh);
    }
    viewer.scene.add(viewer.currentMesh);
    return;
  } else {
    if (!viewer.wireframeMesh) {
      const edgesGeo = new THREE.EdgesGeometry(viewer.currentMesh.geometry);
      const lineGeo = new LineSegmentsGeometry();
      lineGeo.fromEdgesGeometry(edgesGeo);
      
      const lineMat = new LineMaterial({
        color: 0xffffff,
        linewidth: 0.5,
        resolution: new THREE.Vector2(
          viewer.renderer.domElement.width, 
          viewer.renderer.domElement.height
        )
      });
      
      viewer.wireframeMesh = new Line2(lineGeo, lineMat);
      viewer.wireframeMesh.computeLineDistances();
    }
    viewer.scene.remove(viewer.currentMesh);
    viewer.scene.add(viewer.wireframeMesh);
  }
}

function toggleBackfaceCullingForViewer(sceneName, enabled) {
  const viewer = viewers[sceneName];
  
  if (!viewer.currentMesh) return;
  
  // Set the material side property based on the enabled state
  if (enabled) {
    // Backface culling enabled - only show front faces
    viewer.currentMesh.material.side = THREE.FrontSide;
  } else {
    // Backface culling disabled - show both front and back faces
    viewer.currentMesh.material.side = THREE.DoubleSide;
  }
  
  // The material needs update flag to be set for the change to take effect
  viewer.currentMesh.material.needsUpdate = true;
}

// Handle window resize
window.addEventListener('resize', () => {
  Object.keys(viewers).forEach(sceneName => {
    const viewer = viewers[sceneName];
    const width = viewer.element.clientWidth;
    const height = viewer.element.clientHeight;
    
    viewer.camera.aspect = width / height;
    viewer.camera.updateProjectionMatrix();
    viewer.renderer.setSize(width, height);
  });
});

// Initial load
loadMeshForAllViewers();

function animate() {
  requestAnimationFrame(animate);
  
  Object.keys(viewers).forEach(sceneName => {
    const viewer = viewers[sceneName];
    
    // Check if enough time has passed since last interaction (3 seconds)
    const timeSinceLastInteraction = Date.now() - viewer.lastInteraction;
    const shouldAutoRotate = timeSinceLastInteraction > 3000;
    
    if (shouldAutoRotate && viewer.autoRotate && viewer.currentMesh) {
      // Set the flag to indicate we're auto-rotating
      viewer.isAutoRotating = true;
      
      // Get the rotation axis from the viewer configuration
      const rotationAxis = viewer.rotationAxis.clone().normalize();
      
      // Get the position relative to target
      const relativePos = viewer.camera.position.clone().sub(viewer.controls.target);
      
      // Create a rotation quaternion around the specified axis
      const rotationAngle = viewer.autoRotateSpeed;
      const quaternion = new THREE.Quaternion().setFromAxisAngle(rotationAxis, rotationAngle);
      
      // Apply the rotation to the relative position vector
      relativePos.applyQuaternion(quaternion);
      
      // Set the new camera position
      viewer.camera.position.copy(viewer.controls.target).add(relativePos);
      
      // Ensure the camera's up vector remains constant
      viewer.camera.up.copy(viewer.cameraUp);
      
      // Update the camera to look at the target with the correct up vector
      viewer.camera.lookAt(viewer.cameraTarget);
    }
    
    viewer.controls.update();
    viewer.renderer.render(viewer.scene, viewer.camera);
  });
}
animate();
