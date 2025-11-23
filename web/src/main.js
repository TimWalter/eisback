import './style.css';
import { SimulationLoader } from './loader.js';
import { WaterRenderer } from './renderer.js';

// 1. SETUP UI
document.querySelector('#app').innerHTML = `
  <div style="position: absolute; top: 0; left: 0; width: 100%; z-index: 10;">
    <div style="padding: 20px; color: #eee; text-shadow: 1px 1px 2px #000;">
      <h1 style="margin: 0;">Manual Playback</h1>
      <p>Frame: <span id="frameInfo">-</span> | Particles: <span id="particleCount">-</span></p>
      <div style="margin-top: 10px; display: flex; gap: 10px;">
        <button id="btnPlay" style="padding: 8px 16px; cursor: pointer; background: #00aa44; color: white; border: 1px solid #666;">Play</button>
        <button id="btnNext" style="padding: 8px 16px; cursor: pointer; background: #444; color: white; border: 1px solid #666;">Next Frame >></button>
      </div>
      <p id="status" style="font-size: 0.8em; color: #aaa;">Waiting...</p>
    </div>
  </div>
  <div id="canvas-container" style="width: 100vw; height: 100vh; background: #000;"></div>
`;

// 2. CONFIGURATION
const CONFIG = {
    PARTICLE_COUNT: 90000, 
    RIVER_WIDTH: 0.4,      
    LAYERS: 30,
    PLAYBACK_SPEED: 30 // ms delay between frames
};

// 3. INIT SYSTEMS
const loader = new SimulationLoader('/output/'); 
const renderer = new WaterRenderer(
    document.getElementById('canvas-container'), 
    CONFIG.PARTICLE_COUNT, 
    CONFIG.RIVER_WIDTH, 
    CONFIG.LAYERS
);

// 4. LOGIC
const btnNext = document.getElementById('btnNext');
const btnPlay = document.getElementById('btnPlay');
const statusLabel = document.getElementById('status');
const frameLabel = document.getElementById('frameInfo');
const countLabel = document.getElementById('particleCount');

let isPlaying = false;
let isFetching = false;
let lastFrameTime = 0;

async function loadNextFrame() {
    // Prevent stacking requests
    if (isFetching) return;
    isFetching = true;

    btnNext.disabled = true;
    if (!isPlaying) statusLabel.innerText = "Fetching...";
    
    const data = await loader.getNextFrame();
    
    if (data) {
        renderer.updateFromBuffers(data.pos, data.vel, data.C);
        
        frameLabel.innerText = loader.currentFrame - 1; 
        countLabel.innerText = data.pos.length / 2;
        if (!isPlaying) statusLabel.innerText = "Rendered.";
    } else {
        statusLabel.innerText = "Error or End of Stream (Looping...)";
    }
    
    btnNext.disabled = false;
    isFetching = false;
}

// Load Frame 0 immediately
loadNextFrame();

// Button Handlers
btnNext.addEventListener('click', () => {
    loadNextFrame();
});

btnPlay.addEventListener('click', () => {
    isPlaying = !isPlaying;
    btnPlay.innerText = isPlaying ? "Pause" : "Play";
    btnPlay.style.background = isPlaying ? "#aa4400" : "#00aa44";
});

// 5. ANIMATION LOOP
function animate(timestamp) {
    requestAnimationFrame(animate);
    
    // 1. Handle Automatic Playback
    if (isPlaying) {
        if (timestamp - lastFrameTime > CONFIG.PLAYBACK_SPEED) {
            loadNextFrame();
            lastFrameTime = timestamp;
        }
    }

    // 2. Handle Rendering (Camera controls always work)
    renderer.render();
}
animate(0);