import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export class WaterRenderer {
    constructor(container, maxParticles, width3D = 0.4, layers = 30) {
        this.maxParticles = maxParticles;
        this.width3D = width3D;
        this.layers = layers;

        // 1. SETUP THREE.JS
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x111111); 

        this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
        this.camera.position.set(0.5, 0.8, 2.0); 
        this.camera.lookAt(0.5, 0.4, 0);

        // UPDATED: Added powerPreference hint
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            powerPreference: "high-performance"
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        container.appendChild(this.renderer.domElement);

        // 2. CONTROLS
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.target.set(0.5, 0.5, 0); 
        this.controls.update();

        const axesHelper = new THREE.AxesHelper(1);
        this.scene.add(axesHelper);

        // 3. INITIALIZE SYSTEM
        this.initDataTexture();
        this.initInstancedMesh();

        // Debug flag to print max speed once
        this.hasLoggedStats = false;

        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    initDataTexture() {
        this.texSize = Math.ceil(Math.sqrt(this.maxParticles));
        this.dataArray = new Float32Array(this.texSize * this.texSize * 4);
        this.dataTexture = new THREE.DataTexture(
            this.dataArray, this.texSize, this.texSize, THREE.RGBAFormat, THREE.FloatType
        );
        this.dataTexture.needsUpdate = true;
    }

    initInstancedMesh() {
        const geometry = new THREE.PlaneGeometry(0.015, 0.015); 

        const instancedGeo = new THREE.InstancedBufferGeometry();
        instancedGeo.index = geometry.index;
        instancedGeo.attributes.position = geometry.attributes.position;
        instancedGeo.attributes.uv = geometry.attributes.uv;

        const totalInstances = this.maxParticles * this.layers;
        const indices = new Float32Array(totalInstances);
        const offsets = new Float32Array(totalInstances * 3);

        for (let i = 0; i < totalInstances; i++) {
            indices[i] = i % this.maxParticles;
            const stride = i * 3;
            offsets[stride]     = (Math.random() - 0.5) * 0.02; 
            offsets[stride + 1] = (Math.random() - 0.5) * 0.02; 
            offsets[stride + 2] = (Math.random() - 0.5) * this.width3D; 
        }

        instancedGeo.setAttribute('aSimIndex', new THREE.InstancedBufferAttribute(indices, 1));
        instancedGeo.setAttribute('aOffset', new THREE.InstancedBufferAttribute(offsets, 3));

        const material = new THREE.ShaderMaterial({
            uniforms: {
                uTexture: { value: this.dataTexture },
                uTexSize: { value: this.texSize },
                // DEBUG COLORS: Blue -> Red
                uColorDeep: { value: new THREE.Color(0x0044ff) },
                uColorFoam: { value: new THREE.Color(1.0, 0.0, 0.0) } 
            },
            vertexShader: `
                uniform sampler2D uTexture;
                uniform float uTexSize;
                attribute float aSimIndex;
                attribute vec3 aOffset;
                varying vec2 vUv;
                varying float vVal; 

                void main() {
                    vUv = uv;

                    float tx = mod(aSimIndex, uTexSize) / uTexSize;
                    float ty = floor(aSimIndex / uTexSize) / uTexSize;
                    vec2 texUV = vec2(tx + (0.5/uTexSize), ty + (0.5/uTexSize));
                    vec4 data = texture2D(uTexture, texUV);

                    if (data.x <= 0.0 && data.y <= 0.0) {
                        gl_Position = vec4(2.0, 2.0, 2.0, 1.0); 
                        return;
                    }

                    // Read intensity from Z channel
                    vVal = data.z;

                    vec3 pos = vec3(
                        data.x + aOffset.x, 
                        data.y + aOffset.y, 
                        aOffset.z
                    );
                    
                    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                    mvPosition.xy += position.xy; 

                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                varying vec2 vUv;
                varying float vVal;
                uniform vec3 uColorDeep;
                uniform vec3 uColorFoam;

                void main() {
                    float dist = length(vUv - 0.5);
                    if (dist > 0.5) discard;

                    float alpha = 1.0 - smoothstep(0.3, 0.5, dist);

                    // DEBUG GRADIENT
                    // Use a wider range (0.0 to 2.0) to capture fast particles
                    // If vVal is 0.0 -> Blue
                    // If vVal is 2.0 -> Red
                    float mixFactor = smoothstep(0.0, 2.0, vVal);
                    
                    vec3 color = mix(uColorDeep, uColorFoam, mixFactor);

                    gl_FragColor = vec4(color, alpha);
                }
            `,
            transparent: true,
            depthWrite: false, 
            depthTest: true,
            side: THREE.DoubleSide
        });

        this.mesh = new THREE.InstancedMesh(instancedGeo, material, totalInstances);
        this.mesh.frustumCulled = false; 
        this.scene.add(this.mesh);
    }

    updateFromBuffers(posFlat, velFlat, cFlat) {
        const d = this.dataArray;
        const count = Math.min(posFlat.length / 2, this.maxParticles);
        let maxSpeedFound = 0.0;
        
        for (let i = 0; i < count; i++) {
            const strideTex = i * 4;
            
            // 1. Position
            d[strideTex] = posFlat[i * 2];
            d[strideTex + 1] = posFlat[i * 2 + 1];
            
            // 2. DEBUG MODE: Use Velocity ONLY
            let val = 0;

            if (velFlat && velFlat.length > 0) {
                const vx = velFlat[i * 2];
                const vy = velFlat[i * 2 + 1];
                val = Math.sqrt(vx*vx + vy*vy);
                
                if (val > maxSpeedFound) maxSpeedFound = val;

                // Scale: Assuming velocity is in 0-1ish range, let's leave it raw first
                // The shader expects 0.0 to 2.0. 
                // If your river is slow (0.1), multiply this by 10.0
                // If your river is fast (10.0), multiply by 0.1
                // Let's try raw for now to see the console log.
            }

            d[strideTex + 2] = val; 
        }
        
        if (!this.hasLoggedStats && maxSpeedFound > 0) {
            console.log("--- DATA STATS ---");
            console.log("Max Speed observed:", maxSpeedFound);
            console.log("If this is < 0.1, multiply val by 10 in updateFromBuffers");
            console.log("If this is > 5.0, divide val in updateFromBuffers");
            this.hasLoggedStats = true;
        }

        this.dataTexture.needsUpdate = true;
    }

    render() {
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}