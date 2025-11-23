export class SimulationLoader {
    constructor(baseUrl = '/output/', maxFrames = 100) {
        this.baseUrl = baseUrl;
        this.currentFrame = 0;
        this.maxFrames = maxFrames;
        this.isPlaying = true;
        
        // Cache for pre-loaded frames
        this.cache = new Map();
        this.isLoading = false;
    }

    async loadFrame(frameId) {
        // Pad with zeros (e.g., 0 -> "00000")
        const id = frameId.toString().padStart(5, '0');
        
        try {
            // Fetch Position, Velocity, and C-Matrix in parallel
            const [posRes, velRes, cRes] = await Promise.all([
                fetch(`${this.baseUrl}pos_${id}.bin`),
                fetch(`${this.baseUrl}vel_${id}.bin`),
                fetch(`${this.baseUrl}C_${id}.bin`)
            ]);

            if (!posRes.ok || !velRes.ok || !cRes.ok) throw new Error('Frame not found');

            const posBuffer = await posRes.arrayBuffer();
            const velBuffer = await velRes.arrayBuffer();
            const cBuffer = await cRes.arrayBuffer();

            return {
                pos: new Float32Array(posBuffer),
                vel: new Float32Array(velBuffer),
                C: new Float32Array(cBuffer)
            };
        } catch (e) {
            console.warn(`Stopped at frame ${frameId}:`, e);
            return null;
        }
    }

    async getNextFrame() {
        // Simple buffering logic
        const frameData = await this.loadFrame(this.currentFrame);
        
        if (frameData) {
            this.currentFrame++;
            return frameData;
        } else {
            // Loop back to start
            this.currentFrame = 0;
            return await this.loadFrame(0);
        }
    }
}