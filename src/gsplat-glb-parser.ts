import { Asset, GSplatData, GSplatResource } from 'playcanvas';
import type { AppBase, GraphicsDevice } from 'playcanvas';

const GLB_MAGIC = 0x46546c67;
const GLB_CHUNK_JSON = 0x4e4f534a;
const GLB_CHUNK_BIN = 0x004e4942;

// Resource construction packs the splat data into GPU textures on the main thread. A burst of
// tiles finishing together would otherwise stack several hundred milliseconds of work into one
// frame, so build at most one resource per animation frame.
const resourceJobs: (() => void)[] = [];
let resourceJobScheduled = false;

const runResourceJobs = () => {
    const job = resourceJobs.shift();
    if (job) {
        job();
    }
    if (resourceJobs.length > 0) {
        requestAnimationFrame(runResourceJobs);
    } else {
        resourceJobScheduled = false;
    }
};

const scheduleResourceJob = (job: () => void) => {
    resourceJobs.push(job);
    if (!resourceJobScheduled) {
        resourceJobScheduled = true;
        requestAnimationFrame(runResourceJobs);
    }
};
const PERF = typeof location !== 'undefined' && new URLSearchParams(location.search).has('perf');

const EXTENSION = 'KHR_gaussian_splatting';
const SH_DEGREE_COEF_COUNTS = [3, 5, 7];
const SH_BAND_COEF_COUNTS: Record<number, number> = { 1: 3, 2: 8, 3: 15 };
const SH_C0 = 0.28209479177387814;

// UWA Spatial Vivid EGSC decoder, built from the MIT reference viewer. The decoder accepts the
// original GLB, extracts the UWA_gaussian_splatting_compression_EGSC bufferView and returns CPU
// arrays (positions, scales, rotations, RGBA colors and coefficient-major SH).
const UWA_DECODER_URL = '/uwa/SplatDecoderBootstrap.worker.js';
const UWA_PACK_WORKER_URL = '/uwa/uwa-pack.worker.js';
const UWA_PROTOCOL_VERSION = 2;
const UWA_BUILD_VERSION = 20260909;

type ParserContext = {
    url: string | null;
    ext: string;
    basename: string;
    asset: Asset;
    app: AppBase;
};

type ParserUrl = {
    load: string;
    original: string;
};

type ParserCallback = (err: string | null, resource?: GSplatResource) => void;

type GlbJson = {
    accessors?: any[];
    bufferViews?: any[];
    meshes?: any[];
    extensionsUsed?: string[];
    extensionsRequired?: string[];
    extensions?: Record<string, any>;
};

type Glb = {
    json: GlbJson;
    bin: Uint8Array;
    binOffset: number;
};

const parseGlb = (buffer: ArrayBuffer): Glb => {
    const view = new DataView(buffer);
    if (view.getUint32(0, true) !== GLB_MAGIC) {
        throw new Error('not a binary glTF file');
    }
    if (view.getUint32(4, true) !== 2) {
        throw new Error(`unsupported glTF version ${view.getUint32(4, true)}`);
    }

    const length = Math.min(view.getUint32(8, true), buffer.byteLength);
    let json: GlbJson | null = null;
    let bin: Uint8Array | null = null;
    let binOffset = 0;

    let offset = 12;
    while (offset + 8 <= length) {
        const chunkLength = view.getUint32(offset, true);
        const chunkType = view.getUint32(offset + 4, true);
        const start = offset + 8;
        if (start + chunkLength > buffer.byteLength) {
            throw new Error('truncated GLB chunk');
        }

        if (chunkType === GLB_CHUNK_JSON) {
            const text = new TextDecoder().decode(new Uint8Array(buffer, start, chunkLength));
            json = JSON.parse(text);
        } else if (chunkType === GLB_CHUNK_BIN) {
            bin = new Uint8Array(buffer, start, chunkLength);
            binOffset = start;
        }

        offset = start + chunkLength;
        offset = (offset + 3) & ~3;
    }

    if (!json || !bin) {
        throw new Error('GLB is missing its JSON or BIN chunk');
    }

    return { json, bin, binOffset };
};

const readAccessor = (buffer: ArrayBuffer, glb: Glb, index: number | undefined) => {
    const { json, binOffset } = glb;
    const accessor = index !== undefined ? json.accessors?.[index] : undefined;
    if (!accessor) {
        throw new Error(`missing accessor ${index}`);
    }
    if (accessor.bufferView === undefined) {
        throw new Error('accessor has no bufferView');
    }
    if (accessor.componentType !== 5126) {
        throw new Error(`unsupported accessor component type ${accessor.componentType}`);
    }

    const componentCount = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[accessor.type as string];
    if (!componentCount) {
        throw new Error(`unsupported accessor type ${accessor.type}`);
    }

    const bufferView = json.bufferViews?.[accessor.bufferView];
    if (!bufferView) {
        throw new Error(`missing bufferView ${accessor.bufferView}`);
    }

    const count = accessor.count as number;
    const byteOffset = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
    const stride = bufferView.byteStride ?? componentCount * 4;
    const view = new DataView(buffer, binOffset + byteOffset, bufferView.byteLength - (accessor.byteOffset ?? 0));
    const data = new Float32Array(count * componentCount);

    for (let i = 0; i < count; ++i) {
        for (let c = 0; c < componentCount; ++c) {
            data[i * componentCount + c] = view.getFloat32(i * stride + c * 4, true);
        }
    }

    return { data, componentCount, count };
};

const extractComponent = (source: Float32Array, componentCount: number, component: number, count: number) => {
    const result = new Float32Array(count);
    for (let i = 0; i < count; ++i) {
        result[i] = source[i * componentCount + component];
    }
    return result;
};

const createGSplatData = (device: GraphicsDevice, buffer: ArrayBuffer) => {
    const glb = parseGlb(buffer);
    const { json } = glb;

    const uwa = [...new Set([...(json.extensionsUsed ?? []), ...(json.extensionsRequired ?? [])])]
    .filter(extension => extension.startsWith('UWA_'));
    if (uwa.length > 0) {
        throw new Error(`UWA-compressed GLB (${uwa.join(', ')}) is not supported. Use the SOG variant, or provide the UWA EGSC decoder.`);
    }

    const primitive = json.meshes
    ?.flatMap(mesh => mesh.primitives ?? [])
    .find(primitive => primitive.attributes?.POSITION !== undefined && primitive.extensions?.[EXTENSION]);
    if (!primitive) {
        throw new Error(`GLB has no ${EXTENSION} primitive`);
    }

    const attributes = primitive.attributes;
    const readAttribute = (name: string) => {
        const accessorIndex = attributes[name];
        if (accessorIndex === undefined) {
            throw new Error(`missing ${name} attribute`);
        }
        return readAccessor(buffer, glb, accessorIndex);
    };

    const positions = readAttribute('POSITION');
    const rotations = readAttribute(`${EXTENSION}:ROTATION`);
    const scales = readAttribute(`${EXTENSION}:SCALE`);
    const opacities = readAttribute(`${EXTENSION}:OPACITY`);
    const sh0 = readAttribute(`${EXTENSION}:SH_DEGREE_0_COEF_0`);

    const count = positions.count;
    if (count === 0) {
        throw new Error('GLB splat primitive is empty');
    }

    const properties: any[] = [];
    const addProp = (name: string, storage: Float32Array) => {
        properties.push({ type: 'float', name, storage, byteSize: 4 });
    };

    addProp('x', extractComponent(positions.data, 3, 0, count));
    addProp('y', extractComponent(positions.data, 3, 1, count));
    addProp('z', extractComponent(positions.data, 3, 2, count));

    // glTF stores the quaternion as xyzw, PLY layout uses rot_0 for w
    addProp('rot_0', extractComponent(rotations.data, 4, 3, count));
    addProp('rot_1', extractComponent(rotations.data, 4, 0, count));
    addProp('rot_2', extractComponent(rotations.data, 4, 1, count));
    addProp('rot_3', extractComponent(rotations.data, 4, 2, count));

    addProp('scale_0', extractComponent(scales.data, 3, 0, count));
    addProp('scale_1', extractComponent(scales.data, 3, 1, count));
    addProp('scale_2', extractComponent(scales.data, 3, 2, count));

    addProp('opacity', opacities.data);

    addProp('f_dc_0', extractComponent(sh0.data, 3, 0, count));
    addProp('f_dc_1', extractComponent(sh0.data, 3, 1, count));
    addProp('f_dc_2', extractComponent(sh0.data, 3, 2, count));

    // determine the number of complete SH bands present
    let bands = 0;
    for (let degree = 1; degree <= 3; ++degree) {
        let complete = true;
        for (let c = 0; c < SH_DEGREE_COEF_COUNTS[degree - 1]; ++c) {
            if (attributes[`${EXTENSION}:SH_DEGREE_${degree}_COEF_${c}`] === undefined) {
                complete = false;
                break;
            }
        }
        if (!complete) {
            break;
        }
        bands = degree;
    }

    // convert per-coefficient RGB attributes to the channel-major PLY layout
    if (bands > 0) {
        const numCoefs = SH_BAND_COEF_COUNTS[bands];
        const rest = new Array(numCoefs * 3);
        let k = 0;
        for (let degree = 1; degree <= bands; ++degree) {
            for (let c = 0; c < SH_DEGREE_COEF_COUNTS[degree - 1]; ++c) {
                const name = `${EXTENSION}:SH_DEGREE_${degree}_COEF_${c}`;
                const source = readAttribute(name);
                rest[k] = extractComponent(source.data, 3, 0, count);
                rest[numCoefs + k] = extractComponent(source.data, 3, 1, count);
                rest[numCoefs * 2 + k] = extractComponent(source.data, 3, 2, count);
                ++k;
            }
        }
        for (let i = 0; i < numCoefs * 3; ++i) {
            addProp(`f_rest_${i}`, rest[i]);
        }
    }

    const data = new GSplatData([{
        name: 'vertex',
        count,
        properties
    }]);

    // the extension stores linear scale and post-sigmoid opacity
    data.activated = true;

    // LOD manifests reference splat ranges by the file's original point order, so the data
    // must not be Morton-reordered (unlike a single-file splat, where order is irrelevant).

    return { data, bands };
};

const isUwaGlb = (json: GlbJson) => [
    ...new Set([...(json.extensionsUsed ?? []), ...(json.extensionsRequired ?? [])])
].some(extension => extension.startsWith('UWA_'));

const createUwaGSplatData = (decoded: any) => {
    const count = decoded.numPoints as number;
    const positions = decoded.positions as Float32Array;
    const scales = decoded.scales as Float32Array;
    const rotations = decoded.rotations as Float32Array;
    const colors = decoded.colors as Uint8Array;
    const featuresRest = decoded.featuresRest as Float32Array | null;
    const featuresRestChannelMajor = decoded.featuresRestChannelMajor as Float32Array | null;
    const shDegree = (decoded.shDegree ?? 0) as number;
    if (PERF) {
        console.log('[GLB perf] decoded shDegree=' + shDegree + ' rest=' + (decoded.featuresRest?.length ?? -1) + ' cm=' + (decoded.featuresRestChannelMajor?.length ?? -1) + ' points=' + count);
    }

    if (!count || !positions || !scales || !rotations || !colors) {
        throw new Error('UWA decoder returned incomplete splat data');
    }

    const properties: any[] = [];
    const addProp = (name: string, storage: Float32Array) => {
        properties.push({ type: 'float', name, storage, byteSize: 4 });
    };

    addProp('x', extractComponent(positions, 3, 0, count));
    addProp('y', extractComponent(positions, 3, 1, count));
    addProp('z', extractComponent(positions, 3, 2, count));

    // UWA returns the quaternion as wxyz, matching the PLY layout
    addProp('rot_0', extractComponent(rotations, 4, 0, count));
    addProp('rot_1', extractComponent(rotations, 4, 1, count));
    addProp('rot_2', extractComponent(rotations, 4, 2, count));
    addProp('rot_3', extractComponent(rotations, 4, 3, count));

    addProp('scale_0', extractComponent(scales, 3, 0, count));
    addProp('scale_1', extractComponent(scales, 3, 1, count));
    addProp('scale_2', extractComponent(scales, 3, 2, count));

    // UWA colors are display-space DC color (0.5 + SH_C0 * f_dc) * 255 and post-sigmoid alpha.
    // Convert them back to the engine's activated representation: SH DC and linear opacity.
    const opacity = new Float32Array(count);
    const dc0 = new Float32Array(count);
    const dc1 = new Float32Array(count);
    const dc2 = new Float32Array(count);
    for (let i = 0; i < count; ++i) {
        const r = colors[i * 4 + 0] / 255;
        const g = colors[i * 4 + 1] / 255;
        const b = colors[i * 4 + 2] / 255;
        opacity[i] = colors[i * 4 + 3] / 255;
        dc0[i] = (r - 0.5) / SH_C0;
        dc1[i] = (g - 0.5) / SH_C0;
        dc2[i] = (b - 0.5) / SH_C0;
    }
    addProp('opacity', opacity);
    addProp('f_dc_0', dc0);
    addProp('f_dc_1', dc1);
    addProp('f_dc_2', dc2);

    const numCoefs = SH_BAND_COEF_COUNTS[shDegree] ?? 0;
    if (numCoefs > 0) {
        if (featuresRestChannelMajor && featuresRestChannelMajor.length >= count * numCoefs * 3) {
            // The decoder worker already produced channel-major f_rest; expose views into its
            // transferred buffer without touching the data on the main thread.
            for (let k = 0; k < numCoefs; ++k) {
                addProp(`f_rest_${k}`, featuresRestChannelMajor.subarray(k * count, (k + 1) * count));
            }
            for (let k = 0; k < numCoefs; ++k) {
                addProp(`f_rest_${numCoefs + k}`, featuresRestChannelMajor.subarray((numCoefs + k) * count, (numCoefs + k + 1) * count));
            }
            for (let k = 0; k < numCoefs; ++k) {
                addProp(`f_rest_${2 * numCoefs + k}`, featuresRestChannelMajor.subarray((2 * numCoefs + k) * count, (2 * numCoefs + k + 1) * count));
            }
        } else if (featuresRest && featuresRest.length >= count * 45) {
            // Fallback for a decoder that still returns coefficient-major SH.
            const rest = new Array(numCoefs * 3);
            for (let k = 0; k < numCoefs; ++k) {
                const r = new Float32Array(count);
                const g = new Float32Array(count);
                const b = new Float32Array(count);
                for (let i = 0; i < count; ++i) {
                    const offset = i * 45 + k * 3;
                    r[i] = featuresRest[offset];
                    g[i] = featuresRest[offset + 1];
                    b[i] = featuresRest[offset + 2];
                }
                rest[k] = r;
                rest[numCoefs + k] = g;
                rest[numCoefs * 2 + k] = b;
            }
            for (let i = 0; i < numCoefs * 3; ++i) {
                addProp(`f_rest_${i}`, rest[i]);
            }
        }
    }

    const data = new GSplatData([{
        name: 'vertex',
        count,
        properties
    }]);

    data.activated = true;

    return data;
};

type PackedUwa = {
    color: Uint16Array;
    transformA: Uint32Array;
    transformB: Uint16Array;
    sh: Record<string, Uint32Array>;
};

const packedMap = new WeakMap<object, PackedUwa>();

// The engine packs GSplatData into GPU streams on the main thread. For UWA data we pre-pack the
// same streams in a worker and let the resource constructor just copy them into its textures.
const installUwaFastPath = () => {
    const proto = GSplatResource.prototype as any;
    if (proto.__uwaPackedFastPath) {
        return;
    }

    const updateColorData = proto.updateColorData;
    const updateTransformData = proto.updateTransformData;
    const updateSHData = proto.updateSHData;

    proto.updateColorData = function (gsplatData: any) {
        const packed = packedMap.get(gsplatData);
        if (packed) {
            const texture = this.streams.getTexture('splatColor');
            if (texture) {
                const data = texture.lock();
                data.set(packed.color);
                texture.unlock();
            }
            return;
        }
        return updateColorData.call(this, gsplatData);
    };

    proto.updateTransformData = function (gsplatData: any) {
        const packed = packedMap.get(gsplatData);
        if (packed) {
            const transformA = this.streams.getTexture('transformA');
            const transformB = this.streams.getTexture('transformB');
            if (transformA) {
                const dataA = transformA.lock();
                dataA.set(packed.transformA);
                transformA.unlock();
            }
            if (transformB) {
                const dataB = transformB.lock();
                dataB.set(packed.transformB);
                transformB.unlock();
            }
            return;
        }
        return updateTransformData.call(this, gsplatData);
    };

    proto.updateSHData = function (gsplatData: any) {
        const packed = packedMap.get(gsplatData);
        if (packed) {
            for (const [name, values] of Object.entries(packed.sh)) {
                const texture = this.streams.getTexture(name);
                if (texture) {
                    const data = texture.lock();
                    data.set(values);
                    texture.unlock();
                }
            }
            return;
        }
        return updateSHData.call(this, gsplatData);
    };

    proto.__uwaPackedFastPath = true;
};

installUwaFastPath();

const packUwaData = (decoded: any) => new Promise<{ data: GSplatData; packed: PackedUwa }>((resolve, reject) => {
    const worker = new Worker(UWA_PACK_WORKER_URL, { type: 'module' });
    let settled = false;
    let timeout: ReturnType<typeof setTimeout>;

    const finish = (err: Error | null, result?: { data: GSplatData; packed: PackedUwa }) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        worker.terminate();
        if (err) {
            reject(err);
        } else {
            resolve(result!);
        }
    };

    timeout = setTimeout(() => finish(new Error('UWA pack timed out')), 180000);

    worker.onmessage = (event) => {
        const message = event.data;
        if (!message) {
            return;
        }
        if (message.error) {
            finish(new Error(message.error));
            return;
        }

        try {
            const count = decoded.numPoints as number;
            const positions = message.positions as Float32Array;
            const scales = message.scales as Float32Array;
            const properties: any[] = [];
            const addProp = (name: string, storage: Float32Array) => {
                properties.push({ type: 'float', name, storage, byteSize: 4 });
            };

            addProp('x', extractComponent(positions, 3, 0, count));
            addProp('y', extractComponent(positions, 3, 1, count));
            addProp('z', extractComponent(positions, 3, 2, count));
            addProp('scale_0', extractComponent(scales, 3, 0, count));
            addProp('scale_1', extractComponent(scales, 3, 1, count));
            addProp('scale_2', extractComponent(scales, 3, 2, count));

            const data = new GSplatData([{
                name: 'vertex',
                count,
                properties
            }]);
            Object.defineProperty(data, 'shBands', {
                value: message.shBands,
                configurable: true
            });
            data.activated = true;
            packedMap.set(data, message.packed);
            finish(null, { data, packed: message.packed });
        } catch (e) {
            finish(e as Error);
        }
    };

    worker.onerror = (event) => {
        finish(new Error(`UWA pack worker failed: ${event.message}`));
    };

    const transferList: Transferable[] = [];
    const seen = new Set<ArrayBuffer>();
    const transfer = (value: any) => {
        const buffer = value?.buffer;
        if (buffer && !seen.has(buffer)) {
            seen.add(buffer);
            transferList.push(buffer);
        }
    };
    transfer(decoded.positions);
    transfer(decoded.scales);
    transfer(decoded.rotations);
    transfer(decoded.colors);
    transfer(decoded.featuresRestChannelMajor);

    worker.postMessage({
        numPoints: decoded.numPoints,
        positions: decoded.positions,
        scales: decoded.scales,
        rotations: decoded.rotations,
        colors: decoded.colors,
        featuresRestChannelMajor: decoded.featuresRestChannelMajor,
        shDegree: decoded.shDegree
    }, transferList);
});

const decodeUwaGlbAttempt = (buffer: ArrayBuffer, textureStrategies: string[]) => new Promise<any>((resolve, reject) => {
    const decodeStarted = PERF ? performance.now() : 0;
    const worker = new Worker(UWA_DECODER_URL, { type: 'module' });
    let settled = false;
    let timeout: ReturnType<typeof setTimeout>;

    const finish = (err: Error | null, data?: any) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        worker.terminate();
        if (err) {
            reject(err);
        } else {
            resolve(data!);
        }
    };

    timeout = setTimeout(() => finish(new Error('UWA decode timed out')), 180000);

    worker.onmessage = (event) => {
        const message = event.data;
        if (!message || message.type !== 'decodeResult') {
            return;
        }
        if (!message.success) {
            finish(new Error(message.error || 'UWA decode failed'));
            return;
        }

        if (PERF) {
            console.log('[GLB perf] decode=' + (performance.now() - decodeStarted).toFixed(0) + 'ms points=' + message.data.numPoints + ' sh=' + message.data.shDegree);
        }
        finish(null, message.data);
    };

    worker.onerror = (event) => {
        finish(new Error(`UWA decoder worker failed: ${event.message}`));
    };

    worker.postMessage({
        type: 'decode',
        protocolVersion: UWA_PROTOCOL_VERSION,
        buildVersion: UWA_BUILD_VERSION,
        requestId: 1,
        buffer,
        textureStrategy: textureStrategies[0],
        textureStrategies,
        postedEpochMs: Date.now(),
        verboseLog: false,
        traceOriginAbsMs: performance.now(),
        reconstructionTraceLevel: 'off',
        reconstructionWorkerCount: 2,
        postprocessCompaction: true,
        minimumAlpha: 0,
        enablePreview: false
    });
});

const decodeUwaGlb = async (buffer: ArrayBuffer): Promise<any> => {
    try {
        return await decodeUwaGlbAttempt(buffer, ['cpu']);
    } catch (e) {
        // Some UWA payloads (e.g. SH0 / masked scenes) cannot reconstruct the CPU SH attribute;
        // retry with the compressed SH texture path and render the DC color instead.
        const message = (e as Error).message ?? '';
        if (!/unexpected length|All texture strategies failed|Reconstructed shard/i.test(message)) {
            throw e;
        }
        return await decodeUwaGlbAttempt(buffer, ['astc', 'cpu']);
    }
};

class GSplatGlbParser {
    handler: any;

    private _app: AppBase;

    constructor(app: AppBase) {
        this._app = app;
    }

    canParse(context: ParserContext) {
        return context.ext === 'glb';
    }

    load(url: ParserUrl, callback: ParserCallback, asset?: Asset) {
        Asset.fetchArrayBuffer(url.load, (err, result) => {
            if (err) {
                callback(err);
                return;
            }

            const prepareCenters = (this._app.scene as any).gsplatCentersEnabled !== false;

            try {
                const { json } = parseGlb(result);
                if (isUwaGlb(json)) {
                    decodeUwaGlb(result)
                    .then(async (decoded) => {
                        const legacy = new URLSearchParams(location.search).has('legacy');
                        let data;
                        if (legacy) {
                            const mapStarted = PERF ? performance.now() : 0;
                            data = createUwaGSplatData(decoded);
                            if (PERF) {
                                console.log('[GLB perf] legacy map=' + (performance.now() - mapStarted).toFixed(1) + 'ms');
                            }
                        } else {
                            const packStarted = PERF ? performance.now() : 0;
                            data = (await packUwaData(decoded)).data;
                            if (PERF) {
                                console.log('[GLB perf] pack=' + (performance.now() - packStarted).toFixed(1) + 'ms');
                            }
                        }

                        scheduleResourceJob(() => {
                            const buildStarted = PERF ? performance.now() : 0;
                            try {
                                callback(null, new GSplatResource(this._app.graphicsDevice, data, { prepareCenters }));
                                if (PERF) {
                                    console.log('[GLB perf] build=' + (performance.now() - buildStarted).toFixed(1) + 'ms queue=' + resourceJobs.length);
                                }
                            } catch (e) {
                                callback((e as Error).message);
                            }
                        });
                    })
                    .catch((e: Error) => {
                        callback(e.message);
                    });
                    return;
                }

                const { data } = createGSplatData(this._app.graphicsDevice, result);
                scheduleResourceJob(() => {
                    try {
                        callback(null, new GSplatResource(this._app.graphicsDevice, data, { prepareCenters }));
                    } catch (e) {
                        callback((e as Error).message);
                    }
                });
            } catch (e) {
                callback((e as Error).message);
            }
        }, undefined, this.handler?.maxRetries ?? 0);
    }
}

export { GSplatGlbParser };



















