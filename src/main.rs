//! A `#![no_std]` TRUEOS boat scene built with Picasso.
//!
//! At boot the platform resolves Dealer IDs to opaque vGPU buffers. Picasso
//! receives only stable resource identities and byte-relative ranges.
#![no_std]

extern crate alloc;

mod demodata;

use alloc::{format, string::String, vec::Vec};
use trueos::ui4_scene::{
    CursorIcon, CursorSource, Damage, Error as Ui4Error, Frame, POINTER_BUTTON_PRIMARY,
    output_dimensions,
};
use trueos::vgpu::{
    BUFFER_USAGE_INDEX, BUFFER_USAGE_MAP_READ, BUFFER_USAGE_MAP_WRITE, BUFFER_USAGE_VERTEX, Buffer, BufferSlice,
    Capabilities, Device, PRIMITIVE_TOPOLOGY_LINE_LIST, PRIMITIVE_TOPOLOGY_LINE_LOOP,
    PRIMITIVE_TOPOLOGY_LINE_STRIP, PRIMITIVE_TOPOLOGY_POINT_LIST, PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
    PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, Queue, QueueClass,
    RETAINED_VERTEX_LAYOUT_POS_NORMAL, RETAINED_VERTEX_LAYOUT_POS_NORMAL_UV_TANGENT,
    RetainedCamera, RetainedDrawRange, RetainedFrameSubmit, RetainedFrameSubmitV2,
    RetainedFrameSubmitV3, RetainedMaterial,
    RetainedMaterialParameters, RetainedMesh, RetainedMeshDescriptor, RetainedTransformSeed,
    VVideoMem,
};
use trueos::{
    clock,
    logl::{self, level},
    vsys,
};
use trueos_picasso::ExecRing;
use trueos_picasso::Picasso;
use trueos_picasso::cam::{Camera, FlyCam, Projection, Quaternion};
use trueos_picasso::{CubismError, SharedByteRange, VVideoRingError, VisibilityOps};
use trueos_picasso::{
    ExecutablePrimitive, PreparedRange, PrimitiveTopology, ResourceId, SharedResourceId,
    TransformRefList, TransformStateRange, TransformValue,
};

pub struct PreparedAsset {
    pub name: &'static str,
    pub vertices: &'static [u8],
    pub indices: &'static [u8],
    pub vertex_count: u32,
    pub index_count: u32,
    pub vertex_stride: usize,
    /// Every image and scalar required by one glTF material. Images remain
    /// encoded until the owner-scoped retained-material admission at startup.
    pub material: PreparedMaterial,
    pub sampled_material: bool,
    pub helmet_program: bool,
    /// Every glTF primitive imported from this asset.  The current retained
    /// presentation consumes triangle lists, but import keeps the source
    /// topology intact for topology-capable consumers.
    pub primitives: &'static [PreparedPrimitive],
    /// A whole retained mesh has one native topology. Mixed-topology glTF
    /// assets remain losslessly prepared above and will become one retained
    /// mesh per primitive when multi-draw retained submission is introduced.
    pub retained_topology: Option<PrimitiveTopology>,
    /// True when a source material declares glTF `doubleSided`.
    pub retained_double_sided: bool,
}

#[derive(Clone, Copy)]
pub struct PreparedTexture {
    pub name: &'static str,
    pub bytes: &'static [u8],
}

impl PreparedTexture {
    pub const NONE: Self = Self {
        name: "",
        bytes: &[],
    };
}

#[derive(Clone, Copy)]
pub struct PreparedMaterial {
    pub base_color: PreparedTexture,
    pub metallic_roughness: PreparedTexture,
    pub emissive: PreparedTexture,
    pub occlusion: PreparedTexture,
    pub normal: PreparedTexture,
    pub parameters: RetainedMaterialParameters,
}

/// One primitive's ranges in a build-prepared asset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedPrimitive {
    pub topology: PrimitiveTopology,
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_index: u32,
    pub index_count: u32,
    /// Source glTF material's `doubleSided` value.
    pub double_sided: bool,
}

const fn vgpu_topology(topology: PrimitiveTopology) -> u32 {
    match topology {
        PrimitiveTopology::PointList => PRIMITIVE_TOPOLOGY_POINT_LIST,
        PrimitiveTopology::LineList => PRIMITIVE_TOPOLOGY_LINE_LIST,
        PrimitiveTopology::LineLoop => PRIMITIVE_TOPOLOGY_LINE_LOOP,
        PrimitiveTopology::LineStrip => PRIMITIVE_TOPOLOGY_LINE_STRIP,
        PrimitiveTopology::TriangleList => PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        PrimitiveTopology::TriangleStrip => PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
        PrimitiveTopology::TriangleFan => PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
    }
}
include!(concat!(env!("OUT_DIR"), "/prepared_assets.rs"));

/// Prepared bytes loaded back through Picasso's runtime database boundary.
/// Static build output is permitted to seed the database, but never to feed a
/// GPU upload directly.
struct DatabasePreparedAsset {
    vertices: Vec<u8>,
    indices: Vec<u8>,
    material: DatabasePreparedMaterial,
}

struct DatabasePreparedMaterial {
    base_color: Vec<u8>,
    metallic_roughness: Vec<u8>,
    emissive: Vec<u8>,
    occlusion: Vec<u8>,
    normal: Vec<u8>,
    parameters: RetainedMaterialParameters,
}

struct DatabasePreparedCatalog {
    assets: Vec<DatabasePreparedAsset>,
}

pub const HELMET_VERTEX_COUNT: u32 = ASSETS[0].vertex_count;
pub const HELMET_INDEX_COUNT: u32 = ASSETS[0].index_count;
pub const HELMET_VERTEX_BYTES: u64 = ASSETS[0].vertices.len() as u64;
pub const HELMET_INDEX_BYTES: u64 = ASSETS[0].index_count as u64 * 4;

pub const HELMET_VERTICES: ResourceId = ResourceId(0x0001_0000_0000_0001);
pub const HELMET_INDICES: ResourceId = ResourceId(0x0001_0000_0000_0002);
pub const HELMET_TRANSFORM_REFS: ResourceId = ResourceId(0x0001_0000_0000_0003);
pub const HEAD_INSTANCE_COUNT: u32 = 4;
pub const HEAD_TRANSFORM_STATE_BYTES: u64 =
    HEAD_INSTANCE_COUNT as u64 * core::mem::size_of::<TransformValue>() as u64;

/// Mesh-local `TransformId`s. All four entries reference one retained state
/// table while the vertex/index resources above remain single-copy.
pub static HELMET_TRANSFORM_REFS_U32: &[u8; 16] = &[
    0, 0, 0, 0, // top-left
    1, 0, 0, 0, // top-right
    2, 0, 0, 0, // bottom-left
    3, 0, 0, 0, // bottom-right
];

pub static HELMET_POSNORMAL: &[u8] = ASSETS[0].vertices;
pub static HELMET_INDICES_U32: &[u8] = ASSETS[0].indices;

/// Describe the host-prepared DamagedHelmet geometry without runtime parsing.
pub const fn prepared_geometry() -> ExecutablePrimitive {
    ExecutablePrimitive {
        topology: PrimitiveTopology::TriangleList,
        vertices: PreparedRange {
            resource: HELMET_VERTICES,
            offset: 0,
            byte_length: HELMET_VERTEX_BYTES,
            revision: 1,
        },
        indices: Some(PreparedRange {
            resource: HELMET_INDICES,
            offset: 0,
            byte_length: HELMET_INDEX_BYTES,
            revision: 1,
        }),
        index_format: Some(trueos_picasso::IndexFormat::Uint32),
        vertex_stride: ASSETS[0].vertex_stride as u32,
        vertex_count: HELMET_VERTEX_COUNT,
        index_count: HELMET_INDEX_COUNT,
    }
}

/// GPU-side animation operation applied to one retained transform state.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct HeadInstanceProgram {
    pub initial: TransformValue,
    /// Signed Z angular velocity in radians/second. Negative is clockwise in
    /// the clip-space convention used by this example.
    pub angular_velocity_z: f32,
    /// Zero disables pulsing. A nonzero value is one descending or ascending
    /// half-cycle; the complete scale loop lasts twice this duration.
    pub scale_half_period_seconds: f32,
    pub reserved: [u32; 2],
}

// Preserve the distance of the former first offset helmet from the origin,
// then use it as the signed X/Y offset for a symmetric 2-by-2 layout.
const HEAD_PLANE_OFFSET: f32 = 1.6;
const HELMET_SCALE: f32 = 0.9;
/// Editor-style camera speed in world units per second.
const FLYCAM_SPEED: f32 = 8.0;
/// Camera-local quaternion look angle applied for one pixel of cursor motion.
const FLYCAM_LOOK_SENSITIVITY: f32 = 0.002;
const PRESENTATION_CAMERA_POSITION: [f32; 3] = [0.0, 0.0, -5.0];
const PRESENTATION_CAMERA_TARGET: [f32; 3] = [0.0; 3];
const PRESENTATION_CAMERA_WORLD_UP: [f32; 3] = [0.0, -1.0, 0.0];
const SHIP_INSTANCE_COUNT: usize = 3;
const SHIP_SCALE: f32 = 0.65;
const SHIP_BIRD_HEIGHT: f32 = 38.0;
const HID_KEY_TAB: u8 = 0x2b;
const FLYCAM_KEYS: [u8; 6] = [0x04, 0x07, 0x08, 0x14, 0x16, 0x1a]; // A/D/E/Q/S/W
/// Initial Blender-style editor camera for the retained scene.
fn presentation_camera() -> Camera {
    Camera {
        // Sit on the world-blue axis and keep the origin centred as the aim
        // point. The negative Z placement gives the blue guide a forward
        // direction through the scene.
        position: PRESENTATION_CAMERA_POSITION,
        // Preserve the aim point, but roll the complete view 180°.
        rotation: look_at_camera_rotation(
            PRESENTATION_CAMERA_POSITION,
            PRESENTATION_CAMERA_TARGET,
            PRESENTATION_CAMERA_WORLD_UP,
        ),
        projection: Projection::Perspective {
            yfov: core::f32::consts::FRAC_PI_3,
            znear: 0.1,
            zfar: Some(100.0),
            aspect_ratio: None,
        },
    }
}

/// Return the camera-to-world rotation whose local -Z axis targets `target`
/// and whose local +Y axis is the world-up closest valid basis direction.
fn look_at_camera_rotation(position: [f32; 3], target: [f32; 3], world_up: [f32; 3]) -> Quaternion {
    let forward = [
        target[0] - position[0],
        target[1] - position[1],
        target[2] - position[2],
    ];
    let forward_length =
        libm::sqrtf(forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]);
    let forward = [
        forward[0] / forward_length,
        forward[1] / forward_length,
        forward[2] / forward_length,
    ];
    // Local +X is camera right; local +Y completes the orthonormal frame.
    let right = [
        forward[1] * world_up[2] - forward[2] * world_up[1],
        forward[2] * world_up[0] - forward[0] * world_up[2],
        forward[0] * world_up[1] - forward[1] * world_up[0],
    ];
    let right_length = libm::sqrtf(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
    let right = [
        right[0] / right_length,
        right[1] / right_length,
        right[2] / right_length,
    ];
    let up = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ];
    // Matrix columns map local +X, +Y, +Z to world right, up, and -forward.
    let (m00, m01, m02) = (right[0], up[0], -forward[0]);
    let (m10, m11, m12) = (right[1], up[1], -forward[1]);
    let (m20, m21, m22) = (right[2], up[2], -forward[2]);
    let trace = m00 + m11 + m22;
    let rotation = if trace > 0.0 {
        let s = libm::sqrtf(trace + 1.0) * 2.0;
        Quaternion([(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s])
    } else if m00 > m11 && m00 > m22 {
        let s = libm::sqrtf(1.0 + m00 - m11 - m22) * 2.0;
        Quaternion([0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s])
    } else if m11 > m22 {
        let s = libm::sqrtf(1.0 + m11 - m00 - m22) * 2.0;
        Quaternion([(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s])
    } else {
        let s = libm::sqrtf(1.0 + m22 - m00 - m11) * 2.0;
        Quaternion([(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s])
    };
    rotation.normalized()
}

/// Ten independent cubic Bézier lanes across a medium X/Z water plane.
/// The first and last controls join, so each vessel loops without a teleport.
const SHIP_PATHS: [[[f32; 3]; 4]; SHIP_INSTANCE_COUNT] = [
    [[-20.0, 0.0, -12.0], [-7.0, 0.0, -26.0], [13.0, 0.0, -24.0], [21.0, 0.0, -10.0]],
    [[21.0, 0.0, -10.0], [29.0, 0.0, 5.0], [21.0, 0.0, 20.0], [7.0, 0.0, 23.0]],
    [[7.0, 0.0, 23.0], [-9.0, 0.0, 30.0], [-25.0, 0.0, 17.0], [-24.0, 0.0, 2.0]],
];

fn bezier_point(path: [[f32; 3]; 4], t: f32) -> [f32; 3] {
    let u = 1.0 - t;
    core::array::from_fn(|axis| u * u * u * path[0][axis]
        + 3.0 * u * u * t * path[1][axis]
        + 3.0 * u * t * t * path[2][axis]
        + t * t * t * path[3][axis])
}

fn bezier_tangent(path: [[f32; 3]; 4], t: f32) -> [f32; 3] {
    let u = 1.0 - t;
    core::array::from_fn(|axis| 3.0 * u * u * (path[1][axis] - path[0][axis])
        + 6.0 * u * t * (path[2][axis] - path[1][axis])
        + 3.0 * t * t * (path[3][axis] - path[2][axis]))
}

fn ship_pose(slot: usize, elapsed_millis: u64) -> ([f32; 3], [f32; 4]) {
    let t = ((elapsed_millis as f32 * 0.000018) + slot as f32 / SHIP_INSTANCE_COUNT as f32) % 1.0;
    let position = bezier_point(SHIP_PATHS[slot], t);
    let tangent = bezier_tangent(SHIP_PATHS[slot], t);
    let yaw = libm::atan2f(tangent[0], tangent[2]);
    (position, Quaternion::from_axis_angle([0.0, 1.0, 0.0], yaw).0)
}

fn ship_seeds(elapsed_millis: u64) -> [RetainedTransformSeed; SHIP_INSTANCE_COUNT] {
    core::array::from_fn(|slot| {
        let (translation, rotation) = ship_pose(slot, elapsed_millis);
        RetainedTransformSeed {
            translation,
            scale: [SHIP_SCALE; 3],
            rotation,
            local_radius: 3.0,
            previous_translation: translation,
            draw_group: 0,
            flags: (slot as u32) << 16,
        }
    })
}

fn retained_seed_bytes(seeds: &[RetainedTransformSeed]) -> &[u8] {
    // RetainedTransformSeed is a C-layout vGPU ABI row; this only exposes its
    // already-initialised bytes for the dynamic scene buffer upload.
    unsafe {
        core::slice::from_raw_parts(
            seeds.as_ptr().cast::<u8>(),
            core::mem::size_of_val(seeds),
        )
    }
}

fn bird_camera(slot: usize, elapsed_millis: u64) -> Camera {
    let (ship, _) = ship_pose(slot, elapsed_millis);
    let position = [ship[0], -SHIP_BIRD_HEIGHT, ship[2] - 2.0];
    Camera {
        position,
        rotation: look_at_camera_rotation(position, ship, PRESENTATION_CAMERA_WORLD_UP),
        projection: Projection::Perspective {
            yfov: core::f32::consts::FRAC_PI_3,
            znear: 0.1,
            zfar: Some(150.0),
            aspect_ratio: None,
        },
    }
}

const fn placed_head(x: f32, y: f32) -> TransformValue {
    TransformValue {
        translation: [x, y, 0.0],
        translation_pad: 0.0,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [HELMET_SCALE; 3],
        scale_pad: 0.0,
    }
}

/// Four GPU-authored transform programs over one immutable helmet mesh.
///
/// The CPU publishes this compact program once. GPU time advances it forever:
/// clockwise, counter-clockwise, 0.5-second collapse/expand, and identity.
pub const HEAD_INSTANCE_PROGRAMS: [HeadInstanceProgram; HEAD_INSTANCE_COUNT as usize] = [
    HeadInstanceProgram {
        initial: placed_head(-HEAD_PLANE_OFFSET, HEAD_PLANE_OFFSET),
        angular_velocity_z: -core::f32::consts::FRAC_PI_2,
        scale_half_period_seconds: 0.0,
        reserved: [0; 2],
    },
    HeadInstanceProgram {
        initial: placed_head(HEAD_PLANE_OFFSET, HEAD_PLANE_OFFSET),
        angular_velocity_z: core::f32::consts::FRAC_PI_2,
        scale_half_period_seconds: 0.0,
        reserved: [0; 2],
    },
    HeadInstanceProgram {
        initial: placed_head(-HEAD_PLANE_OFFSET, -HEAD_PLANE_OFFSET),
        angular_velocity_z: 0.0,
        scale_half_period_seconds: 0.5,
        reserved: [0; 2],
    },
    HeadInstanceProgram {
        initial: placed_head(HEAD_PLANE_OFFSET, -HEAD_PLANE_OFFSET),
        angular_velocity_z: 0.0,
        scale_half_period_seconds: 0.0,
        reserved: [0; 2],
    },
];

/// Bind the mesh's four local references to one restored/mapped shared state
/// table. Neither this descriptor nor Picasso observes its GPU virtual address.
pub const fn head_transform_refs(
    state_resource: SharedResourceId,
    generation: u64,
) -> TransformRefList {
    TransformRefList {
        states: TransformStateRange {
            resource: state_resource,
            offset: 0,
            byte_length: HEAD_TRANSFORM_STATE_BYTES,
            state_count: HEAD_INSTANCE_COUNT,
            state_stride: core::mem::size_of::<TransformValue>() as u32,
            generation,
        },
        references: PreparedRange {
            resource: HELMET_TRANSFORM_REFS,
            offset: 0,
            byte_length: HELMET_TRANSFORM_REFS_U32.len() as u64,
            revision: 1,
        },
        reference_count: HEAD_INSTANCE_COUNT,
    }
}

/// One live, end-to-end ship submission kept resident for UI4 display.
///
/// Picasso contributes only logical resource ranges. This TRUEOS adapter maps
/// those ranges to opaque vGPU buffers; no GPU virtual address crosses it.
pub struct GeometryProbe {
    frame: Frame,
    device: Device,
    queue: Queue,
    asset_vertex_buffers: [Option<Buffer>; ASSET_COUNT],
    asset_index_buffers: [Option<Buffer>; ASSET_COUNT],
    material_textures: [ResidentMaterial; ASSET_COUNT],
    material_parameters: [RetainedMaterialParameters; ASSET_COUNT],
    retained_meshes: [Option<RetainedMesh>; ASSET_COUNT],
    ship_seed_buffer: Buffer,
    flycam: FlyCam,
    camera_mode: CameraMode,
    tab_down: bool,
    resize_drag: Option<ResizeDrag>,
    timeline: u64,
    previous_elapsed_millis: u64,
    previous_view_projection: [f32; 16],
    deferred_frames: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CameraMode {
    Fly,
    BirdFollow(usize),
}

#[derive(Default)]
struct ResidentMaterial {
    base_color: Option<trueos::vmedia::RetainedTexture>,
    metallic_roughness: Option<trueos::vmedia::RetainedTexture>,
    emissive: Option<trueos::vmedia::RetainedTexture>,
    occlusion: Option<trueos::vmedia::RetainedTexture>,
    normal: Option<trueos::vmedia::RetainedTexture>,
}

#[derive(Clone, Copy)]
struct ResizeDrag {
    source: CursorSource,
    anchor_x: u32,
    anchor_y: u32,
    width: u32,
    height: u32,
}

impl GeometryProbe {
    fn open(catalog: &DatabasePreparedCatalog) -> Result<Self, GeometryProbeError> {
        // 784×441 is exact 16:9 and 1.5× the former 640×360 surface area.
        // This remains an ordinary positioned UI4 frame, not fullscreen.
        const WIDTH: u32 = 784;
        const HEIGHT: u32 = 441;

        if catalog.assets.len() != ASSET_COUNT {
            return Err(GeometryProbeError::Contract);
        }

        let primitive = prepared_geometry();
        let indices = primitive.indices.ok_or(GeometryProbeError::Contract)?;
        if primitive.vertex_stride != ASSETS[0].vertex_stride as u32
            || primitive.vertex_count == 0
            || primitive.index_count == 0
            || primitive.index_format != Some(trueos_picasso::IndexFormat::Uint32)
            || primitive.vertices.byte_length != HELMET_VERTEX_BYTES
            || indices.byte_length != HELMET_INDICES_U32.len() as u64
        {
            return Err(GeometryProbeError::Contract);
        }

        let (initial_x, initial_y) = output_dimensions()
            .map(|(output_width, output_height)| {
                (
                    i32::try_from(output_width.saturating_sub(WIDTH) / 2).unwrap_or(0),
                    i32::try_from(output_height.saturating_sub(HEIGHT) / 2).unwrap_or(0),
                )
            })
            // Positioning is optional presentation polish; preserve the old
            // request if UI4 cannot report an output extent during startup.
            .unwrap_or((120, 96));
        let frame = Frame::open_streaming(initial_x, initial_y, WIDTH, HEIGHT)
            .map_err(|error| GeometryProbeError::Ui4("frame-open", error))?;
        let device = Device::open(Capabilities::DEFAULT.union(Capabilities::PRESENT))
            .map_err(|code| GeometryProbeError::Vgpu("device-open", code))?;
        let queue = device
            .create_queue(QueueClass::Render)
            .map_err(|code| GeometryProbeError::Vgpu("queue-create", code))?;
        let mut asset_vertex_buffers = [None; ASSET_COUNT];
        let mut asset_index_buffers = [None; ASSET_COUNT];
        let mut retained_meshes = [None; ASSET_COUNT];
        for slot in 0..ASSET_COUNT {
            let asset = &ASSETS[slot];
            let runtime_asset = &catalog.assets[slot];
            if runtime_asset.vertices != asset.vertices
                || runtime_asset.indices != asset.indices
                || !material_matches(&runtime_asset.material, asset.material)
            {
                return Err(GeometryProbeError::Contract);
            }
            let vertex_buffer = device
                .create_buffer(
                    runtime_asset.vertices.len(),
                    BUFFER_USAGE_MAP_WRITE | BUFFER_USAGE_VERTEX,
                )
                .map_err(|code| GeometryProbeError::Vgpu("asset-vertex-buffer-create", code))?;
            let index_buffer = device
                .create_buffer(
                    runtime_asset.indices.len(),
                    BUFFER_USAGE_MAP_WRITE | BUFFER_USAGE_INDEX,
                )
                .map_err(|code| GeometryProbeError::Vgpu("asset-index-buffer-create", code))?;
            write_exact(device, vertex_buffer, 0, &runtime_asset.vertices)
                .map_err(|code| GeometryProbeError::Vgpu("asset-vertex-upload", code))?;
            write_exact(device, index_buffer, 0, &runtime_asset.indices)
                .map_err(|code| GeometryProbeError::Vgpu("asset-index-upload", code))?;
            let retained_mesh = device
                .create_retained_mesh(
                    vertex_buffer,
                    index_buffer,
                    RetainedMeshDescriptor {
                        vertex_count: asset.vertex_count,
                        index_count: asset.index_count,
                        vertex_layout: if asset.sampled_material {
                            RETAINED_VERTEX_LAYOUT_POS_NORMAL_UV_TANGENT
                        } else {
                            RETAINED_VERTEX_LAYOUT_POS_NORMAL
                        },
                        topology: vgpu_topology(
                            asset
                                .retained_topology
                                .ok_or(GeometryProbeError::Contract)?,
                        ) | if asset.retained_double_sided {
                            trueos::vgpu::RETAINED_MESH_FLAG_DOUBLE_SIDED
                        } else {
                            0
                        },
                        ..RetainedMeshDescriptor::default()
                    },
                )
                .map_err(|code| GeometryProbeError::Vgpu("retained-mesh-create", code))?;
            asset_vertex_buffers[slot] = Some(vertex_buffer);
            asset_index_buffers[slot] = Some(index_buffer);
            retained_meshes[slot] = Some(retained_mesh);
        }
        let ship_seed_buffer = device
            .create_buffer(
                SHIP_INSTANCE_COUNT * core::mem::size_of::<RetainedTransformSeed>(),
                BUFFER_USAGE_MAP_READ | BUFFER_USAGE_MAP_WRITE,
            )
            .map_err(|code| GeometryProbeError::Vgpu("ship-seed-buffer-create", code))?;
        let camera = presentation_camera();
        let previous_view_projection =
            retained_camera(camera, WIDTH, HEIGHT, [0.0; 16]).view_projection;

        let mut probe = Self {
            frame,
            device,
            queue,
            asset_vertex_buffers,
            asset_index_buffers,
            material_textures: core::array::from_fn(|_| ResidentMaterial::default()),
            material_parameters: core::array::from_fn(|slot| {
                catalog.assets[slot].material.parameters
            }),
            retained_meshes,
            ship_seed_buffer,
            flycam: FlyCam::new(camera, FLYCAM_SPEED),
            camera_mode: CameraMode::Fly,
            tab_down: false,
            resize_drag: None,
            timeline: 0,
            previous_elapsed_millis: 0,
            previous_view_projection,
            deferred_frames: 0,
        };
        // Move the Blender-style editor camera, never the world objects.
        probe.flycam.set_look_sensitivity(FLYCAM_LOOK_SENSITIVITY);
        for (slot, asset) in ASSETS.iter().enumerate() {
            let runtime_material = &catalog.assets[slot].material;
            // Keep the bundle local until every requested decode has reached
            // the owner-scoped Render1 carrier. Any error drops the completed
            // members and therefore releases the partial admission.
            let material = ResidentMaterial {
                base_color: decode_material_texture(
                    probe.device,
                    asset.name,
                    "base-color",
                    asset.material.base_color,
                    &runtime_material.base_color,
                )?,
                metallic_roughness: decode_material_texture(
                    probe.device,
                    asset.name,
                    "metallic-roughness",
                    asset.material.metallic_roughness,
                    &runtime_material.metallic_roughness,
                )?,
                emissive: decode_material_texture(
                    probe.device,
                    asset.name,
                    "emissive",
                    asset.material.emissive,
                    &runtime_material.emissive,
                )?,
                occlusion: decode_material_texture(
                    probe.device,
                    asset.name,
                    "occlusion",
                    asset.material.occlusion,
                    &runtime_material.occlusion,
                )?,
                normal: decode_material_texture(
                    probe.device,
                    asset.name,
                    "normal",
                    asset.material.normal,
                    &runtime_material.normal,
                )?,
            };
            if asset.sampled_material && material.base_color.is_none() {
                return Err(GeometryProbeError::Contract);
            }
            probe.material_textures[slot] = material;
        }
        while !probe.render_frame(0)? {
            vsys::poll_once();
            vsys::sleep_ms(16);
        }
        Ok(probe)
    }

    /// Advance input and return whether a complete frame was rendered. A busy
    /// back buffer defers this frame without submitting or losing the last view.
    pub fn render_frame(&mut self, elapsed_millis: u64) -> Result<bool, GeometryProbeError> {
        let delta_seconds =
            elapsed_millis.saturating_sub(self.previous_elapsed_millis) as f32 * 0.001;
        self.previous_elapsed_millis = elapsed_millis;
        self.service_camera_mode(elapsed_millis)?;
        self.flycam
            .step_ui4(&self.frame, delta_seconds)
            .map_err(|error| GeometryProbeError::Ui4("flycam-ui4", error))?;
        self.service_frame_interaction()?;
        let width = self.frame.width();
        let height = self.frame.height();
        let camera = retained_camera(
            self.flycam.camera,
            width,
            height,
            self.previous_view_projection,
        );

        let admission = self.frame.begin_gpu_frame();
        let rendered = render_admitted_frame(admission, || {
            let surface = self
                .device
                .acquire_ui4_surface(self.frame.window_id())
                .map_err(|code| GeometryProbeError::Vgpu("surface-acquire", code))?;
            let mesh = self.retained_meshes[0].ok_or(GeometryProbeError::Contract)?;
            let vertices =
                self.asset_vertex_buffers[0].ok_or(GeometryProbeError::Contract)?;
            let indices =
                self.asset_index_buffers[0].ok_or(GeometryProbeError::Contract)?;
            let seeds = ship_seeds(elapsed_millis);
            write_exact(self.device, self.ship_seed_buffer, 0, retained_seed_bytes(&seeds))
                .map_err(|code| GeometryProbeError::Vgpu("ship-seed-upload", code))?;
            let submit = RetainedFrameSubmit {
                camera,
                material: RetainedMaterial {
                    textures: retained_material_texture_ids(
                        &self.material_textures[0],
                        ASSETS[0].sampled_material,
                    ),
                    // V2 carries every scalar in material_parameters;
                    // the legacy scalar stays zero to avoid ambiguity.
                    emissive_factor: [0.0; 3],
                    ..RetainedMaterial::default()
                },
                clear_rgba8_srgb: u32::from_le_bytes([0, 128, 0, 0]),
                ..RetainedFrameSubmit::default()
            };
            let point = self.device.submit_retained_frame_v3(
                self.queue,
                surface,
                mesh,
                vertices,
                indices,
                RetainedFrameSubmitV3 {
                    frame: RetainedFrameSubmitV2 {
                        frame: submit,
                        material_parameters: self.material_parameters[0],
                    },
                    seed_buffer: self.ship_seed_buffer.raw(),
                    seed_offset: 0,
                    seed_count: SHIP_INSTANCE_COUNT as u32,
                    draw_count: 1,
                    draws: [
                        RetainedDrawRange { first_index: 0, index_count: ASSETS[0].index_count },
                        RetainedDrawRange::default(),
                        RetainedDrawRange::default(),
                        RetainedDrawRange::default(),
                    ],
                    reserved: [0; 2],
                },
            )
            .map_err(|code| GeometryProbeError::Vgpu("retained-frame-submit", code))?;
            self.device
                .wait(self.queue, point.value)
                .map_err(|code| GeometryProbeError::Vgpu("timeline-wait", code))?;
            self.frame
                .publish(Damage::full(width, height))
                .map_err(|error| GeometryProbeError::Ui4("frame-publish", error))?;
            self.timeline = point.value;
            self.previous_view_projection = camera.view_projection;
            Ok(())
        })?;
        if rendered {
            if self.deferred_frames != 0 {
                logl::log(
                    level::INFO,
                    format_args!(
                        "Boat: frame buffer resumed deferred_frames={} timeline={}",
                        self.deferred_frames, self.timeline,
                    ),
                );
                self.deferred_frames = 0;
            }
        } else {
            self.deferred_frames = self.deferred_frames.saturating_add(1);
            if self.deferred_frames == 1 || self.deferred_frames % 120 == 0 {
                logl::log(
                    level::INFO,
                    format_args!(
                        "Boat: frame buffer busy; frame deferred count={} timeline={}",
                        self.deferred_frames, self.timeline,
                    ),
                );
            }
        }
        Ok(rendered)
    }

    /// Tab cycles a world-following bird's-eye view. Any normal flycam key
    /// hands control back immediately, including the same key press that
    /// begins the manual movement on this frame.
    fn service_camera_mode(&mut self, elapsed_millis: u64) -> Result<(), GeometryProbeError> {
        let keyboard = self
            .frame
            .keyboard_state()
            .map_err(|error| GeometryProbeError::Ui4("fleet-camera-hotkeys", error))?;
        let tab_down = keyboard.is_some_and(|state| state.is_down(HID_KEY_TAB));
        let fly_key_down = keyboard.is_some_and(|state| FLYCAM_KEYS.iter().any(|key| state.is_down(*key)));
        if fly_key_down {
            self.camera_mode = CameraMode::Fly;
        } else if tab_down && !self.tab_down {
            self.camera_mode = match self.camera_mode {
                CameraMode::Fly => CameraMode::BirdFollow(0),
                CameraMode::BirdFollow(slot) => CameraMode::BirdFollow((slot + 1) % SHIP_INSTANCE_COUNT),
            };
        }
        self.tab_down = tab_down;
        if let CameraMode::BirdFollow(slot) = self.camera_mode {
            self.flycam.camera = bird_camera(slot, elapsed_millis);
        }
        Ok(())
    }

    /// Consume UI4's one-shot maximize/restore extent and the selected
    /// frame's cursor-sized bottom-right resize grip. The grip has no pixels;
    /// it is only a hit region fully contained in the application frame.
    fn service_frame_interaction(&mut self) -> Result<(), GeometryProbeError> {
        const RESIZE_GRIP_PX: i32 = 16;
        const MIN_WIDTH: u32 = 160;
        const MIN_HEIGHT: u32 = 90;

        let mut requested_extent = None;
        while let Some(event) = self
            .frame
            .take_resize_event()
            .map_err(|error| GeometryProbeError::Ui4("resize-event", error))?
        {
            requested_extent = Some((event.width, event.height));
        }
        if let Some((width, height)) = requested_extent
            && (width != self.frame.width() || height != self.frame.height())
        {
            self.frame
                .resize(width, height)
                .map_err(|error| GeometryProbeError::Ui4("maximize-resize", error))?;
        }

        let (output_width, output_height) =
            output_dimensions().map_err(|error| GeometryProbeError::Ui4("output-extent", error))?;
        let mut drag = self.resize_drag;
        let mut live_extent = None;
        while let Some(event) = self
            .frame
            .take_pointer_event()
            .map_err(|error| GeometryProbeError::Ui4("pointer-event", error))?
        {
            let in_grip = event.local_x >= self.frame.width() as i32 - RESIZE_GRIP_PX
                && event.local_y >= self.frame.height() as i32 - RESIZE_GRIP_PX
                && event.local_x < self.frame.width() as i32
                && event.local_y < self.frame.height() as i32;
            let owns_drag = drag.is_some_and(|active| active.source == event.source);
            // The app owns bottom-right resizing. Everything else remains
            // available to Picasso through the same UI4-routed event; nothing
            // is drained by the library behind the application's back.
            self.flycam
                .handle_ui4_pointer_event(&event, !in_grip && !owns_drag);
            self.frame
                .set_cursor_icon_for(
                    event.source,
                    if in_grip || owns_drag {
                        CursorIcon::ResizeDiagonal
                    } else {
                        CursorIcon::Default
                    },
                )
                .map_err(|error| GeometryProbeError::Ui4("resize-cursor", error))?;

            if event.buttons_pressed & POINTER_BUTTON_PRIMARY != 0 && in_grip {
                drag = Some(ResizeDrag {
                    source: event.source,
                    anchor_x: event.x,
                    anchor_y: event.y,
                    width: self.frame.width(),
                    height: self.frame.height(),
                });
            }
            if let Some(active) = drag
                && active.source == event.source
            {
                let width = (i64::from(active.width) + i64::from(event.x)
                    - i64::from(active.anchor_x))
                .clamp(i64::from(MIN_WIDTH), i64::from(output_width))
                    as u32;
                let height = (i64::from(active.height) + i64::from(event.y)
                    - i64::from(active.anchor_y))
                .clamp(i64::from(MIN_HEIGHT), i64::from(output_height))
                    as u32;
                live_extent = Some((width, height));
                if event.buttons_released & POINTER_BUTTON_PRIMARY != 0
                    || event.buttons_down & POINTER_BUTTON_PRIMARY == 0
                {
                    drag = None;
                }
            }
        }
        self.resize_drag = drag;
        if let Some((width, height)) = live_extent
            && (width != self.frame.width() || height != self.frame.height())
        {
            self.frame
                .resize(width, height)
                .map_err(|error| GeometryProbeError::Ui4("live-resize", error))?;
        }
        Ok(())
    }

    pub const fn timeline(&self) -> u64 {
        self.timeline
    }

    pub fn take_first_presentation(&mut self) -> Result<bool, GeometryProbeError> {
        self.frame
            .take_first_presentation()
            .map_err(|error| GeometryProbeError::Ui4("first-presentation", error))
    }
}

/// Build the shader's WGSL-compatible camera block.  Model seeds remain in
/// world space; this is the one normal `projection * view * model` path.
fn retained_camera(
    camera: Camera,
    viewport_width: u32,
    viewport_height: u32,
    previous_view_projection: [f32; 16],
) -> RetainedCamera {
    let [qx, qy, qz, qw] = camera.rotation.normalized().0;
    let world_to_view = Quaternion([-qx, -qy, -qz, qw]);
    let x_axis = world_to_view.rotate([1.0, 0.0, 0.0]);
    let y_axis = world_to_view.rotate([0.0, 1.0, 0.0]);
    let z_axis = world_to_view.rotate([0.0, 0.0, 1.0]);
    let translation = world_to_view.rotate([
        -camera.position[0],
        -camera.position[1],
        -camera.position[2],
    ]);
    // WGSL matrices are column-major. `world_to_view.rotate(e_i)` is column i
    // of the rotation, and the fourth column supplies camera translation.
    let view = [
        x_axis[0],
        x_axis[1],
        x_axis[2],
        0.0,
        y_axis[0],
        y_axis[1],
        y_axis[2],
        0.0,
        z_axis[0],
        z_axis[1],
        z_axis[2],
        0.0,
        translation[0],
        translation[1],
        translation[2],
        1.0,
    ];
    let (projection, znear, zfar) = match camera.projection {
        Projection::Perspective {
            yfov,
            znear,
            zfar,
            aspect_ratio,
        } => {
            let aspect = aspect_ratio
                .unwrap_or_else(|| viewport_width as f32 / viewport_height.max(1) as f32);
            let focal_y = 1.0 / libm::tanf(yfov * 0.5);
            let focal_x = focal_y / aspect.max(f32::EPSILON);
            let (depth_scale, depth_offset) = match zfar {
                Some(zfar) => (zfar / (znear - zfar), zfar * znear / (znear - zfar)),
                None => (-1.0, -znear),
            };
            (
                [
                    focal_x,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    focal_y,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    depth_scale,
                    -1.0,
                    0.0,
                    0.0,
                    depth_offset,
                    0.0,
                ],
                znear,
                zfar.unwrap_or(f32::MAX),
            )
        }
        Projection::Orthographic {
            xmag,
            ymag,
            znear,
            zfar,
        } => {
            let range = (zfar - znear).max(f32::EPSILON);
            (
                [
                    1.0 / xmag.max(f32::EPSILON),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0 / ymag.max(f32::EPSILON),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0 / range,
                    0.0,
                    0.0,
                    0.0,
                    -znear / range,
                    1.0,
                ],
                znear,
                zfar,
            )
        }
    };
    let view_projection = multiply_mat4(projection, view);
    let inverse_view_projection = invert_mat4(view_projection).unwrap_or(identity_mat4());
    let forward = camera.rotation.rotate([0.0, 0.0, -1.0]);
    RetainedCamera {
        view,
        projection,
        view_projection,
        inverse_view_projection,
        position_near: [
            camera.position[0],
            camera.position[1],
            camera.position[2],
            znear,
        ],
        forward_far: [forward[0], forward[1], forward[2], zfar],
        jitter_frame: [0.0; 4],
        previous_view_projection,
    }
}

const fn identity_mat4() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn multiply_mat4(left: [f32; 16], right: [f32; 16]) -> [f32; 16] {
    let mut product = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            product[column * 4 + row] = (0..4)
                .map(|inner| left[inner * 4 + row] * right[column * 4 + inner])
                .sum();
        }
    }
    product
}

fn invert_mat4(matrix: [f32; 16]) -> Option<[f32; 16]> {
    let mut augmented = [[0.0; 8]; 4];
    for row in 0..4 {
        for column in 0..4 {
            augmented[row][column] = matrix[column * 4 + row];
            augmented[row][column + 4] = if row == column { 1.0 } else { 0.0 };
        }
    }
    for pivot_column in 0..4 {
        let mut pivot_row = pivot_column;
        for candidate in pivot_column + 1..4 {
            if libm::fabsf(augmented[candidate][pivot_column])
                > libm::fabsf(augmented[pivot_row][pivot_column])
            {
                pivot_row = candidate;
            }
        }
        let pivot = augmented[pivot_row][pivot_column];
        if libm::fabsf(pivot) <= f32::EPSILON {
            return None;
        }
        augmented.swap(pivot_column, pivot_row);
        for value in &mut augmented[pivot_column] {
            *value /= pivot;
        }
        for row in 0..4 {
            if row == pivot_column {
                continue;
            }
            let factor = augmented[row][pivot_column];
            for column in 0..8 {
                augmented[row][column] -= factor * augmented[pivot_column][column];
            }
        }
    }
    let mut inverse = [0.0; 16];
    for row in 0..4 {
        for column in 0..4 {
            inverse[column * 4 + row] = augmented[row][column + 4];
        }
    }
    Some(inverse)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryProbeError {
    Contract,
    Ui4(&'static str, Ui4Error),
    Vgpu(&'static str, i32),
}

/// Backpressure is recoverable only before the frame write lease is acquired.
/// Once admitted, every rendering error retains its normal failure semantics.
fn render_admitted_frame(
    admission: Result<(), Ui4Error>,
    render: impl FnOnce() -> Result<(), GeometryProbeError>,
) -> Result<bool, GeometryProbeError> {
    match admission {
        Ok(()) => render().map(|()| true),
        Err(Ui4Error::Busy) => Ok(false),
        Err(error) => Err(GeometryProbeError::Ui4("frame-begin", error)),
    }
}

#[cfg(test)]
mod frame_admission_tests {
    use super::{GeometryProbeError, Ui4Error, render_admitted_frame};

    #[test]
    fn busy_frame_defers_submission_and_success_retires_once() {
        let mut timeline = 7;
        let mut previous_view = 3;
        for admission in [Err(Ui4Error::Busy), Err(Ui4Error::Busy), Ok(())] {
            let rendered = render_admitted_frame(admission, || {
                timeline += 1;
                previous_view = 4;
                Ok(())
            })
            .unwrap();
            assert_eq!(rendered, admission.is_ok());
            assert_eq!(timeline, if rendered { 8 } else { 7 });
            assert_eq!(previous_view, if rendered { 4 } else { 3 });
        }
    }

    #[test]
    fn other_admission_errors_remain_fatal_without_submission() {
        for error in [Ui4Error::NotFound, Ui4Error::InvalidState, Ui4Error::Ui4] {
            let result = render_admitted_frame(Err(error), || panic!("lease not acquired"));
            assert_eq!(result, Err(GeometryProbeError::Ui4("frame-begin", error)));
        }
    }

    #[test]
    fn errors_after_acquiring_a_lease_are_never_treated_as_deferral() {
        for error in [
            GeometryProbeError::Vgpu("retained-frame-submit", -16),
            GeometryProbeError::Ui4("frame-publish", Ui4Error::Busy),
        ] {
            assert_eq!(render_admitted_frame(Ok(()), || Err(error)), Err(error));
        }
    }
}

fn write_exact(device: Device, buffer: Buffer, offset: u64, bytes: &[u8]) -> Result<(), i32> {
    let offset = usize::try_from(offset).map_err(|_| trueos::vgpu::ERR_UNSUPPORTED)?;
    let written = device.write_buffer(buffer, offset, bytes)?;
    (written == bytes.len())
        .then_some(())
        .ok_or(trueos::vgpu::ERR_IO)
}

fn material_matches(runtime: &DatabasePreparedMaterial, prepared: PreparedMaterial) -> bool {
    runtime.base_color == prepared.base_color.bytes
        && runtime.metallic_roughness == prepared.metallic_roughness.bytes
        && runtime.emissive == prepared.emissive.bytes
        && runtime.occlusion == prepared.occlusion.bytes
        && runtime.normal == prepared.normal.bytes
        && runtime.parameters == prepared.parameters
}

fn retained_material_texture_ids(material: &ResidentMaterial, sampled: bool) -> [u64; 5] {
    if !sampled {
        return [0; 5];
    }
    [
        material
            .base_color
            .as_ref()
            .map_or(0, |texture| texture.id().raw()),
        material
            .metallic_roughness
            .as_ref()
            .map_or(0, |texture| texture.id().raw()),
        material
            .emissive
            .as_ref()
            .map_or(0, |texture| texture.id().raw()),
        material
            .occlusion
            .as_ref()
            .map_or(0, |texture| texture.id().raw()),
        material
            .normal
            .as_ref()
            .map_or(0, |texture| texture.id().raw()),
    ]
}

fn decode_material_texture(
    device: Device,
    asset_name: &str,
    role: &str,
    prepared: PreparedTexture,
    encoded: &[u8],
) -> Result<Option<trueos::vmedia::RetainedTexture>, GeometryProbeError> {
    if prepared.bytes.is_empty() {
        return encoded
            .is_empty()
            .then_some(None)
            .ok_or(GeometryProbeError::Contract);
    }
    if prepared.name.is_empty() || encoded.len() != prepared.bytes.len() {
        return Err(GeometryProbeError::Contract);
    }
    let texture = trueos::async_fs::block_on(trueos::vmedia::decode_retained_asset(
        device,
        prepared.name,
        encoded,
    ))
    .map_err(|code| {
        logl::log(
            level::ERROR,
            format_args!(
                "PicassoExample: material bundle accepted=0 asset={} role={} encoded_bytes={} error={} action=release-partial-bundle",
                asset_name,
                role,
                encoded.len(),
                code,
            ),
        );
        GeometryProbeError::Vgpu("material-texture-decode", code)
    })?;
    let info = texture.info();
    logl::log(
        level::INFO,
        format_args!(
            "PicassoExample: material bundle member accepted=1 asset={} role={} encoded_bytes={} texture_id=0x{:X} decoded={}x{} stride={} residency={:?} kernel_rgba_readback=0",
            asset_name,
            role,
            encoded.len(),
            info.id.raw(),
            info.width,
            info.height,
            info.stride_bytes,
            info.residency,
        ),
    );
    Ok(Some(texture))
}

/// TRUEOS-specific materialization adapter. Field order is intentional: the
/// ring is dropped before `memory`, so `VVideoMem` backs every live ring view.
pub struct VVideoRing {
    ring: ExecRing,
    memory: VVideoMem,
}

impl VVideoRing {
    /// Allocate and initialize a new page-pinned shared region.
    pub fn allocate_fresh(
        device: Device,
        resource: SharedResourceId,
        slot_count: usize,
        slot_stride: usize,
        usage: u32,
    ) -> Result<Self, VVideoRingError> {
        let total_bytes = slot_count
            .checked_mul(slot_stride)
            .ok_or(VVideoRingError::Cubism(CubismError::SizeOverflow))?;
        let memory = device
            .allocate_vvideo_mem(total_bytes, usage)
            .map_err(VVideoRingError::Vgpu)?;
        let ring = unsafe {
            ExecRing::from_raw_parts(
                memory.as_ptr() as *mut u8,
                memory.len(),
                resource,
                slot_count,
                slot_stride,
            )
        }
        .map_err(VVideoRingError::Cubism)?;
        unsafe { ring.initialize_fresh() };
        Ok(Self { ring, memory })
    }

    pub const fn ring(&self) -> &ExecRing {
        &self.ring
    }

    /// Opaque kernel buffer identity for submissions outside Picasso.
    pub const fn buffer(&self) -> Buffer {
        self.memory.buffer()
    }

    /// Form a bounds-checked buffer-relative range for an executor.
    pub fn slice(&self, offset: usize, bytes: usize) -> Result<BufferSlice, i32> {
        self.memory.slice(offset, bytes)
    }

    pub fn visibility(&self) -> VVideoVisibility<'_> {
        VVideoVisibility {
            memory: &self.memory,
            resource: self.ring.resource(),
        }
    }
}

/// Cache adapter for a `VVideoRing`. It uses `VVideoMem` offset operations;
/// the GPU address and underlying PPGTT mapping remain inside TRUEOS. Cubism
/// flushes slot data first, then the published header control line. It
/// invalidates just the GPU-owned payload after retirement has read the
/// CPU-owned header metadata.
pub struct VVideoVisibility<'a> {
    memory: &'a VVideoMem,
    resource: SharedResourceId,
}

impl VisibilityOps for VVideoVisibility<'_> {
    fn cpu_make_gpu_visible(&self, range: SharedByteRange) -> Result<(), CubismError> {
        self.maintain(range, true)
    }

    fn cpu_publish_slot(&self, range: SharedByteRange) -> Result<(), CubismError> {
        self.maintain(range, true)
    }

    fn gpu_make_cpu_visible(&self, range: SharedByteRange) -> Result<(), CubismError> {
        self.maintain(range, false)
    }
}

impl VVideoVisibility<'_> {
    fn maintain(&self, range: SharedByteRange, flush: bool) -> Result<(), CubismError> {
        if range.resource != self.resource {
            return Err(CubismError::VisibilityFailed);
        }
        let offset = usize::try_from(range.offset).map_err(|_| CubismError::VisibilityFailed)?;
        let bytes =
            usize::try_from(range.byte_length).map_err(|_| CubismError::VisibilityFailed)?;
        let result = if flush {
            self.memory.flush(offset, bytes)
        } else {
            self.memory.invalidate(offset, bytes)
        };
        result.map_err(|_| CubismError::VisibilityFailed)
    }
}

fn prepared_asset_key(asset: &str, object: &str) -> String {
    format!("{asset}/prepared/{object}")
}

/// Publish the build-prepared representation into the same Picasso-owned
/// runtime database as the exact source files. This is the packaging seam:
/// build-time glTF parsing remains off the Blueprint, while every byte used by
/// the renderer must cross Picasso's database read boundary first.
fn put_prepared_assets(picasso: &Picasso) -> Result<(), trueos_picasso::PicassoError> {
    for asset in &ASSETS {
        picasso.put_embedded_asset(
            &prepared_asset_key(asset.name, "mesh/vertices"),
            asset.vertices,
        )?;
        picasso.put_embedded_asset(
            &prepared_asset_key(asset.name, "mesh/indices"),
            asset.indices,
        )?;
        for (role, texture) in [
            ("base-color", asset.material.base_color),
            ("metallic-roughness", asset.material.metallic_roughness),
            ("emissive", asset.material.emissive),
            ("occlusion", asset.material.occlusion),
            ("normal", asset.material.normal),
        ] {
            picasso.put_embedded_asset(
                &prepared_asset_key(asset.name, &format!("material/{role}")),
                texture.bytes,
            )?;
        }
        let parameters = material_parameter_bytes(asset.material.parameters);
        picasso.put_embedded_asset(
            &prepared_asset_key(asset.name, "material/parameters.f32le"),
            &parameters,
        )?;
    }
    Ok(())
}

fn material_parameter_bytes(parameters: RetainedMaterialParameters) -> [u8; 64] {
    let mut bytes = [0; 64];
    let values = parameters
        .base_color_factor
        .into_iter()
        .chain(parameters.emissive_factor)
        .chain([
            parameters.normal_scale,
            parameters.metallic_factor,
            parameters.roughness_factor,
            parameters.occlusion_strength,
            parameters.alpha_cutoff,
        ]);
    for (component, value) in values.enumerate() {
        let offset = component * core::mem::size_of::<f32>();
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (component, value) in core::iter::once(parameters.flags)
        .chain(parameters.reserved)
        .enumerate()
    {
        let offset = 48 + component * core::mem::size_of::<u32>();
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn material_parameters_from_bytes(bytes: &[u8]) -> Option<RetainedMaterialParameters> {
    let bytes: &[u8; 64] = bytes.try_into().ok()?;
    let words: [u32; 16] = core::array::from_fn(|component| {
        let offset = component * 4;
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    });
    Some(RetainedMaterialParameters {
        base_color_factor: core::array::from_fn(|component| f32::from_bits(words[component])),
        emissive_factor: core::array::from_fn(|component| f32::from_bits(words[4 + component])),
        normal_scale: f32::from_bits(words[7]),
        metallic_factor: f32::from_bits(words[8]),
        roughness_factor: f32::from_bits(words[9]),
        occlusion_strength: f32::from_bits(words[10]),
        alpha_cutoff: f32::from_bits(words[11]),
        flags: words[12],
        reserved: [words[13], words[14], words[15]],
    })
}

#[cfg(test)]
mod material_parameter_tests {
    use super::{
        RetainedMaterialParameters, material_parameter_bytes, material_parameters_from_bytes,
    };

    #[test]
    fn distinct_material_parameters_survive_picasso_database_serialization() {
        let parameters = RetainedMaterialParameters {
            base_color_factor: [0.1, 0.2, 0.3, 0.4],
            emissive_factor: [0.5, 0.6, 0.7],
            normal_scale: 0.8,
            metallic_factor: 0.9,
            roughness_factor: 0.25,
            occlusion_strength: 0.75,
            alpha_cutoff: 0.45,
            flags: 4,
            reserved: [0; 3],
        };
        let picasso = trueos_picasso::Picasso::new().unwrap();
        let bytes = material_parameter_bytes(parameters);
        picasso
            .put_embedded_asset("material/parameters.f32le", &bytes)
            .unwrap();
        let restored = picasso
            .embedded_asset("material/parameters.f32le")
            .unwrap()
            .unwrap();
        assert_eq!(restored.len(), 64);
        assert_eq!(material_parameters_from_bytes(&restored), Some(parameters));
        assert!(material_parameters_from_bytes(&restored[..63]).is_none());
    }
}

fn load_prepared_assets(
    picasso: &Picasso,
) -> Result<Option<DatabasePreparedCatalog>, trueos_picasso::PicassoError> {
    let mut assets = Vec::with_capacity(ASSET_COUNT);
    for asset in &ASSETS {
        let Some(vertices) =
            picasso.embedded_asset(&prepared_asset_key(asset.name, "mesh/vertices"))?
        else {
            return Ok(None);
        };
        let Some(indices) =
            picasso.embedded_asset(&prepared_asset_key(asset.name, "mesh/indices"))?
        else {
            return Ok(None);
        };
        let mut material_bytes = Vec::with_capacity(5);
        for role in [
            "base-color",
            "metallic-roughness",
            "emissive",
            "occlusion",
            "normal",
        ] {
            let Some(bytes) = picasso
                .embedded_asset(&prepared_asset_key(asset.name, &format!("material/{role}")))?
            else {
                return Ok(None);
            };
            material_bytes.push(bytes);
        }
        let Some(parameter_bytes) =
            picasso.embedded_asset(&prepared_asset_key(asset.name, "material/parameters.f32le"))?
        else {
            return Ok(None);
        };
        let Some(parameters) = material_parameters_from_bytes(&parameter_bytes) else {
            return Ok(None);
        };
        assets.push(DatabasePreparedAsset {
            vertices,
            indices,
            material: DatabasePreparedMaterial {
                base_color: material_bytes.remove(0),
                metallic_roughness: material_bytes.remove(0),
                emissive: material_bytes.remove(0),
                occlusion: material_bytes.remove(0),
                normal: material_bytes.remove(0),
                parameters,
            },
        });
    }
    Ok(Some(DatabasePreparedCatalog { assets }))
}

fn main() {
    run();
    if !trueos::vshell::shutdown_current_blueprint(
        "Boat terminated after a fatal scene error",
    ) {
        logl::log(
            level::ERROR,
            "Boat: fatal scene return could not request Blueprint shutdown",
        );
    }
}

fn run() {
    // The example owns only the immutable source bytes. Picasso owns their
    // runtime representation behind this public asset-ingestion boundary.
    let picasso = match Picasso::new() {
        Ok(picasso) => picasso,
        Err(error) => {
            logl::log(
                level::ERROR,
                format_args!("Boat: Picasso creation failed: {error:?}"),
            );
            return;
        }
    };
    let mut source_bytes = 0usize;
    for asset in &demodata::ASSETS {
        if let Err(error) = picasso.put_embedded_asset(asset.name, asset.bytes) {
            logl::log(
                level::ERROR,
                format_args!(
                    "Boat: embedded asset rejected asset={} bytes={} error={error:?}",
                    asset.name,
                    asset.bytes.len(),
                ),
            );
            return;
        }
        match picasso.embedded_asset(asset.name) {
            Ok(Some(stored)) if stored == asset.bytes => {}
            Ok(_) => {
                logl::log(
                    level::ERROR,
                    format_args!(
                        "Boat: embedded source round-trip mismatch asset={}",
                        asset.name
                    ),
                );
                return;
            }
            Err(error) => {
                logl::log(
                    level::ERROR,
                    format_args!(
                        "Boat: embedded source read failed asset={} error={error:?}",
                        asset.name
                    ),
                );
                return;
            }
        }
        source_bytes += asset.bytes.len();
    }
    if let Err(error) = put_prepared_assets(&picasso) {
        logl::log(
            level::ERROR,
            format_args!("Boat: prepared asset rejected error={error:?}"),
        );
        return;
    }
    let runtime_catalog = match load_prepared_assets(&picasso) {
        Ok(Some(assets)) => assets,
        Ok(None) => {
            logl::log(
                level::ERROR,
                "Boat: prepared asset missing after Picasso database publication",
            );
            return;
        }
        Err(error) => {
            logl::log(
                level::ERROR,
                format_args!("Boat: prepared asset read failed error={error:?}"),
            );
            return;
        }
    };
    logl::log(
        level::INFO,
        format_args!(
            "Boat: database-backed ship ready assets={} exact_source_bytes={}",
            demodata::ASSETS.len(),
            source_bytes,
        ),
    );

    let mut probe = match GeometryProbe::open(&runtime_catalog) {
        Ok(probe) => probe,
        Err(error) => {
            logl::log(
                level::ERROR,
                format_args!("Boat: scene setup failed: {error:?}"),
            );
            return;
        }
    };

    logl::log(
        level::INFO,
        format_args!(
            "Boat: retained Ship material bundle submitted and retired: vertices={} indices={} base_color_bound={} emissive_resident={} metallic_roughness_resident={} occlusion_resident={} normal_resident={} timeline={}",
            HELMET_VERTEX_COUNT,
            HELMET_INDEX_COUNT,
            probe.material_textures[0].base_color.is_some() as u8,
            probe.material_textures[0].emissive.is_some() as u8,
            probe.material_textures[0].metallic_roughness.is_some() as u8,
            probe.material_textures[0].occlusion.is_some() as u8,
            probe.material_textures[0].normal.is_some() as u8,
            probe.timeline(),
        ),
    );
    logl::log(
        level::INFO,
        format_args!(
            "Boat: controls=Tab bird-follow-next; WASD+QE releases to flycam; middle_drag looks; ships={}",
            SHIP_INSTANCE_COUNT,
        ),
    );

    let start = clock::monotonic_millis();
    let mut presentation_logged = false;
    loop {
        vsys::poll_once();
        if let Err(error) = probe.render_frame(clock::monotonic_millis().saturating_sub(start)) {
            logl::log(
                level::ERROR,
                format_args!("Boat: retained animation failed: {error:?}"),
            );
            return;
        }
        if !presentation_logged {
            match probe.take_first_presentation() {
                Ok(true) => {
                    logl::log(
                        level::INFO,
                        "Boat: fleet frame crossed UI4 SURFLIVE",
                    );
                    presentation_logged = true;
                }
                Ok(false) => {}
                Err(error) => {
                    logl::log(
                        level::ERROR,
                        format_args!("Boat: presentation probe failed: {error:?}"),
                    );
                    return;
                }
            }
        }
        vsys::sleep_ms(16);
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use std::alloc::{Layout, alloc_zeroed, dealloc};

    struct Region {
        ptr: *mut u8,
        layout: Layout,
    }
    impl Region {
        fn new(bytes: usize) -> Self {
            let layout = Layout::from_size_align(bytes, 64).unwrap();
            let ptr = unsafe { alloc_zeroed(layout) };
            assert!(!ptr.is_null());
            Self { ptr, layout }
        }
    }
    impl Drop for Region {
        fn drop(&mut self) {
            unsafe { dealloc(self.ptr, self.layout) }
        }
    }

    #[test]
    fn prepared_primitives_keep_source_ranges_and_topology() {
        for asset in ASSETS {
            assert_eq!(
                asset.sampled_material,
                ENABLE_SAMPLED_MATERIAL && !asset.material.base_color.bytes.is_empty()
            );
            assert_eq!(
                asset.vertex_stride,
                if asset.sampled_material { 48 } else { 24 }
            );
            assert!(!asset.primitives.is_empty());
            for primitive in asset.primitives {
                assert!(primitive.first_vertex < asset.vertex_count);
                assert!(primitive.vertex_count > 0);
                assert!(primitive.first_vertex + primitive.vertex_count <= asset.vertex_count);
                assert!(primitive.first_index < asset.index_count);
                assert!(primitive.index_count > 0);
                assert!(primitive.first_index + primitive.index_count <= asset.index_count);
            }
        }
    }

    #[test]
    fn no_std_consumer_uses_resource_relative_ranges() {
        const SLOT_COUNT: usize = 1;
        const SLOT_STRIDE: usize = 128;
        let region = Region::new(SLOT_COUNT * SLOT_STRIDE);
        let ring = unsafe {
            ExecRing::from_raw_parts(
                region.ptr,
                SLOT_COUNT * SLOT_STRIDE,
                SharedResourceId(9),
                SLOT_COUNT,
                SLOT_STRIDE,
            )
            .unwrap()
        };
        unsafe { ring.initialize_fresh() };
        let visibility = trueos_picasso::CoherentVisibility;
        let mut write = ring.try_acquire().unwrap();
        let index = write.slot_index();
        let generation = write.generation();
        write.payload_mut()[..4].copy_from_slice(b"draw");
        let sealed = write.publish(4, 1, &visibility).unwrap();
        assert_eq!(sealed.payload_range().offset, 64);
        sealed.mark_in_flight(17).unwrap();
        ring.retire(index, generation, 17, &visibility).unwrap();
        let primitive = prepared_geometry();
        assert!(primitive.vertex_count > 3);
        assert!(primitive.index_count > 3);
        assert_eq!(
            primitive.index_format,
            Some(trueos_picasso::IndexFormat::Uint32)
        );
        assert_eq!(
            primitive.vertices.byte_length,
            HELMET_POSNORMAL.len() as u64
        );
        assert_eq!(
            primitive.indices.unwrap().byte_length,
            HELMET_INDICES_U32.len() as u64
        );
    }

    #[test]
    fn four_heads_share_geometry_and_reference_retained_gpu_state() {
        let primitive = prepared_geometry();
        let refs = head_transform_refs(SharedResourceId(21), 7);
        assert_eq!(refs.reference_count, 4);
        assert_eq!(refs.states.state_count, 4);
        assert_eq!(refs.states.generation, 7);
        assert_eq!(primitive.vertices.resource, HELMET_VERTICES);
        assert_eq!(primitive.indices.unwrap().resource, HELMET_INDICES);
        assert_eq!(HELMET_TRANSFORM_REFS_U32.len(), 16);
        assert!(
            HEAD_INSTANCE_PROGRAMS
                .iter()
                .all(|program| program.initial.is_valid())
        );
        assert!(HEAD_INSTANCE_PROGRAMS[0].angular_velocity_z < 0.0);
        assert!(HEAD_INSTANCE_PROGRAMS[1].angular_velocity_z > 0.0);
        assert_eq!(HEAD_INSTANCE_PROGRAMS[2].scale_half_period_seconds, 0.5);
        assert_eq!(HEAD_INSTANCE_PROGRAMS[3].angular_velocity_z, 0.0);
        assert_eq!(HEAD_INSTANCE_PROGRAMS[3].scale_half_period_seconds, 0.0);
        assert_eq!(
            HEAD_WORLD_TRANSLATIONS,
            [
                [-HEAD_PLANE_OFFSET, HEAD_PLANE_OFFSET, 0.0],
                [HEAD_PLANE_OFFSET, HEAD_PLANE_OFFSET, 0.0],
                [-HEAD_PLANE_OFFSET, -HEAD_PLANE_OFFSET, 0.0],
                [HEAD_PLANE_OFFSET, -HEAD_PLANE_OFFSET, 0.0],
            ]
        );
        for (program, translation) in HEAD_INSTANCE_PROGRAMS.iter().zip(HEAD_WORLD_TRANSLATIONS) {
            assert_eq!(program.initial.translation, translation);
            assert_eq!(translation[2], 0.0);
        }
    }

    #[test]
    fn presentation_camera_stays_on_blue_axis_and_looks_at_origin() {
        let camera = presentation_camera();
        assert_eq!(camera.position, PRESENTATION_CAMERA_POSITION);
        assert_eq!(camera.position[0], 0.0);
        assert_eq!(camera.position[1], 0.0);
        let forward = camera.rotation.rotate([0.0, 0.0, -1.0]);
        let expected_forward = [0.0, 0.0, 1.0];
        for axis in 0..3 {
            assert!((forward[axis] - expected_forward[axis]).abs() < 1.0e-5);
        }
        let distance_to_origin = [
            PRESENTATION_CAMERA_TARGET[0] - camera.position[0],
            PRESENTATION_CAMERA_TARGET[1] - camera.position[1],
            PRESENTATION_CAMERA_TARGET[2] - camera.position[2],
        ];
        assert!(distance_to_origin[0] == 0.0 && distance_to_origin[1] == 0.0);
        assert!(distance_to_origin[2] > 0.0);
        assert!(camera.rotation.rotate([0.0, 1.0, 0.0])[1] < 0.0);
        assert_eq!(
            camera.projection,
            Projection::Perspective {
                yfov: core::f32::consts::FRAC_PI_3,
                znear: 0.1,
                zfar: Some(100.0),
                aspect_ratio: None,
            }
        );

        let seeds = retained_seeds(0, true);
        assert!(seeds[..HEAD_INSTANCE_COUNT as usize].iter().all(|seed| {
            seed.translation
                .iter()
                .all(|coordinate| coordinate.is_finite())
        }));
        assert!(
            seeds[..HEAD_INSTANCE_COUNT as usize]
                .iter()
                .all(|seed| seed.scale == [HELMET_SCALE; 3])
        );
        assert!(
            seeds[..HEAD_INSTANCE_COUNT as usize]
                .iter()
                .all(|seed| seed.draw_group == 0)
        );
        assert!(
            seeds[..HEAD_INSTANCE_COUNT as usize]
                .iter()
                .enumerate()
                .all(|(slot, seed)| seed.flags == (slot as u32) << 16)
        );
    }

    #[test]
    fn editor_flycam_moves_only_the_camera() {
        let camera = presentation_camera();
        let mut flycam = FlyCam::new(camera, FLYCAM_SPEED);
        flycam.set_look_sensitivity(FLYCAM_LOOK_SENSITIVITY);
        let heads_before = HEAD_WORLD_TRANSLATIONS;
        let forward = camera.rotation.rotate([0.0, 0.0, -1.0]);

        flycam.step(
            trueos_picasso::cam::Wasd {
                w: true,
                ..Default::default()
            },
            1.0,
        );
        let position_after_wasd = flycam.camera.position;
        for axis in 0..3 {
            assert!(
                (position_after_wasd[axis]
                    - (camera.position[axis] + forward[axis] * FLYCAM_SPEED))
                    .abs()
                    < 1.0e-5
            );
        }
        flycam.look(32.0, -16.0);

        assert_ne!(flycam.camera.rotation, camera.rotation);
        assert_eq!(HEAD_WORLD_TRANSLATIONS, heads_before);
        assert_eq!(flycam.speed(), FLYCAM_SPEED);
        assert_eq!(flycam.look_sensitivity(), FLYCAM_LOOK_SENSITIVITY);
    }

    #[test]
    fn camera_near_plane_is_applied_by_the_gpu_projection_not_cpu_seed_hiding() {
        let mut camera = presentation_camera();
        camera.position = [0.0, 0.0, 3.0];

        let seeds = retained_seeds(0, true);
        for seed in &seeds[..HEAD_INSTANCE_COUNT as usize] {
            assert_ne!(seed.scale, [0.0; 3]);
        }
        let camera = retained_camera(camera, 640, 360, identity_mat4());
        assert!(camera.view_projection.iter().all(|value| value.is_finite()));
    }
}
