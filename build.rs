use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
};
const ASSETS: [(&str, &str); 1] = [("Ship", "Assets/Ship/mud.glb")];

// Sample authored material maps through the retained Intel renderer. Assets
// without material images retain their established position/normal layout.
const ENABLE_SAMPLED_MATERIAL: bool = true;

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let mut catalog = format!(
        "pub const ENABLE_SAMPLED_MATERIAL: bool = {ENABLE_SAMPLED_MATERIAL};\npub const ASSET_COUNT: usize = {};\npub static ASSETS: [PreparedAsset; ASSET_COUNT] = [\n",
        ASSETS.len(),
    );
    for (slot, (name, source)) in ASSETS.iter().enumerate() {
        println!("cargo:rerun-if-changed={source}");
        let source_path = Path::new(source);
        let PreparedGeometry {
            vertices,
            indices,
            material,
            vertex_stride,
            primitives: prepared_primitives,
        } = prepare(source_path, ENABLE_SAMPLED_MATERIAL);
        let stem = name.to_ascii_lowercase();
        let vertex_file = format!("{stem}.posnormal.f32le");
        let index_file = format!("{stem}.indices.u32le");
        fs::write(out.join(&vertex_file), &vertices).expect("write vertices");
        fs::write(out.join(&index_file), &indices).expect("write indices");
        let base_color = write_texture(
            out.as_path(),
            &stem,
            name,
            "base-color",
            material.base_color.as_ref(),
        );
        let metallic_roughness = write_texture(
            out.as_path(),
            &stem,
            name,
            "metallic-roughness",
            material.metallic_roughness.as_ref(),
        );
        let emissive = write_texture(
            out.as_path(),
            &stem,
            name,
            "emissive",
            material.emissive.as_ref(),
        );
        let occlusion = write_texture(
            out.as_path(),
            &stem,
            name,
            "occlusion",
            material.occlusion.as_ref(),
        );
        let normal = write_texture(
            out.as_path(),
            &stem,
            name,
            "normal",
            material.normal.as_ref(),
        );
        let primitives = prepared_primitives
            .iter()
            .map(|primitive| format!(
                "PreparedPrimitive {{ topology: PrimitiveTopology::{}, first_vertex: {}, vertex_count: {}, first_index: {}, index_count: {}, double_sided: {} }}",
                primitive.topology,
                primitive.first_vertex,
                primitive.vertex_count,
                primitive.first_index,
                primitive.index_count,
                primitive.double_sided,
            ))
            .collect::<Vec<_>>()
            .join(", ");
        let retained_topology = prepared_primitives
            .first()
            .filter(|first| {
                prepared_primitives
                    .iter()
                    .all(|primitive| primitive.topology == first.topology)
            })
            .map(|primitive| format!("Some(PrimitiveTopology::{})", primitive.topology))
            .unwrap_or_else(|| String::from("None"));
        // One retained mesh currently contains all source primitives. If their
        // cull policies differ, disable culling conservatively: that keeps
        // every glTF `doubleSided` face visible until material batching exists.
        let retained_double_sided = prepared_primitives
            .iter()
            .any(|primitive| primitive.double_sided);
        let sampled_material = ENABLE_SAMPLED_MATERIAL && material.base_color.is_some();
        catalog.push_str(&format!(
            "PreparedAsset {{ name: \"{name}\", vertices: include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{vertex_file}\")), indices: include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{index_file}\")), vertex_count: {vertex_count}, index_count: {index_count}, vertex_stride: {vertex_stride}, material: PreparedMaterial {{ base_color: {base_color}, metallic_roughness: {metallic_roughness}, emissive: {emissive}, occlusion: {occlusion}, normal: {normal}, parameters: {parameters} }}, sampled_material: {sampled_material}, helmet_program: {helmet_program}, primitives: &[{primitives}], retained_topology: {retained_topology}, retained_double_sided: {retained_double_sided} }},\n",
            vertex_count = vertices.len() / vertex_stride,
            index_count = indices.len() / 4,
            parameters = material.parameters.rust_literal(),
            sampled_material = sampled_material,
            helmet_program = slot == 0,
        ));
    }
    catalog.push_str("];\n");
    fs::write(out.join("prepared_assets.rs"), catalog).expect("write asset catalog");
}

struct PreparedGeometry {
    vertices: Vec<u8>,
    indices: Vec<u8>,
    material: PreparedMaterial,
    vertex_stride: usize,
    primitives: Vec<PreparedPrimitive>,
}

struct PreparedMaterial {
    base_color: Option<(&'static str, Vec<u8>)>,
    metallic_roughness: Option<(&'static str, Vec<u8>)>,
    emissive: Option<(&'static str, Vec<u8>)>,
    occlusion: Option<(&'static str, Vec<u8>)>,
    normal: Option<(&'static str, Vec<u8>)>,
    parameters: PreparedMaterialParameters,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PreparedMaterialParameters {
    base_color_factor: [f32; 4],
    emissive_factor: [f32; 3],
    normal_scale: f32,
    metallic_factor: f32,
    roughness_factor: f32,
    occlusion_strength: f32,
    alpha_cutoff: f32,
    double_sided: bool,
}

impl PreparedMaterialParameters {
    fn rust_literal(self) -> String {
        format!(
            "RetainedMaterialParameters {{ base_color_factor: {:?}, emissive_factor: {:?}, normal_scale: {:?}, metallic_factor: {:?}, roughness_factor: {:?}, occlusion_strength: {:?}, alpha_cutoff: {:?}, flags: {}, reserved: [0; 3] }}",
            self.base_color_factor,
            self.emissive_factor,
            self.normal_scale,
            self.metallic_factor,
            self.roughness_factor,
            self.occlusion_strength,
            self.alpha_cutoff,
            if self.double_sided {
                "trueos::vgpu::RETAINED_MATERIAL_FLAG_DOUBLE_SIDED"
            } else {
                "0"
            },
        )
    }
}

struct PreparedPrimitive {
    topology: &'static str,
    first_vertex: u32,
    vertex_count: u32,
    first_index: u32,
    index_count: u32,
    double_sided: bool,
}

fn prepare(source: &Path, sampled_material: bool) -> PreparedGeometry {
    let bytes = fs::read(source).expect("read asset");
    let parsed = gltf::Gltf::from_slice(&bytes).expect("parse asset");
    let base = source.parent().unwrap_or(Path::new("."));
    let buffers: Vec<Vec<u8>> = parsed
        .buffers()
        .map(|b| match b.source() {
            gltf::buffer::Source::Bin => parsed.blob.clone().expect("BIN chunk"),
            gltf::buffer::Source::Uri(uri) if uri.starts_with("data:") => base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                uri.split_once(',').expect("data URI").1,
            )
            .expect("base64 buffer"),
            gltf::buffer::Source::Uri(uri) => fs::read(base.join(uri)).expect("external buffer"),
        })
        .collect();
    let material = parsed.materials().next();
    let texture = |source: Option<gltf::image::Source<'_>>| {
        source.map(|image| {
            load_supported_image(image, &buffers, base)
                .expect("source material image must use a supported encoding")
        })
    };
    if let Some(material) = material.as_ref() {
        assert_eq!(
            material.alpha_mode(),
            gltf::material::AlphaMode::Opaque,
            "the retained material example currently requires opaque glTF materials"
        );
        // All checked-in maps use TEXCOORD_0 and repeat addressing. Reject a
        // future fixture requiring another contract rather than silently
        // sampling it with the helmet's coordinate set or sampler mode.
        for (coordinates, texture) in [
            material
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|info| (info.tex_coord(), info.texture())),
            material
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
                .map(|info| (info.tex_coord(), info.texture())),
            material
                .emissive_texture()
                .map(|info| (info.tex_coord(), info.texture())),
            material
                .occlusion_texture()
                .map(|info| (info.tex_coord(), info.texture())),
            material
                .normal_texture()
                .map(|info| (info.tex_coord(), info.texture())),
        ]
        .into_iter()
        .flatten()
        {
            assert_eq!(
                coordinates, 0,
                "material requires unsupported texture coordinates"
            );
            let sampler = texture.sampler();
            assert_eq!(sampler.wrap_s(), gltf::texture::WrappingMode::Repeat);
            assert_eq!(sampler.wrap_t(), gltf::texture::WrappingMode::Repeat);
            // Retained PBR sampling is linear/repeat at mip 0. Preserve the
            // image bytes even when an authored asset explicitly requests
            // linear or mipmapped filtering: the current carrier has no
            // per-texture sampler object, so its fixed resident sampler is
            // the deliberately documented presentation policy.
            let _ = (sampler.mag_filter(), sampler.min_filter());
        }
    }
    let prepared_material = PreparedMaterial {
        base_color: texture(material.as_ref().and_then(|material| {
            material
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|info| info.texture().source().source())
        })),
        metallic_roughness: texture(material.as_ref().and_then(|material| {
            material
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
                .map(|info| info.texture().source().source())
        })),
        emissive: texture(
            material
                .as_ref()
                .and_then(|material| material.emissive_texture())
                .map(|info| info.texture().source().source()),
        ),
        occlusion: texture(
            material
                .as_ref()
                .and_then(|material| material.occlusion_texture())
                .map(|info| info.texture().source().source()),
        ),
        normal: texture(
            material
                .as_ref()
                .and_then(|material| material.normal_texture())
                .map(|info| info.texture().source().source()),
        ),
        parameters: PreparedMaterialParameters {
            base_color_factor: material.as_ref().map_or([1.0; 4], |material| {
                material.pbr_metallic_roughness().base_color_factor()
            }),
            emissive_factor: material
                .as_ref()
                .map_or([0.0; 3], |material| material.emissive_factor()),
            normal_scale: material
                .as_ref()
                .and_then(|m| m.normal_texture())
                .map_or(1.0, |t| t.scale()),
            metallic_factor: material
                .as_ref()
                .map_or(1.0, |m| m.pbr_metallic_roughness().metallic_factor()),
            roughness_factor: material
                .as_ref()
                .map_or(1.0, |m| m.pbr_metallic_roughness().roughness_factor()),
            occlusion_strength: material
                .as_ref()
                .and_then(|m| m.occlusion_texture())
                .map_or(1.0, |t| t.strength()),
            alpha_cutoff: material
                .as_ref()
                .and_then(|m| m.alpha_cutoff())
                .unwrap_or(0.5),
            double_sided: material.as_ref().is_some_and(|m| m.double_sided()),
        },
    };
    // The sampled layout is selected per material, never globally: the four
    // image-free fixtures must retain their original 24-byte vertex records.
    let sampled_material = sampled_material && prepared_material.base_color.is_some();
    let (mut positions, mut normals, mut uvs, mut tangents, mut indices) = (
        Vec::<[f32; 3]>::new(),
        Vec::<[f32; 3]>::new(),
        Vec::<[f32; 2]>::new(),
        Vec::<[f32; 4]>::new(),
        Vec::<u32>::new(),
    );
    let mut primitives = Vec::<PreparedPrimitive>::new();
    for mesh in parsed.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|b| Some(buffers[b.index()].as_slice()));
            let base_vertex = positions.len() as u32;
            let mut p: Vec<_> = reader.read_positions().expect("POSITION").collect();
            let mut local: Vec<u32> = reader
                .read_indices()
                .map(|v| v.into_u32().collect())
                .unwrap_or_else(|| (0..p.len() as u32).collect());
            let mut n: Vec<_> = reader
                .read_normals()
                .map(|v| v.collect())
                .unwrap_or_else(|| generated_normals(primitive.mode(), &p, &local));
            let mut uv: Vec<_> = reader
                .read_tex_coords(0)
                .map(|coords| coords.into_f32().collect())
                .unwrap_or_else(|| {
                    assert!(
                        !sampled_material,
                        "sampled material requires authored TEXCOORD_0"
                    );
                    vec![[0.0; 2]; p.len()]
                });
            assert_eq!(p.len(), n.len());
            assert_eq!(p.len(), uv.len());
            if sampled_material {
                let t = if let Some(tangents) = reader.read_tangents() {
                    tangents.collect::<Vec<_>>()
                } else {
                    assert_eq!(
                        primitive.mode(),
                        gltf::mesh::Mode::Triangles,
                        "normal-map tangent generation requires triangle lists"
                    );
                    generate_tangents(&mut p, &mut n, &mut uv, &mut local)
                };
                assert_eq!(p.len(), t.len());
                tangents.extend(t);
            }
            let vertex_count = p.len() as u32;
            positions.extend(p);
            normals.extend(n);
            uvs.extend(uv);
            let first_index = indices.len() as u32;
            let index_count = local.len() as u32;
            indices.extend(local.into_iter().map(|i| i + base_vertex));
            primitives.push(PreparedPrimitive {
                topology: primitive_topology(primitive.mode()),
                first_vertex: base_vertex,
                vertex_count,
                first_index,
                index_count,
                double_sided: primitive.material().double_sided(),
            });
        }
    }
    assert!(!positions.is_empty() && !indices.is_empty());
    normalize(&mut positions);
    let vertex_stride = if sampled_material { 48 } else { 24 };
    let mut vb = Vec::with_capacity(positions.len() * vertex_stride);
    for (vertex, ((p, n), uv)) in positions.iter().zip(normals).zip(uvs).enumerate() {
        for v in p.iter().chain(n.iter()) {
            vb.extend(v.to_le_bytes());
        }
        if sampled_material {
            for v in uv {
                vb.extend(v.to_le_bytes());
            }
            for v in tangents[vertex] {
                vb.extend(v.to_le_bytes());
            }
        }
    }
    let mut ib = Vec::with_capacity(indices.len() * 4);
    for i in indices {
        ib.extend(i.to_le_bytes());
    }
    PreparedGeometry {
        vertices: vb,
        indices: ib,
        material: prepared_material,
        vertex_stride,
        primitives,
    }
}

/// Bake glTF tangent frames once on the host. MikkTSpace returns a tangent per
/// face corner: one indexed vertex can therefore need distinct tangent frames
/// at a mirrored UV seam. Split only those corners while preserving their
/// positions, normals, UVs, triangle order, and winding exactly.
fn generate_tangents(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut [u32],
) -> Vec<[f32; 4]> {
    struct Geometry<'a> {
        positions: &'a [[f32; 3]],
        normals: &'a [[f32; 3]],
        uvs: &'a [[f32; 2]],
        indices: &'a [u32],
        corners: &'a mut [[f32; 4]],
    }
    impl bevy_mikktspace::Geometry for Geometry<'_> {
        fn num_faces(&self) -> usize {
            self.indices.len() / 3
        }
        fn num_vertices_of_face(&self, _face: usize) -> usize {
            3
        }
        fn position(&self, face: usize, vertex: usize) -> [f32; 3] {
            self.positions[self.indices[face * 3 + vertex] as usize]
        }
        fn normal(&self, face: usize, vertex: usize) -> [f32; 3] {
            self.normals[self.indices[face * 3 + vertex] as usize]
        }
        fn tex_coord(&self, face: usize, vertex: usize) -> [f32; 2] {
            self.uvs[self.indices[face * 3 + vertex] as usize]
        }
        fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vertex: usize) {
            self.corners[face * 3 + vertex] = tangent;
        }
    }
    assert!(indices.len().is_multiple_of(3));
    let mut corners = vec![[0.0; 4]; indices.len()];
    assert!(
        bevy_mikktspace::generate_tangents(&mut Geometry {
            positions,
            normals,
            uvs,
            indices,
            corners: &mut corners,
        }),
        "MikkTSpace tangent generation failed"
    );
    let mut tangents = vec![[0.0; 4]; positions.len()];
    let mut initialized = vec![false; positions.len()];
    let mut variants = BTreeMap::<(u32, [u32; 4]), u32>::new();
    for (index, mut tangent) in indices.iter_mut().zip(corners) {
        assert!(tangent.into_iter().all(f32::is_finite));
        assert!(tangent[3] == 1.0 || tangent[3] == -1.0);
        let source_index = *index as usize;
        // A collapsed UV triangle has no defined tangent direction. Mikk can
        // return zero or its default axis there (the helmet contains both).
        // Preserve every valid Mikk frame, and give only the undefined case
        // a unit direction perpendicular to the authored normal so fragment
        // normalization cannot introduce NaNs.
        let length_squared = tangent[..3].iter().map(|v| v * v).sum::<f32>();
        let normal = normals[source_index];
        let normal_length = normal.into_iter().map(|v| v * v).sum::<f32>();
        assert!(normal_length.is_finite() && normal_length > 1e-20);
        let dot = normal
            .into_iter()
            .zip(tangent)
            .map(|(n, t)| n * t)
            .sum::<f32>();
        if length_squared <= 1e-20 || dot.abs() > 1e-4 || (length_squared - 1.0).abs() > 1e-4 {
            let mut direction = core::array::from_fn::<_, 3, _>(|component| {
                tangent[component] - normal[component] * dot / normal_length
            });
            if direction.into_iter().map(|v| v * v).sum::<f32>() <= 1e-20 {
                let axis =
                    if normal[0].abs() <= normal[1].abs() && normal[0].abs() <= normal[2].abs() {
                        [1.0, 0.0, 0.0]
                    } else if normal[1].abs() <= normal[2].abs() {
                        [0.0, 1.0, 0.0]
                    } else {
                        [0.0, 0.0, 1.0]
                    };
                direction = [
                    normal[1] * axis[2] - normal[2] * axis[1],
                    normal[2] * axis[0] - normal[0] * axis[2],
                    normal[0] * axis[1] - normal[1] * axis[0],
                ];
            }
            let inverse_length = direction
                .into_iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
                .recip();
            for component in 0..3 {
                tangent[component] = direction[component] * inverse_length;
            }
        }
        let key = (*index, tangent.map(f32::to_bits));
        let vertex = if let Some(&vertex) = variants.get(&key) {
            vertex
        } else {
            let vertex = if !initialized[source_index] {
                initialized[source_index] = true;
                tangents[source_index] = tangent;
                *index
            } else {
                let vertex = u32::try_from(positions.len()).expect("tangent vertex count");
                positions.push(positions[source_index]);
                normals.push(normals[source_index]);
                uvs.push(uvs[source_index]);
                tangents.push(tangent);
                vertex
            };
            variants.insert(key, vertex);
            vertex
        };
        *index = vertex;
    }
    tangents
}

fn write_texture(
    out: &Path,
    stem: &str,
    asset_name: &str,
    role: &str,
    texture: Option<&(&'static str, Vec<u8>)>,
) -> String {
    let Some((extension, bytes)) = texture else {
        return String::from("PreparedTexture::NONE");
    };
    let file = format!("{stem}.{role}.{extension}");
    fs::write(out.join(&file), bytes).expect("write prepared material texture");
    format!(
        "PreparedTexture {{ name: \"{asset_name}.{role}.{extension}\", bytes: include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{file}\")) }}"
    )
}

fn primitive_topology(mode: gltf::mesh::Mode) -> &'static str {
    match mode {
        gltf::mesh::Mode::Points => "PointList",
        gltf::mesh::Mode::Lines => "LineList",
        gltf::mesh::Mode::LineLoop => "LineLoop",
        gltf::mesh::Mode::LineStrip => "LineStrip",
        gltf::mesh::Mode::Triangles => "TriangleList",
        gltf::mesh::Mode::TriangleStrip => "TriangleStrip",
        gltf::mesh::Mode::TriangleFan => "TriangleFan",
    }
}

fn load_supported_image(
    source: gltf::image::Source<'_>,
    buffers: &[Vec<u8>],
    base: &Path,
) -> Option<(&'static str, Vec<u8>)> {
    match source {
        gltf::image::Source::View { view, mime_type } => {
            let extension = supported_image_extension(mime_type)?;
            let start = view.offset();
            let end = start.checked_add(view.length())?;
            Some((
                extension,
                buffers
                    .get(view.buffer().index())?
                    .get(start..end)?
                    .to_vec(),
            ))
        }
        gltf::image::Source::Uri { uri, mime_type } => {
            let extension = mime_type.and_then(supported_image_extension).or_else(|| {
                uri.rsplit_once('.')
                    .and_then(|(_, ext)| supported_image_extension(ext))
            })?;
            let bytes = if uri.starts_with("data:") {
                base64::Engine::decode(
                    &base64::engine::general_purpose::STANDARD,
                    uri.split_once(',')?.1,
                )
                .ok()?
            } else {
                fs::read(base.join(uri)).ok()?
            };
            Some((extension, bytes))
        }
    }
}

fn supported_image_extension(value: &str) -> Option<&'static str> {
    match value.to_ascii_lowercase().as_str() {
        "image/jpeg" | "jpeg" | "jpg" => Some("jpg"),
        "image/png" | "png" => Some("png"),
        "image/bmp" | "bmp" => Some("bmp"),
        _ => None,
    }
}

fn generated_normals(mode: gltf::mesh::Mode, p: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0; 3]; p.len()];
    let mut add_triangle = |a: u32, b: u32, c: u32| {
        let a_position = p[a as usize];
        let b_position = p[b as usize];
        let c_position = p[c as usize];
        let u = [
            b_position[0] - a_position[0],
            b_position[1] - a_position[1],
            b_position[2] - a_position[2],
        ];
        let v = [
            c_position[0] - a_position[0],
            c_position[1] - a_position[1],
            c_position[2] - a_position[2],
        ];
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        for i in [a, b, c] {
            for a in 0..3 {
                out[i as usize][a] += n[a];
            }
        }
    };
    match mode {
        gltf::mesh::Mode::Triangles => {
            for triangle in indices.chunks_exact(3) {
                add_triangle(triangle[0], triangle[1], triangle[2]);
            }
        }
        gltf::mesh::Mode::TriangleStrip => {
            for (index, triangle) in indices.windows(3).enumerate() {
                let [a, b, c] = [triangle[0], triangle[1], triangle[2]];
                if index.is_multiple_of(2) {
                    add_triangle(a, b, c);
                } else {
                    add_triangle(b, a, c);
                }
            }
        }
        gltf::mesh::Mode::TriangleFan => {
            if let Some(&center) = indices.first() {
                for triangle in indices[1..].windows(2) {
                    add_triangle(center, triangle[0], triangle[1]);
                }
            }
        }
        gltf::mesh::Mode::Points
        | gltf::mesh::Mode::Lines
        | gltf::mesh::Mode::LineLoop
        | gltf::mesh::Mode::LineStrip => {}
    }
    for n in &mut out {
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if l > 0.0 {
            for v in n {
                *v /= l;
            }
        } else {
            n[2] = 1.0;
        }
    }
    out
}
fn normalize(p: &mut [[f32; 3]]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in p.iter() {
        for a in 0..3 {
            min[a] = min[a].min(v[a]);
            max[a] = max[a].max(v[a]);
        }
    }
    let c = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let e = (max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2]);
    assert!(e > 0.0);
    for v in p {
        for a in 0..3 {
            v[a] = (v[a] - c[a]) * 1.6 / e;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset_path(relative: &str) -> PathBuf {
        Path::new(file!()).parent().unwrap().join(relative)
    }

    fn component(vertex: &[u8], component: usize) -> f32 {
        f32::from_le_bytes(vertex[component * 4..component * 4 + 4].try_into().unwrap())
    }

    #[test]
    fn helmet_bake_preserves_authored_uvs_and_produces_orthogonal_tangents() {
        let source = asset_path("Assets/DamagedHelmet/DamagedHelmet.glb");
        let prepared = prepare(&source, true);
        assert_eq!(prepared.vertex_stride, 48);
        let bytes = fs::read(source).unwrap();
        let gltf = gltf::Gltf::from_slice(&bytes).unwrap();
        let primitive = gltf.meshes().next().unwrap().primitives().next().unwrap();
        let reader = primitive.reader(|_| gltf.blob.as_deref());
        let source_uvs = reader
            .read_tex_coords(0)
            .unwrap()
            .into_f32()
            .collect::<Vec<_>>();
        let source_indices = reader
            .read_indices()
            .unwrap()
            .into_u32()
            .collect::<Vec<_>>();
        assert!(source_uvs.iter().any(|uv| uv[1] > 1.0));
        assert_eq!(prepared.indices.len(), source_indices.len() * 4);
        for (index, source_index) in prepared.indices.chunks_exact(4).zip(source_indices) {
            let index = u32::from_le_bytes(index.try_into().unwrap()) as usize;
            let vertex = &prepared.vertices[index * 48..index * 48 + 48];
            assert_eq!(
                component(vertex, 6).to_bits(),
                source_uvs[source_index as usize][0].to_bits()
            );
            assert_eq!(
                component(vertex, 7).to_bits(),
                source_uvs[source_index as usize][1].to_bits()
            );
            let normal = [
                component(vertex, 3),
                component(vertex, 4),
                component(vertex, 5),
            ];
            let tangent = [
                component(vertex, 8),
                component(vertex, 9),
                component(vertex, 10),
            ];
            let dot = normal
                .into_iter()
                .zip(tangent)
                .map(|(n, t)| n * t)
                .sum::<f32>();
            let length = tangent.into_iter().map(|t| t * t).sum::<f32>();
            assert!(
                dot.abs() < 1e-3,
                "nonorthogonal tangent at {index}: {dot}; normal={normal:?} tangent={tangent:?}"
            );
            assert!((length - 1.0).abs() < 1e-3, "nonunit tangent: {length}");
            assert!(component(vertex, 11).abs() == 1.0);
        }
        assert!(!prepared.primitives[0].double_sided);
        assert_eq!(prepared.primitives[0].topology, "TriangleList");
        assert_eq!(
            prepared.material.parameters,
            PreparedMaterialParameters {
                base_color_factor: [1.0; 4],
                emissive_factor: [1.0; 3],
                normal_scale: 1.0,
                metallic_factor: 1.0,
                roughness_factor: 1.0,
                occlusion_strength: 1.0,
                alpha_cutoff: 0.5,
                double_sided: false,
            }
        );
    }

    #[test]
    fn prepared_vertex_uv_tangents_and_every_map_round_trip_through_picasso_redb() {
        let prepared = prepare(&asset_path("Assets/DamagedHelmet/DamagedHelmet.glb"), true);
        let picasso = trueos_picasso::Picasso::new().unwrap();
        let mut entries = vec![
            ("vertices", prepared.vertices.as_slice()),
            ("indices", prepared.indices.as_slice()),
        ];
        for (role, texture) in [
            ("base-color", &prepared.material.base_color),
            ("metallic-roughness", &prepared.material.metallic_roughness),
            ("emissive", &prepared.material.emissive),
            ("occlusion", &prepared.material.occlusion),
            ("normal", &prepared.material.normal),
        ] {
            let (_, bytes) = texture.as_ref().expect("helmet map must be retained");
            entries.push((role, bytes.as_slice()));
        }
        for (role, bytes) in entries {
            picasso.put_embedded_asset(role, bytes).unwrap();
            assert_eq!(picasso.embedded_asset(role).unwrap().unwrap(), bytes);
        }
    }

    #[test]
    fn image_free_assets_keep_position_normal_layout() {
        for (_, source) in ASSETS.iter().filter(|(name, _)| {
            // The large Ship fixture intentionally has a sampled material;
            // this test covers only the small image-free fixtures.
            *name != "DamagedHelmet" && *name != "Ship"
        }) {
            let prepared = prepare(&asset_path(source), true);
            assert!(prepared.material.base_color.is_none());
            assert_eq!(prepared.vertex_stride, 24);
            assert!(prepared.vertices.len().is_multiple_of(24));
        }
    }

    #[test]
    fn ship_bake_keeps_all_primitive_ranges_and_uses_the_sampled_layout() {
        let prepared = prepare(&asset_path("Assets/Ship/mud.glb"), true);
        assert_eq!(prepared.vertex_stride, 48);
        assert!(prepared.material.base_color.is_some());
        assert_eq!(prepared.primitives.len(), 9);
        assert!(prepared.primitives.iter().all(|primitive| {
            primitive.topology == "TriangleList"
                && primitive.vertex_count > 0
                && primitive.index_count > 0
                && primitive.first_vertex + primitive.vertex_count
                    <= (prepared.vertices.len() / prepared.vertex_stride) as u32
                && primitive.first_index + primitive.index_count <= (prepared.indices.len() / 4) as u32
        }));
    }

    #[test]
    fn mirrored_uv_tangent_seam_splits_shared_vertex_without_changing_winding() {
        let mut positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        let original_positions = positions.clone();
        let mut normals = vec![[0.0, 0.0, 1.0]; 5];
        let mut uvs = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]];
        let mut indices = [0, 1, 2, 0, 3, 4];
        let original_indices = indices;
        let tangents = generate_tangents(&mut positions, &mut normals, &mut uvs, &mut indices);
        assert_ne!(indices[0], indices[3]);
        assert_eq!(
            tangents[indices[0] as usize][3],
            -tangents[indices[3] as usize][3]
        );
        for (actual, original) in indices.into_iter().zip(original_indices) {
            assert_eq!(
                positions[actual as usize],
                original_positions[original as usize]
            );
        }
    }
}
