//! The farm's net cages: a floating collar and the net hanging from it.
//!
//! A circular cage (`CageFootprint::Circle`) is drawn as the usual plastic ring cage:
//! two collar pipes on the surface and a cylindrical net down to the net depth, closed
//! by a flat bottom. A polygonal one gets the net's walls and bottom only. The net is
//! a translucent mesh texture tiled in world units, with mipmaps, so from afar it fades
//! to a veil instead of shimmering. Each cage rides the surface: its height follows η
//! at its centre.
//!
//! The drag model (`dg_rs::source::CageDrag2D`) sees only the footprint and net depth;
//! the collar is decoration.

use std::f32::consts::TAU;

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use dg_rs::source::{CageFootprint, NetCage};

use crate::field::{Field, Frame, Probe};
use crate::screen::{RUN, RunResources, Screen, Viewing};

/// World units per repeat of the net texture.
const NET_TILE: f32 = 2.0;
/// Radius of a collar pipe (m).
const PIPE_RADIUS: f32 = 0.3;
/// Spacing of the collar's two pipes (m).
const PIPE_SPACING: f32 = 1.2;

/// The cages to draw, each with a probe at its centre.
#[derive(Resource)]
pub struct CageLayout(pub Vec<(NetCage, Option<Probe>)>);

#[derive(Component)]
struct Cage(Option<Probe>);

pub struct CagesPlugin;

impl Plugin for CagesPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(Screen::Viewer), spawn)
            .add_systems(
                Update,
                ride.after(crate::field::interpolate).in_set(Viewing),
            )
            .run_resource::<CageLayout>();
    }
}

fn spawn(
    mut commands: Commands,
    layout: Res<CageLayout>,
    frame: Res<Frame>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let net = materials.add(StandardMaterial {
        base_color: Color::srgb(0.32, 0.36, 0.3),
        base_color_texture: Some(images.add(net_texture())),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.8,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    let collar = materials.add(StandardMaterial {
        base_color: Color::srgb(0.03, 0.03, 0.035),
        perceptual_roughness: 0.4,
        ..default()
    });
    for (cage, probe) in &layout.0 {
        let depth = cage.net_depth as f32 * frame.vz;
        let (centre, outline) = match &cage.footprint {
            CageFootprint::Circle { center, radius } => {
                let r = *radius as f32;
                let outline = (0..=64)
                    .map(|i| Vec2::from_angle(TAU * i as f32 / 64.0) * r)
                    .collect::<Vec<_>>();
                (*center, outline)
            }
            CageFootprint::Polygon(vertices) => {
                let n = vertices.len() as f64;
                let c = vertices
                    .iter()
                    .fold([0.0; 2], |a, v| [a[0] + v[0] / n, a[1] + v[1] / n]);
                let mut outline: Vec<Vec2> = vertices
                    .iter()
                    .map(|v| Vec2::new((v[0] - c[0]) as f32, -(v[1] - c[1]) as f32))
                    .collect();
                outline.push(outline[0]);
                (c, outline)
            }
        };
        let mut entity = commands.spawn((
            RUN,
            Name::new("Cage"),
            Cage(probe.clone()),
            Transform::from_translation(frame.world(centre, 0.0)),
            Visibility::default(),
        ));
        entity.with_child((
            Mesh3d(meshes.add(net_mesh(&outline, depth))),
            MeshMaterial3d(net.clone()),
        ));
        if let CageFootprint::Circle { radius, .. } = cage.footprint {
            for r in [radius as f32, radius as f32 + PIPE_SPACING] {
                let pipe = Torus::new(r - PIPE_RADIUS, r + PIPE_RADIUS);
                entity.with_child((
                    Mesh3d(meshes.add(pipe)),
                    MeshMaterial3d(collar.clone()),
                    Transform::from_xyz(0.0, 0.1, 0.0),
                ));
            }
        }
    }
}

/// Moves each cage to the surface at its centre.
fn ride(field: Res<Field>, frame: Res<Frame>, mut cages: Query<(&Cage, &mut Transform)>) {
    if field.t.is_none() || !field.is_changed() {
        return;
    }
    for (cage, mut transform) in &mut cages {
        if let Some(probe) = &cage.0 {
            transform.translation.y = probe.eval(&field.eta) * frame.vz;
        }
    }
}

/// The net: walls down from the closed `outline` (world X, Z about the centre) to
/// `depth`, and a bottom fanned from the centre. UVs are in [`NET_TILE`]s of world.
fn net_mesh(outline: &[Vec2], depth: f32) -> Mesh {
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    let mut along = 0.0;
    for (i, p) in outline.iter().enumerate() {
        if i > 0 {
            along += p.distance(outline[i - 1]);
        }
        let u = along / NET_TILE;
        positions.extend([[p.x, 0.0, p.y], [p.x, -depth, p.y]]);
        uvs.extend([[u, 0.0], [u, depth / NET_TILE]]);
        if i > 0 {
            let t = 2 * i as u32;
            indices.extend([t - 2, t - 1, t + 1, t - 2, t + 1, t]);
        }
    }
    let centre = positions.len() as u32;
    positions.push([0.0, -depth, 0.0]);
    uvs.push([0.0, 0.0]);
    for (i, p) in outline.iter().enumerate() {
        positions.push([p.x, -depth, p.y]);
        uvs.push([p.x / NET_TILE, p.y / NET_TILE]);
        if i > 0 {
            let v = centre + 1 + i as u32;
            indices.extend([centre, v - 1, v]);
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(indices))
    .with_computed_normals()
}

/// One cell of netting: opaque twine along two edges over a faint veil, white (the
/// material gives the colour), with a box-filtered mip chain.
fn net_texture() -> Image {
    const SIZE: usize = 64;
    const TWINE: usize = 5;
    let mut level: Vec<f32> = (0..SIZE * SIZE)
        .map(|p| {
            let (x, y) = (p % SIZE, p / SIZE);
            if x < TWINE || y < TWINE { 1.0 } else { 0.1 }
        })
        .collect();
    let mut data = Vec::new();
    let mut size = SIZE;
    let mut levels = 0;
    loop {
        data.extend(
            level
                .iter()
                .flat_map(|&a| [255, 255, 255, (a * 255.0).round() as u8]),
        );
        levels += 1;
        if size == 1 {
            break;
        }
        let half = size / 2;
        level = (0..half * half)
            .map(|p| {
                let (x, y) = (2 * (p % half), 2 * (p / half));
                0.25 * (level[y * size + x]
                    + level[y * size + x + 1]
                    + level[(y + 1) * size + x]
                    + level[(y + 1) * size + x + 1])
            })
            .collect();
        size = half;
    }
    let mut image = Image::new_fill(
        Extent3d {
            width: SIZE as u32,
            height: SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[255; 4],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(data);
    image.texture_descriptor.mip_level_count = levels;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        anisotropy_clamp: 8,
        ..ImageSamplerDescriptor::linear()
    });
    image
}
