// The photo view's sea, bed and sky (`crate::photo`): one material, three kinds.
//
// - Sea surface (kind 0): waves as a sum of sinusoids from `crate::photo::SeaState` (the
//   first `flow.w` of them the model's sea for an H_s of 1 m, scaled by the model's H_s
//   at the pixel, UV 1's x; then the wind's),
//   their slopes per pixel, advected with the model's current (a vertex attribute) by a
//   two-phase flow map and steepened against it; sky reflection by Fresnel, the sun's
//   glint by a Beckmann distribution whose width is the slope variance the pixel does
//   not resolve, and whitecaps on the crests, and surf where the model's waves break
//   (UV 1's y: the share of their energy lost per period). Premultiplied alpha: what is
//   not reflected lets the bed through.
// - Bed and ground (kind 1): lit by the sun and the sky through the water above (its
//   depth a vertex attribute), seen through it: Beer–Lambert attenuation per channel on
//   the way down and back up, the water's own colour scattered in.
// - Sky (kind 2): a dome around the camera; below the horizon, the haze of the far sea.
//
// World frame: +Y up, the mesh's (x, y) at (X, −Z).

import bevy_pbr::render::forward_io::VertexOutput;
import bevy_pbr::render::mesh_view_bindings::view;
@if(TONEMAP_IN_SHADER) import bevy_core_pipeline::tonemapping::tone_mapping;

const N_WAVES: u32 = 32u;
const PI: f32 = 3.14159265;
const G: f32 = 9.81;

struct WaterParams {
    // kx, kz (rad/m), slope amplitude a·k, phase (rad) at the shown time
    waves: array<vec4<f32>, 32>,
    // x: flow-map phase (time over its period, fractional); y: its period (s); z: slope
    // variance below the shortest wave; w: how many of the waves are the model's
    flow: vec4<f32>,
    // xyz: toward the sun; w: its illuminance (lux)
    sun: vec4<f32>,
    // rgb: the sun's colour; w: ambient sky brightness (cd/m²)
    sun_colour: vec4<f32>,
    // Sky luminance (cd/m², rgb) overhead and at the horizon; zenith w: how low and red
    // the sun is, 0 to 1
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    // rgb: diffuse attenuation of the water (1/m)
    attenuation: vec4<f32>,
    // rgb: irradiance reflectance of deep water (its colour)
    deep: vec4<f32>,
    // rgb: haze colour (cd/m²); w: haze density (1/m)
    fog: vec4<f32>,
    // x: whitecap threshold (rms heights above the mean); y: rms height of the wind's
    // waves (m); z: whitecap coverage of the wind sea; w: direction the wind blows toward
    // (rad from +X toward +Z)
    foam: vec4<f32>,
    // x: kind (0 sea, 1 bed, 2 sky); y: the tile (m) the wavevectors repeat on; z: wave
    // time (s, wrapped at an hour); w: vertical exaggeration
    misc: vec4<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> params: WaterParams;

fn hash(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2(0.1031, 0.1030));
    let r = q + dot(q, q.yx + 33.33);
    return fract((r.x + r.y) * r.x);
}

// Smooth value noise in [0, 1].
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash(i);
    let b = hash(i + vec2(1.0, 0.0));
    let c = hash(i + vec2(0.0, 1.0));
    let d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Sky luminance toward `d`; the sun's disk only when `disk`.
fn sky(d: vec3<f32>, disk: bool) -> vec3<f32> {
    let sun = params.sun.xyz;
    let up = clamp(d.y, 0.0, 1.0);
    var c = mix(params.sky_horizon.rgb, params.sky_zenith.rgb, pow(up, 0.45));
    let mu = max(dot(d, sun), 0.0);
    // Forward scattering about the sun
    // reddened when the sun is low
    let glow = mix(vec3(1.0), vec3(1.7, 0.85, 0.45), params.sky_zenith.w);
    c += params.sky_horizon.rgb * glow * (0.6 * pow(mu, 8.0) + 0.25 * pow(mu, 64.0));
    if disk {
        // 0.27° radius, as bright as the sun's illuminance over its solid angle
        let edge = smoothstep(0.999985, 0.999990, mu);
        c += params.sun_colour.rgb * params.sun.w / 6.8e-5 * edge;
    }
    if d.y < 0.0 {
        c = mix(c, params.fog.rgb, smoothstep(0.0, -0.02, d.y));
    }
    return c;
}

// Fresnel reflectance of water (n = 1.33) by Schlick's approximation.
fn fresnel(cos_theta: f32) -> f32 {
    return 0.02 + 0.98 * pow(1.0 - clamp(cos_theta, 0.0, 1.0), 5.0);
}

// Cosine of the refracted angle in water for a ray at cosine `c` in air.
fn cos_refracted(c: f32) -> f32 {
    return sqrt(1.0 - (1.0 - c * c) / 1.7689);
}

// What the ground is made of where the chart colours it: a coast of grass and heather
// on the gentle slopes, bare rock on the steep ones and the tops, the shore's wet rock
// and wrack, and under water kelp and rock darkening the chart's sand.
fn natural(chart: vec3<f32>, world: vec3<f32>, n: vec3<f32>, depth: f32) -> vec3<f32> {
    let vz = max(params.misc.w, 1e-3);
    let z = world.y / vz;
    let patches = noise(world.xz / 35.0) * 0.6 + noise(world.xz / 9.0) * 0.4;
    if depth > 0.0 || z < 0.0 {
        let bed = mix(vec3(0.07, 0.07, 0.045), vec3(0.16, 0.14, 0.10), patches);
        return mix(bed, chart * 0.45, 0.35);
    }
    let grass = mix(vec3(0.055, 0.075, 0.03), vec3(0.11, 0.10, 0.05), patches);
    let heath = vec3(0.10, 0.075, 0.055);
    let rock = mix(vec3(0.12, 0.115, 0.105), vec3(0.21, 0.20, 0.185), patches);
    var c = mix(grass, heath, smoothstep(0.35, 0.75, patches));
    // Steepness on the true slope, not the exaggerated one
    let steep = smoothstep(0.80, 0.60, normalize(vec3(n.x / vz, n.y, n.z / vz)).y);
    c = mix(c, rock, max(steep, smoothstep(60.0, 140.0, z + 40.0 * patches)));
    // The shore: wet rock, dark with wrack, to a metre or two above mean sea level
    c = mix(vec3(0.05, 0.045, 0.035), c, smoothstep(0.5, 2.5, z + patches));
    return c;
}

fn haze(dist: f32) -> f32 {
    return 1.0 - exp(-dist * params.fog.w);
}

struct Sea {
    // ∂h/∂X, ∂h/∂Z
    slope: vec2<f32>,
    // Height above the mean (m), and of it the model's waves'
    height: f32,
    swell: f32,
    // Slope variance the pixel does not resolve
    lost: f32,
}

// The waves at wrapped position `p` (m), advected by the current `u` (m/s) in the flow
// map's two phases, `tau` (s) of advection each and weights `w`, at wave time `t` (s),
// filtered to the pixel footprint `footprint` (m); the model's waves scaled to its H_s
// `hs` (m) there.
fn waves(p: vec2<f32>, u: vec2<f32>, tau: vec2<f32>, w: vec2<f32>, t: f32, footprint: f32, hs: f32) -> Sea {
    var sea: Sea;
    let n_model = u32(params.flow.w + 0.5);
    for (var i = 0u; i < N_WAVES; i++) {
        let wave = params.waves[i];
        let k = length(wave.xy);
        if k <= 0.0 {
            continue;
        }
        let dir = wave.xy / k;
        // Steepened against the current, eased along it: wave action conserved to first
        // order in U/c (Longuet-Higgins & Stewart 1961), slope ∝ 1 − 3 U·k̂ / c
        let c = sqrt(G / k);
        let steep = clamp(1.0 - 3.0 * dot(u, dir) / c, 0.3, 3.0);
        // In groups, a few wavelengths long, running at the group velocity c/2
        let lambda = 2.0 * PI / k;
        let groups = 0.35 + 1.3 * noise((p - dir * 0.5 * c * t) / (5.0 * lambda) + f32(i) * 7.31);
        let s = wave.z * steep * groups * select(1.0, hs, i < n_model);
        // Waves shorter than 6 pixels fade out, into the roughness
        let keep = clamp(lambda / footprint / 3.0 - 1.0, 0.0, 1.0);
        // Short-crested: the phase warped by a noise carried with the water, a
        // radian or so over a few wavelengths
        let q0 = p - u * tau.x;
        let q1 = p - u * tau.y;
        let seed = f32(i) * 3.17;
        let theta0 = dot(wave.xy, q0) + wave.w + 2.4 * noise(q0 / (3.0 * lambda) + seed);
        let theta1 = dot(wave.xy, q1) + wave.w + 2.4 * noise(q1 / (3.0 * lambda) + seed);
        // The two phases' blend at the wave's own variance: they are one wave apart by
        // k·u (τ₁ − τ₀) in phase, from in phase (no current) to independent
        let norm = inverseSqrt(max(dot(w, w) + 2.0 * w.x * w.y * cos(theta0 - theta1), 0.05));
        let sine = (w.x * sin(theta0) + w.y * sin(theta1)) * norm;
        let cosine = (w.x * cos(theta0) + w.y * cos(theta1)) * norm;
        sea.slope -= keep * s * dir * sine;
        sea.height += s / k * cosine;
        if i < n_model {
            sea.swell += s / k * cosine;
        }
        sea.lost += (1.0 - keep * keep) * 0.5 * s * s;
    }
    return sea;
}

// Foam streaks drawn out downwind, carried with the flow: in [0, 1].
fn streaks(q: vec2<f32>) -> f32 {
    let wind = vec2(cos(params.foam.w), sin(params.foam.w));
    let along = dot(q, wind);
    let across = dot(q, vec2(-wind.y, wind.x));
    return 0.65 * noise(vec2(along / 9.0, across / 1.5)) + 0.35 * noise(vec2(along / 2.5, across / 0.6));
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let kind = u32(params.misc.x + 0.5);
    let world = in.world_position.xyz;
    let to_eye = view.world_position - world;
    let dist = length(to_eye);
    let v = to_eye / dist;
    let exposure = view.exposure;
    let sun = params.sun.xyz;
    let e_sun = params.sun.w * params.sun_colour.rgb;
    let e_sky = params.sun_colour.w;
    // Pixel footprint on the surface, before any branch (derivatives need uniform flow)
    let footprint = max(length(dpdx(world.xz)), length(dpdy(world.xz))) + 1e-4;

    var out: vec4<f32>;
    if kind == 2u {
        out = vec4(sky(-v, true) * exposure, 1.0);
    } else if kind == 1u {
        var chart = vec3(0.3);
        @if(VERTEX_COLORS) {
            chart = in.color.rgb;
        }
        var depth = 0.0;
        @if(VERTEX_UVS_B) {
            depth = max(in.uv_b.x, 0.0);
        }
        let n = normalize(select(-in.world_normal, in.world_normal, front));
        let albedo = natural(chart, world, n, depth);
        let k = params.attenuation.rgb;
        // Light down to the bed, and back up to the surface toward the eye
        let t_sun = exp(-k * depth / cos_refracted(max(sun.y, 0.05)));
        let t_sky = exp(-k * depth * 1.2);
        let t_view = exp(-k * depth / cos_refracted(clamp(v.y, 0.05, 1.0)));
        let lit = e_sun * max(dot(n, sun), 0.0) / PI * t_sun + e_sky * t_sky;
        // Light the water scatters back up, from the sun and the sky over it
        let down = e_sun * max(sun.y, 0.0) / PI + e_sky;
        let water = params.deep.rgb * down * (1.0 - t_view);
        var c = albedo * lit * t_view + water;
        if depth <= 0.0 {
            c = albedo * (e_sun * max(dot(n, sun), 0.0) / PI + e_sky);
        }
        c = mix(c, params.fog.rgb, haze(dist));
        out = vec4(c * exposure, 1.0);
    } else {
        var u = vec2(0.0);
        @if(VERTEX_UVS_A) {
            u = in.uv;
        }
        // The model's H_s and the share of the waves' energy breaking takes per period
        var model = vec2(0.0);
        @if(VERTEX_UVS_B) {
            model = max(in.uv_b, vec2(0.0));
        }
        let tile = params.misc.y;
        let p = world.xz - tile * floor(world.xz / tile);
        // Two phases of the flow map, half a period apart, each restarting where its
        // weight is zero; the restarts are staggered across the sea by a noise
        let period = params.flow.y;
        let phase = params.flow.x + noise(world.xz / 60.0);
        let f0 = fract(phase);
        let f1 = fract(phase + 0.5);
        let w0 = 1.0 - abs(2.0 * f0 - 1.0);
        let w1 = 1.0 - w0;
        let tau = vec2(f0, f1) * period;
        let weights = vec2(w0, w1);
        let sea = waves(p, u, tau, weights, params.misc.z, footprint, model.x);
        let slope = sea.slope;
        let height = sea.height;
        let lost = sea.lost;

        var n = normalize(vec3(-slope.x, 1.0, -slope.y));
        // The model's own surface slope (exaggerated with the heights)
        n = normalize(n + in.world_normal - vec3(0.0, 1.0, 0.0));
        let n_v = max(dot(n, v), 1e-3);
        var r = reflect(-v, n);
        r.y = abs(r.y);
        let fr = fresnel(n_v);
        let reflected = sky(r, false) * fr;

        // The sun's glint: Beckmann with the slope variance below the pixel
        let m2 = max(params.flow.z + lost, 2e-4);
        let h = normalize(v + sun);
        let n_h = max(dot(n, h), 1e-3);
        let tan2 = (1.0 - n_h * n_h) / (n_h * n_h);
        let d = exp(-tan2 / m2) / (PI * m2 * n_h * n_h * n_h * n_h);
        let glint = e_sun * fresnel(dot(v, h)) * d / (4.0 * n_v) * step(0.0, sun.y);

        // Whitecaps on the highest crests of the wind's waves, more of them on the crests
        // of the model's longer ones, broken up by a noise carried with the flow
        let sigma_wind = max(params.foam.y, 1e-4);
        let streak = w0 * streaks(p - u * tau.x) + w1 * streaks(p - u * tau.y);
        let threshold = params.foam.x * sigma_wind;
        let lift = 0.35 * sigma_wind * clamp(4.0 * sea.swell / max(model.x, 1e-3), -2.0, 2.0);
        let crest = smoothstep(
            threshold,
            threshold + 0.5 * sigma_wind,
            height - sea.swell + lift + 0.5 * sigma_wind * (streak - 0.5),
        );
        // The sea here, the wind's waves and the model's (H_s = 4 rms)
        let sigma = max(sqrt(sigma_wind * sigma_wind + model.x * model.x / 16.0), 1e-4);
        var foam = clamp(crest * smoothstep(0.0, 0.002, params.foam.z), 0.0, 0.9);
        // Surf where the model's waves break: white water on the crests and the foam
        // they leave, the more of their energy breaking takes per period
        let surf = smoothstep(0.02, 0.4, model.y);
        let bore = smoothstep(-0.5 * sigma, 0.8 * sigma, height + sigma * (streak - 0.5));
        foam = max(foam, surf * mix(0.35 + 0.4 * streak, 0.95, bore));
        let foam_light = vec3(0.8) * (e_sun * max(n.y * sun.y, 0.0) / PI + e_sky);

        let alpha = fr * (1.0 - foam) + foam;
        var c = (reflected + glint) * (1.0 - foam) + foam_light * foam;
        let f = haze(dist);
        c = c * (1.0 - f) + params.fog.rgb * f * alpha;
        out = vec4(c * exposure, alpha);
    }
    @if(TONEMAP_IN_SHADER) {
        out = vec4(tone_mapping(vec4(out.rgb, 1.0), view.color_grading).rgb, out.a);
    }
    return out;
}
