#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "pillow",
# ]
# ///
"""
Visualize VTU output from dgcom shallow water simulations.

Usage:
    uv run scripts/visualize_vtu.py output/froya/froya_0006.vtu
    uv run scripts/visualize_vtu.py output/froya/  # All frames
"""

import sys
import os
import glob
import xml.etree.ElementTree as ET
import base64
import struct
import numpy as np

def read_vtu(filename):
    """Read VTU file and extract mesh and field data."""
    tree = ET.parse(filename)
    root = tree.getroot()

    # Get the UnstructuredGrid
    grid = root.find('.//UnstructuredGrid')
    piece = grid.find('Piece')

    n_points = int(piece.get('NumberOfPoints'))
    n_cells = int(piece.get('NumberOfCells'))

    # Read points
    points_data = piece.find('Points/DataArray')
    points = decode_data_array(points_data, n_points * 3)
    points = np.array(points).reshape(-1, 3)

    # Read connectivity
    cells_elem = piece.find('Cells')
    connectivity_data = cells_elem.find("DataArray[@Name='connectivity']")
    connectivity = decode_data_array(connectivity_data)

    offsets_data = cells_elem.find("DataArray[@Name='offsets']")
    offsets = decode_data_array(offsets_data)

    # Read point data (fields)
    point_data = {}
    for da in piece.findall('PointData/DataArray'):
        name = da.get('Name')
        n_components = int(da.get('NumberOfComponents', 1))
        data = decode_data_array(da, n_points * n_components)
        if n_components > 1:
            data = np.array(data).reshape(-1, n_components)
        else:
            data = np.array(data)
        point_data[name] = data

    # Read field data (time)
    time = 0.0
    for da in root.findall('.//FieldData/DataArray'):
        if da.get('Name') == 'TimeValue':
            time = decode_data_array(da, 1)[0]

    return {
        'points': points,
        'connectivity': connectivity,
        'offsets': offsets,
        'n_cells': n_cells,
        'point_data': point_data,
        'time': time
    }

def decode_data_array(element, expected_size=None):
    """Decode a VTK DataArray element."""
    format_type = element.get('format', 'ascii')
    data_type = element.get('type', 'Float64')

    if format_type == 'ascii':
        text = element.text.strip()
        if data_type in ('Float64', 'Float32'):
            return [float(x) for x in text.split()]
        else:
            return [int(x) for x in text.split()]
    elif format_type == 'binary':
        # Base64 encoded binary
        encoded = element.text.strip()
        decoded = base64.b64decode(encoded)

        # First 8 bytes are the size (UInt64)
        size = struct.unpack('<Q', decoded[:8])[0]
        data_bytes = decoded[8:]

        if data_type == 'Float64':
            return list(struct.unpack(f'<{size//8}d', data_bytes))
        elif data_type == 'Float32':
            return list(struct.unpack(f'<{size//4}f', data_bytes))
        elif data_type == 'Int64':
            return list(struct.unpack(f'<{size//8}q', data_bytes))
        elif data_type == 'Int32':
            return list(struct.unpack(f'<{size//4}i', data_bytes))
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    else:
        raise ValueError(f"Unknown format: {format_type}")

def plot_field(vtu_data, field_name, ax=None, cmap='viridis', title=None):
    """Plot a field from VTU data using matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.collections import PolyCollection

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    points = vtu_data['points']
    connectivity = vtu_data['connectivity']
    offsets = vtu_data['offsets']

    if field_name not in vtu_data['point_data']:
        print(f"Available fields: {list(vtu_data['point_data'].keys())}")
        raise ValueError(f"Field '{field_name}' not found")

    field = vtu_data['point_data'][field_name]

    # Build polygons from connectivity
    polygons = []
    colors = []
    prev_offset = 0
    for offset in offsets:
        cell_points = connectivity[prev_offset:offset]
        poly = points[cell_points, :2]  # x, y only
        polygons.append(poly)
        # Average field value for cell color
        cell_values = field[cell_points]
        if len(cell_values.shape) > 1:
            cell_values = np.linalg.norm(cell_values, axis=1)
        colors.append(np.mean(cell_values))
        prev_offset = offset

    colors = np.array(colors)

    # Create collection
    collection = PolyCollection(polygons, array=colors, cmap=cmap, edgecolors='face')
    ax.add_collection(collection)
    ax.autoscale()
    ax.set_aspect('equal')

    # Colorbar
    cbar = plt.colorbar(collection, ax=ax)

    # Labels
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{field_name} at t = {vtu_data["time"]:.1f} s')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    return ax

def print_summary(vtu_data):
    """Print summary of VTU data."""
    print(f"\n{'='*60}")
    print(f"VTU Summary")
    print(f"{'='*60}")
    print(f"Time: {vtu_data['time']:.1f} s ({vtu_data['time']/3600:.2f} hours)")
    print(f"Points: {len(vtu_data['points'])}")
    print(f"Cells: {vtu_data['n_cells']}")
    print(f"\nFields:")
    for name, data in vtu_data['point_data'].items():
        if len(data.shape) == 1:
            print(f"  {name}: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
        else:
            mag = np.linalg.norm(data, axis=1)
            print(f"  {name}: |min|={mag.min():.4f}, |max|={mag.max():.4f}, |mean|={mag.mean():.4f}")
    print(f"{'='*60}\n")

def plot_velocity_vectors(vtu_data, ax, skip=4, scale=15, color='black'):
    """Add velocity vectors to an existing plot."""
    points = vtu_data['points']

    if 'u' not in vtu_data['point_data'] or 'v' not in vtu_data['point_data']:
        return

    u = vtu_data['point_data']['u']
    v = vtu_data['point_data']['v']

    # Subsample for clarity
    x = points[::skip, 0]
    y = points[::skip, 1]
    u_sub = u[::skip]
    v_sub = v[::skip]

    # Only plot where there's significant velocity
    mask = np.sqrt(u_sub**2 + v_sub**2) > 0.01

    ax.quiver(x[mask], y[mask], u_sub[mask], v_sub[mask],
              scale=scale, color=color, alpha=0.7, width=0.003)


def create_animation(files, output_file='froya_animation.gif'):
    """Create an animated GIF from VTU files."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Read all frames
    print(f"  Reading {len(files)} frames...")
    frames = [read_vtu(f) for f in files]

    # Find global velocity range for consistent colorbar
    vmin = 0.0
    vmax = max(f['point_data']['velocity_magnitude'].max() for f in frames)
    vmax = max(vmax, 0.1)  # Ensure minimum range

    fig, ax = plt.subplots(figsize=(12, 9))

    # Create initial plot
    vtu_data = frames[0]
    points = vtu_data['points']
    connectivity = vtu_data['connectivity']
    offsets = vtu_data['offsets']

    # Build polygons once (mesh doesn't change)
    polygons = []
    prev_offset = 0
    for offset in offsets:
        cell_points = connectivity[prev_offset:offset]
        poly = points[cell_points, :2]
        polygons.append(poly)
        prev_offset = offset

    # Initial colors
    field = vtu_data['point_data']['velocity_magnitude']
    colors = []
    prev_offset = 0
    for offset in offsets:
        cell_points = connectivity[prev_offset:offset]
        colors.append(np.mean(field[cell_points]))
        prev_offset = offset

    collection = PolyCollection(polygons, array=np.array(colors), cmap='Blues',
                                edgecolors='face', clim=(vmin, vmax))
    ax.add_collection(collection)
    ax.autoscale()
    ax.set_aspect('equal')

    cbar = plt.colorbar(collection, ax=ax, label='Velocity (m/s)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    title = ax.set_title(f'Current Speed at t = {vtu_data["time"]/3600:.2f} h')

    def update(frame_idx):
        vtu_data = frames[frame_idx]
        field = vtu_data['point_data']['velocity_magnitude']

        # Update colors
        colors = []
        prev_offset = 0
        for offset in offsets:
            cell_points = connectivity[prev_offset:offset]
            colors.append(np.mean(field[cell_points]))
            prev_offset = offset

        collection.set_array(np.array(colors))
        title.set_text(f'Current Speed at t = {vtu_data["time"]/3600:.2f} h')
        return [collection, title]

    print(f"  Rendering {len(frames)} frames...")
    anim = FuncAnimation(fig, update, frames=len(frames), interval=400, blit=True)
    anim.save(output_file, writer=PillowWriter(fps=3))
    plt.close()
    print(f"Animation saved to: {output_file}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]

    # Get list of VTU files
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.vtu')))
    else:
        files = [path]

    if not files:
        print(f"No VTU files found in {path}")
        sys.exit(1)

    print(f"Found {len(files)} VTU file(s)")

    # Read and summarize
    for f in files:
        print(f"\nReading: {f}")
        vtu_data = read_vtu(f)
        print_summary(vtu_data)

    # Plot the last frame
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        vtu_data = read_vtu(files[-1])

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot available fields
        fields_to_plot = ['h', 'velocity_magnitude', 'bathymetry', 'eta']
        available = [f for f in fields_to_plot if f in vtu_data['point_data']]

        cmaps = {'h': 'Blues', 'velocity_magnitude': 'Reds', 'bathymetry': 'terrain', 'eta': 'RdBu_r'}
        titles = {'h': 'Water Depth h (m)', 'velocity_magnitude': 'Velocity |u| (m/s)',
                  'bathymetry': 'Bathymetry B (m)', 'eta': 'Surface Elevation η (m)'}

        for i, (ax, field) in enumerate(zip(axes.flat, available)):
            plot_field(vtu_data, field, ax=ax, cmap=cmaps.get(field, 'viridis'),
                      title=titles.get(field, field))
            # Add velocity vectors to velocity plot
            if field == 'velocity_magnitude':
                plot_velocity_vectors(vtu_data, ax, skip=6, scale=8)

        # Hide unused axes
        for i in range(len(available), 4):
            axes.flat[i].set_visible(False)

        plt.suptitle(f'Froya Simulation at t = {vtu_data["time"]/3600:.2f} hours', fontsize=14)
        plt.tight_layout()

        output_file = 'froya_visualization.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_file}")

        # Create velocity-only plot with vectors for better clarity
        fig2, ax2 = plt.subplots(figsize=(14, 10))
        plot_field(vtu_data, 'velocity_magnitude', ax=ax2, cmap='Blues',
                  title=f'Flow Field at t = {vtu_data["time"]/3600:.2f} hours')
        plot_velocity_vectors(vtu_data, ax2, skip=4, scale=6, color='red')
        plt.savefig('froya_flow_vectors.png', dpi=150, bbox_inches='tight')
        print("Flow vectors saved to: froya_flow_vectors.png")

        # Also create a time series plot if multiple frames
        if len(files) > 1:
            times = []
            max_velocities = []
            mean_depths = []

            for f in files:
                data = read_vtu(f)
                times.append(data['time'] / 3600)  # hours
                if 'velocity_magnitude' in data['point_data']:
                    max_velocities.append(data['point_data']['velocity_magnitude'].max())
                if 'h' in data['point_data']:
                    mean_depths.append(data['point_data']['h'].mean())

            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            if max_velocities:
                ax1.plot(times, max_velocities, 'b-o', linewidth=2, markersize=4)
                ax1.set_ylabel('Max Velocity (m/s)')
                ax1.set_title('Time Series')
                ax1.grid(True, alpha=0.3)

            if mean_depths:
                ax2.plot(times, mean_depths, 'g-o', linewidth=2, markersize=4)
                ax2.set_ylabel('Mean Depth (m)')
                ax2.set_xlabel('Time (hours)')
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            ts_file = 'froya_timeseries.png'
            plt.savefig(ts_file, dpi=150, bbox_inches='tight')
            print(f"Time series saved to: {ts_file}")

            # Create animated GIF
            print("\nCreating animation...")
            create_animation(files, 'froya_animation.gif')

    except ImportError as e:
        print(f"\nMatplotlib not available: {e}")
        print("Install with: pip install matplotlib")

if __name__ == '__main__':
    main()
