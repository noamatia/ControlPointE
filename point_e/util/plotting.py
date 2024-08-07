from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import mitsuba as mi

from .point_cloud import PointCloud


def plot_point_cloud(
    pc: PointCloud,
    color: bool = True,
    grid_size: int = 1,
    fixed_bounds: Optional[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ] = (
        (-0.5, -0.5, -0.5),
        (0.5, 0.5, 0.5),
    ),
    theta: float = None,
    color_by_distance: bool = False,
):
    """
    Render a point cloud as a plot to the given image path.

    :param pc: the PointCloud to plot.
    :param image_path: the path to save the image, with a file extension.
    :param color: if True, show the RGB colors from the point cloud.
    :param grid_size: the number of random rotations to render.
    """
    fig = plt.figure(figsize=(8, 8))

    for i in range(grid_size):
        for j in range(grid_size):
            ax = fig.add_subplot(
                grid_size, grid_size, 1 + j + i * grid_size, projection="3d"
            )
            color_args = {}
            c = pc.coords
            if color_by_distance:
                distances = np.linalg.norm(c, axis=1)
                max_distance = np.max(distances)
                min_distance = np.min(distances)
                normalized_distances = (distances - min_distance) / (
                    max_distance - min_distance
                )
                cmap = plt.cm.jet
                rgba_values = cmap(normalized_distances)
                color_args["c"] = np.stack(
                    [rgba_values[:, 0], rgba_values[:, 1], rgba_values[:, 2]], axis=-1
                )
            elif color:
                color_args["c"] = np.stack(
                    [pc.channels["R"], pc.channels["G"], pc.channels["B"]], axis=-1
                )

            if grid_size > 1:
                theta = np.pi * 2 * (i * grid_size + j) / (grid_size**2)
            if theta is not None:
                rotation = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                c = c @ rotation

            ax.scatter(c[:, 0], c[:, 1], c[:, 2], **color_args)

            if fixed_bounds is None:
                min_point = c.min(0)
                max_point = c.max(0)
                size = (max_point - min_point).max() / 2
                center = (min_point + max_point) / 2
                ax.set_xlim3d(center[0] - size, center[0] + size)
                ax.set_ylim3d(center[1] - size, center[1] + size)
                ax.set_zlim3d(center[2] - size, center[2] + size)
            else:
                ax.set_xlim3d(fixed_bounds[0][0], fixed_bounds[1][0])
                ax.set_ylim3d(fixed_bounds[0][1], fixed_bounds[1][1])
                ax.set_zlim3d(fixed_bounds[0][2], fixed_bounds[1][2])

    return fig


# source: https://github.com/hasancaslan/BeautifulPointCloud


class XMLTemplates:
    HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1920"/>
            <integer name="height" value="1080"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
"""
    BALL_SEGMENT = """
    <shape type="sphere">
        <float name="radius" value="0.015"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""
    TAIL = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""


def render_point_cloud(
    pc: PointCloud,
    theta: float = None,
    output_path: str = None,
):
    c = pc.coords
    if theta is not None:
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        c = c @ rotation
    xml_segments = [XMLTemplates.HEAD]
    for point in c:
        color = np.clip(
            np.array([point[0] + 0.5, point[1] + 0.5, point[2] + 0.5 - 0.0125]),
            0.001,
            1.0,
        )
        color /= np.linalg.norm(color)
        xml_segments.append(
            XMLTemplates.BALL_SEGMENT.format(point[0], point[1], point[2], *color)
        )
    xml_segments.append(XMLTemplates.TAIL)
    xml_content = "".join(xml_segments)
    mi.set_variant("scalar_rgb")
    scene = mi.load_string(xml_content)
    img = mi.render(scene)
    if output_path is not None:
        mi.util.write_bitmap(output_path, img)
    img = np.array(img)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img
