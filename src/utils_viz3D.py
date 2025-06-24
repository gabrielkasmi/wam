"""
Necessary functions to visualize 3D point clouds and their explanations.
"""

import pandas as pd
import numpy as np

from plotly.subplots import make_subplots
import plotly.express as px
import plotly
import plotly.graph_objects as go
from plotly.offline import iplot



def _make_voxel_figure(data1, data2, data3, cmap1, cmap2, cmap3, output_file=None):
    """
    function for the plotting the superimposed heatmap
    """
    # Prepare VoxelData for each dataset
    Voxels1 = VoxelData(data1)
    Voxels2 = VoxelData(data2)
    Voxels3 = VoxelData(data3)

    # Layout for the figure
    layout = go.Layout(
        height=500, width=600,
    )

    # Create a figure and add three datasets
    fig = go.Figure(layout=layout)

    # Add data1
    fig.add_trace(go.Mesh3d(
        x=Voxels1.vertices[0],
        y=Voxels1.vertices[1],
        z=Voxels1.vertices[2],
        i=Voxels1.triangles[0],
        j=Voxels1.triangles[1],
        k=Voxels1.triangles[2],
        intensity=Voxels1.intensity,
        colorscale=cmap1,
        showscale=False,
        opacity=0.9
    ))

    # Add data2
    fig.add_trace(go.Mesh3d(
        x=Voxels2.vertices[0],
        y=Voxels2.vertices[1],
        z=Voxels2.vertices[2],
        i=Voxels2.triangles[0],
        j=Voxels2.triangles[1],
        k=Voxels2.triangles[2],
        intensity=Voxels2.intensity,
        colorscale=cmap2,
        showscale=False,
        opacity=0.4
    ))

    # Add data3
    fig.add_trace(go.Mesh3d(
        x=Voxels3.vertices[0],
        y=Voxels3.vertices[1],
        z=Voxels3.vertices[2],
        i=Voxels3.triangles[0],
        j=Voxels3.triangles[1],
        k=Voxels3.triangles[2],
        intensity=Voxels3.intensity,
        colorscale=cmap3,
        showscale=False,
        opacity=0.2
    ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        font=dict(size=18),
        margin=dict(l=0, r=0, b=0, t=0),
        scene_camera=dict(eye=dict(x=0, y=0, z=-2)),
    )

    # Save to output file if provided
    if output_file is not None:
        fig.write_image(output_file, scale=2)

    # Show the plot
    fig.show()


def scatter3D(pc, label): 
    """
    3D scatter plot of the given point cloud.
        pc: numpy array of shape (n, 3)
        label: int, label of the digit
    """
    trace1 = go.Scatter3d(
        x=pc[:,0],
        y=pc[:,1],
        z=pc[:,2],
        mode='markers',
        marker=dict(size=12, color=pc, colorscale='Viridis', opacity=0.7),
    )

    data = [trace1]

    layout = go.Layout(
        height=500, width=600, title=f'Digit: {str(label.item())} in 3D space'
    )
    
    fig = go.Figure(data=data, layout=layout)

    fig.update_layout(
            scene = dict(
                xaxis = dict(visible=False),
                yaxis = dict(visible=False),
                zaxis =dict(visible=False), 
            ),
        )
    fig.update_traces(marker_size = 4)

    iplot(fig)



def scatter3D_batch(batch_pcs, labels, num_per_rows = 2, camera= None):
    """
    Plot batch of point clouds next to each others
        batch_pcs: np.array with shape (B,N,3) where N is the number of points and B the number of digits to plot 
        labels: list of labels for each digit (B, 1)
        num_per_rows: number of digits to plot per row
        camera: camera position for the plot
    """
    num_plots = batch_pcs.shape[0]
    if num_plots > num_per_rows:
        num_rows = int(np.ceil(num_plots/num_per_rows))
        num_cols = num_per_rows
        specs=[[{'type': 'scatter3d'} for _ in range(num_per_rows)] for _ in range(num_rows)]
    else: 
        num_rows = 1
        num_cols = num_plots
        specs = [[{'type': 'scatter3d'} for _ in range(num_plots)]]

    fig = make_subplots(rows=num_rows, cols=num_cols, 
                        subplot_titles=[f'Digit: {str(labels[i].item())} in 3D space' for i in range(num_plots)],
                        specs=specs,)
    
    for c in range(num_cols):
        for r in range(num_rows):
            i = c + r*num_cols
            if i < num_plots:
                trace1 = go.Scatter3d(
                    x=batch_pcs[i][:,0],
                    y=batch_pcs[i][:,1],
                    z=batch_pcs[i][:,2],
                    mode='markers',
                    marker=dict(size=4, color=batch_pcs[i], colorscale='Viridis', opacity=0.7),
                )
                fig.add_trace(trace1, row=r+1, col=c+1)
                
    # add camera for each trace (avoid cropping)
    if camera is None:
        camera = dict(
            eye=dict(x=1.75, y=1.75, z=0.1)
        )
    for i in range(num_plots):
        scene = f'scene{i+1}'
        fig.update_layout({scene: dict(camera=camera)})

    fig.update_layout(showlegend=False)# hovermode='closest')
    fig.update_layout(margin=dict(l=30.0, r=30.0, b=80.0, t=50.0)) # manage white spaces around the plots
    fig.show()


def scatter3D_superpose(pc1, pc2, label): 
    """
    Plot two point clouds pc1 and pc2 on the same plot.
        pc1: numpy array of shape (n, 3)
        pc2: numpy array of shape (n, 3)
        label: int, label of the digit
    """

    trace1 = go.Scatter3d(
        x=pc1[:, 0],
        y=pc1[:, 1],
        z=pc1[:, 2],
        mode='markers',
        marker=dict(size=12, color='red', opacity=0.7), 
        name='Original Point Cloud'
    )
    
    trace2 = go.Scatter3d(
        x=pc2[:, 0],
        y=pc2[:, 1],
        z=pc2[:, 2],
        mode='markers',
        marker=dict(size=12, color='blue', opacity=0.7), 
        name='Important Points'
    )

    data = [trace1, trace2]

    layout = go.Layout(
        height=500, width=600, title=f'Digit: {str(label.item())} in 3D space'
    )
    
    fig = go.Figure(data=data, layout=layout)
    
    fig.update_layout(
            scene = dict(
                xaxis = dict(visible=False),
                yaxis = dict(visible=False),
                zaxis =dict(visible=False), 
            ),
        )
    fig.update_traces(marker_size = 4)

    iplot(fig)

def scatter3d_colors(pc, label): 
        """
        Plot the explanations where the 4th channel contains importance values.
            pc: numpy array of shape (n, 4)
            label: int, label of the digit
            colorbar: bool, whether to show the colorbar or not
        """
        assert pc.shape[-1] == 4, "The plot should have 4 columns (x, y, z, val)"

        plot_df = pd.DataFrame(pc, columns=["x", "y", "z", "val"])
        plot_df = plot_df.loc[plot_df["val"] > 0]


        layout = go.Layout(
                height=500, width=600, title=f'Digit: {str(label.item())} in 3D space', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
        ) 

        fig = go.Figure(data=px.scatter_3d(plot_df, x=plot_df['x'], y=plot_df['y'], z=plot_df['z'],
                color= plot_df['val'], opacity=0.8, 
                color_continuous_scale  = plotly.colors.sequential.Viridis, 
                ),
        )
        
        fig.update_layout(layout)
        fig.update_layout(coloraxis_colorbar=dict(title="Values", tickfont=dict(size=10), tickmode='array'))
        fig.update_layout(
            scene = dict(
                xaxis = dict(visible=False),
                yaxis = dict(visible=False),
                zaxis =dict(visible=False), 
            ),
        )
        fig.update_traces(marker_size = 4)

        fig.show()


def scatter3D_explanationBatch(pcs, labels, camera= None):
    """
    Plot batch of pcs with their explanations next to each others
        pcs: list of np.arrays with shape (N,3) where N is the number of points 
        labels: list of labels for each digit (B, 1)
        camera: camera position for the plot
    """
    assert len(pcs) == 2, "Error, please use list of 2 np.arrays only for WAM visualization"

    num_cols = len(pcs)
    num_digits = pcs[0].shape[0]
    
    num_rows = num_digits
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}] for _ in range(num_digits)]
    labels = [label for label in labels for _ in range(2)]

    num_plots = num_rows * num_cols
    fig = make_subplots(rows=num_rows, cols=num_cols, 
                        subplot_titles=[f'Digit: {str(labels[i])} in 3D space' for i in range(num_plots)],
                        specs=specs,)
    
    for c in range(num_cols):
        for r in range(num_rows):
            trace1 = go.Scatter3d(
                x=pcs[c][r,:,0],
                y=pcs[c][r,:,1],
                z=pcs[c][r,:,2],
                mode='markers',
                marker=dict(size=12, color=pcs[c][r], colorscale='Viridis', opacity=0.7),
            )
            fig.add_trace(trace1, row=r+1, col=c+1)
                
    # add camera for each trace (avoid cropping)
    if camera is None:
        camera = dict(
            eye=dict(x=1.75, y=1.75, z=0.1)
        )
    for i in range(num_plots):
        scene = f'scene{i+1}'
        fig.update_layout({scene: dict(camera=camera)})
   
    fig.update_layout(showlegend=False)# hovermode='closest')
    fig.update_layout(margin=dict(l=30.0, r=30.0, b=80.0, t=50.0)) # manage white spaces around the plots
    fig.update_layout(
            scene = dict(
                xaxis = dict(visible=False),
                yaxis = dict(visible=False),
                zaxis =dict(visible=False), 
                #camera_eye=dict(x=1.85, y=1.85, z=1)
            ),
        )
    fig.update_traces(marker_size = 4)
    
    fig.show()

def get_plotly_cmap(rgb_values):
    """
    Create a custom cmap for plotly using given rgb_values.
    """
    cmap = [[0, 'rgb(255,255,255)'], [1, f'rgb{rgb_values}']]
    return cmap

class Coordinate():

    def __init__(self, xyz):
        self.x = xyz[0]
        self.y = xyz[1]
        self.z = xyz[2]


class VoxelData():
    # Code adapted from  https://github.com/olive004/Plotly-voxel-renderer/tree/master
    """
    Code to plot voxel volume in plotly.
    """

    def __init__(self,data):
        self.data = data
        self.triangles = np.zeros((np.size(np.shape(self.data)),1)) 
        self.xyz = self.get_coords()
        # self.x = self.xyz[0,:]
        # self.y = self.xyz[1,:]
        # self.z = self.xyz[2,:]
        self.x_length = np.size(data,0)
        self.y_length = np.size(data,1)
        self.z_length = np.size(data,2)
        self.vert_count = 0
        self.intensity = []
        self.vertices = self.make_edge_verts()
        self.triangles = np.delete(self.triangles, 0,1)
        self.intensity = np.array(self.intensity)
        #self.make_triangles()


    def get_coords(self):
        indices = np.nonzero(self.data)
        indices = np.stack((indices[0], indices[1],indices[2]))
        return indices

    def has_voxel(self,neighbor_coord):
        return self.data[neighbor_coord[0],neighbor_coord[1],neighbor_coord[2]]


    def get_neighbor(self, voxel_coords, direction):
        x = voxel_coords[0]
        y = voxel_coords[1]
        z = voxel_coords[2]
        offset_to_check = CubeData.offsets[direction]
        neighbor_coord = [x+ offset_to_check[0], y+offset_to_check[1], z+offset_to_check[2]]

        # return 0 if neighbor out of bounds or nonexistent
        if (any(np.less(neighbor_coord,0)) | (neighbor_coord[0] >= self.x_length) | (neighbor_coord[1] >= self.y_length) | (neighbor_coord[2] >= self.z_length)):
            return 0
        else:
            return self.has_voxel(neighbor_coord)


    def remove_redundant_coords(self, cube):
        i = 0
        while(i < np.size(cube,1)):
            coord = (cube.T)[i]
            cu = cube[:, cube[0,:] == coord[0]]
            cu = cu[:, cu[1,:] == coord[1]]
            cu = cu[:, cu[2,:] == coord[2]]
            # if more than one coord of same value, delete
            if i >= np.size(cube,1):
                break
            if np.size(cu, 1) >1:
                cube = np.delete(cube, i, 1) 
                i = i-1
            i+=1    
        return cube

    
    def make_face(self, voxel, direction):
        voxel_coords = self.xyz[:, voxel]
        explicit_dir = CubeData.direction[direction]
        vert_order = CubeData.face_triangles[explicit_dir]

        # Use if triangle order gets fixed
        # next_triangles = np.add(vert_order, voxel)
        # next_i = [next_triangles[0], next_triangles[0]]
        # next_j = [next_triangles[1], next_triangles[2]]
        # next_k = [next_triangles[2], next_triangles[3]]
        
        next_i = [self.vert_count, self.vert_count]
        next_j = [self.vert_count+1, self.vert_count+2]
        next_k = [self.vert_count+2, self.vert_count+3]

        next_tri = np.vstack((next_i, next_j, next_k))
        self.triangles = np.hstack((self.triangles, next_tri))
        # self.triangles = np.vstack((self.triangles, next_triangles))

        face_verts = np.zeros((len(voxel_coords),len(vert_order)))
        for i in range(len(vert_order)):
            face_verts[:,i] = voxel_coords + CubeData.cube_verts[vert_order[i]]   

        self.vert_count = self.vert_count+4       
        return face_verts


    def make_cube_verts(self, voxel):
        voxel_coords = self.xyz[:, voxel]
        cube = np.zeros((len(voxel_coords), 1))

        # only make a new face if there's no neighbor in that direction
        dirs_no_neighbor = []
        for direction in range(len(CubeData.direction)):
            if np.any(self.get_neighbor(voxel_coords, direction)):
                continue
            else: 
                dirs_no_neighbor = np.append(dirs_no_neighbor, direction)
                face = self.make_face(voxel, direction)
                cube = np.append(cube,face, axis=1)

        # remove cube initialization
        cube = np.delete(cube, 0, 1) 

        # remove redundant entries: not doing this cuz it messes up the triangle order
        # and i'm too lazy to fix that so excess vertices it is
        # cube = self.remove_redundant_coords(cube)
        return cube


    def make_edge_verts(self):
        # make only outer vertices 
        edge_verts = np.zeros((np.size(self.xyz, 0),1))
        num_voxels = np.size(self.xyz, 1)
        for voxel in range(num_voxels):
            cube = self.make_cube_verts(voxel)          # passing voxel num rather than 
            edge_verts = np.append(edge_verts, cube, axis=1)
            val = self.data[self.xyz[0,voxel], self.xyz[1,voxel], self.xyz[2,voxel]]
            self.intensity.extend([val]*np.size(cube,1))
        edge_verts = np.delete(edge_verts, 0,1)
        return edge_verts        

    
class CubeData:
    # all data and knowledge from https://github.com/boardtobits/procedural-mesh-tutorial/blob/master/CubeMeshData.cs
    # for creating faces correctly by direction
    face_triangles = {
		'North':  [0, 1, 2, 3 ],        # +y
        'East': [ 5, 0, 3, 6 ],         # +x
	    'South': [ 4, 5, 6, 7 ],        # -y
        'West': [ 1, 4, 7, 2 ],         # -x
        'Up': [ 5, 4, 1, 0 ],           # +z
        'Down': [ 3, 2, 7, 6 ]          # -z
	}

    cube_verts = [
        [1,1,1],
        [0,1,1], 
        [0,1,0],
        [1,1,0],
        [0,0,1],
        [1,0,1],
        [1,0,0],
        [0,0,0],
    ]

    # cool twist
    # cube_verts = [
    #     [0,0,0],
    #     [1,0,0],
    #     [1,0,1], 
    #     [0,0,1],
    #     [0,1,1], 
    #     [1,1,1],
    #     [1,1,0],
    #     [0,1,0],
    # ]

    # og
    # cube_verts = [
    #     [1,1,1],
    #     [0,1,1], 
    #     [0,0,1],
    #     [1,0,1], 
    #     [0,1,0],
    #     [1,1,0],
    #     [1,0,0],
    #     [0,0,0]
    # ]

    direction = [
        'North',
        'East',
        'South',
        'West',
        'Up',
        'Down'
    ]

    opposing_directions = [
        ['North','South'],
        ['East','West'], 
        ['Up', 'Down']
    ]

    # xyz direction corresponding to 'Direction'
    offsets = [  
        [0, 1, 0], 
        [1, 0, 0],   
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]
    # offsets = [             
    #     [0, 0, 1],
    #     [1, 0, 0],
    #     [0, 0, -1],
    #     [-1, 0, 0],
    #     [0, 1, 0],
    #     [0, -1, 0]
    # ]


def make_voxel_figure(data, cmap, output_file=None):
    """
    Plot a voxel grid in a given colormap. 
        data: numpy array of shape [x,y,z] (voxels)
        cmap: colormap to use
        output_file: name of the file to save the plot
    """
    Voxels = VoxelData(data)
    layout = go.Layout(
        height=500, width=600,
    )
    fig = go.Figure(data=go.Mesh3d(
        x=Voxels.vertices[0],
        y=Voxels.vertices[1],
        z=Voxels.vertices[2],
        i=Voxels.triangles[0],
        j=Voxels.triangles[1],
        k=Voxels.triangles[2],
        intensity=Voxels.intensity,
        showscale=False,
        colorscale=cmap,
        opacity=0.5
    ), layout=layout)
    fig.update_layout(
            scene = dict(
                xaxis = dict(visible=False),
                yaxis = dict(visible=False),
                zaxis =dict(visible=False)
            ),
        )

    # update camera
    fig.update_layout(scene_camera=dict(eye=dict(x=0, y=0, z=-2)))

    # # increase font size
    fig.update_layout(font=dict(size=18))
    
    # reduce margin 
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    if output_file is not None:
        fig.write_image(output_file,scale=2)

    fig.show()


def make_voxel_superpose(data, heatmap, cmap_digit='Blues', cmap_heatmap='Viridis', threshold = 0.3, output_file=None):
    """"
    Plot the digit with a heatmap on top.
        data: numpy array of shape [x,y,z] representing the voxelized digit
        heatmap: numpy array of shape [x,y,z] representing the heatmap
        cmap_digit: colormap to use for the digit
        cmap_heatmap: colormap to use for the heatmap
        threshold: threshold to apply to the heatmap
        output_file: name of the file to save the plot
    """

    layout = go.Layout(
        height=500, width=600,
    )

    Voxels = VoxelData(data)

    # keep only areas > threshold
    heatmap = np.where(heatmap > threshold, heatmap, 0)
    heatmap = VoxelData(heatmap)

    # use add trace to add the different layers
    fig = go.Figure(layout=layout)

    # add the original shape
    fig.add_trace(go.Mesh3d(
        x=Voxels.vertices[0],
        y=Voxels.vertices[1],
        z=Voxels.vertices[2],
        i=Voxels.triangles[0],
        j=Voxels.triangles[1],
        k=Voxels.triangles[2],
        intensity=Voxels.intensity,
        colorscale=cmap_digit,
        opacity=0.95, 
        showscale=False)
    )

    # add the heatmap
    fig.add_trace(go.Mesh3d(
        x=heatmap.vertices[0],
        y=heatmap.vertices[1],
        z=heatmap.vertices[2],
        i=heatmap.triangles[0],
        j=heatmap.triangles[1],
        k=heatmap.triangles[2],
        intensity=heatmap.intensity,
        colorscale=cmap_heatmap,
        opacity=0.4, 
        showscale=False)    
    )
    

    fig.update_layout(
            scene = dict(
                xaxis = dict(visible=False),
                yaxis = dict(visible=False),
                zaxis =dict(visible=False), 
            ),
        )

    # update camera
    fig.update_layout(scene_camera=dict(eye=dict(x=0, y=0, z=-2)))

    # reduce margin 
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    if output_file is not None:
        fig.write_image(output_file,scale=2)

    fig.show()
