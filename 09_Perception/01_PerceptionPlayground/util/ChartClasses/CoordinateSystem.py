import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.artist import Artist


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)
    
    
class Frame:
    
    x = [1, 0, 0]
    y = [0, 1, 0]
    z = [0, 0, 1]
    
    translation = [0,0,0]
    
    xColor = 'r'
    yColor = 'b'
    zColor = 'g'
    
    drawAxisLabelDist = 0.1
    
    # Here we create the arrows:
    arrow_prop_dict = dict(mutation_scale=20, 
                           arrowstyle='->', 
                           shrinkA=0, 
                           shrinkB=0)
   
    def setTranslation(self, translation):
        self.translation = translation
    
    def setAttitude(self, dcm):
        self.dcm = dcm
        
    def setPose(self, dcm, translation):
        self.translation = translation
        self.dcm = dcm
        
    def __init__(self, dcm=np.eye(3, dtype=float), t=[0,0,0], arrowLength=1):
        self.x *= arrowLength
        self.y *= arrowLength
        self.z *= arrowLength
        self.arrowLength = arrowLength
        self.drawAxisLabelDist = arrowLength * self.drawAxisLabelDist
        
        self.dcm = dcm
        self.translation = t
        
        self.x = np.dot(dcm, self.x)
        self.y = np.dot(dcm, self.y)
        self.z = np.dot(dcm, self.z)
        
        
    def draw(self, ax):
        """
        Draw coordinate frame axes to ax

        Args:
            ax (_type_): Matplotlib axis 
        """
        self._removeAllArtist(ax)
        
        self.xArtist = Arrow3D([0, self.x[0]], 
                    [0, self.x[1]], 
                    [0, self.x[2]], 
                    **self.arrow_prop_dict, 
                    color=self.xColor)
        ax.add_artist(self.xArtist)

        self.yArtist = Arrow3D([0, self.y[0]], 
                        [0, self.y[1]], 
                        [0, self.y[2]], 
                        **self.arrow_prop_dict, 
                        color=self.tColor)
        ax.add_artist(self.yArtist)
        
        self.zArtist = Arrow3D([0, self.z[0]], 
                        [0, self.z[1]], 
                        [0, self.z[2]], 
                        **self.arrow_prop_dict, 
                        color=self.zColor)
        ax.add_artist(self.zArtist)
        
        # Draw axis and frame origin labels
        ax.text(self.translation[0], 
                self.translation[1], 
                self.translation[2], 
                r'$0$')
        ax.text(self.x[0] + self.drawAxisLabelDist, 
                self.x[1], 
                self.x[2], 
                r'$x$')
        ax.text(self.y[0], 
                self.y[1] + self.drawAxisLabelDist, 
                self.y[2], 
                r'$y$')
        ax.text(self.z[0], 
                self.z[1], 
                self.z[2] + self.drawAxisLabelDist, 
                r'$z$')
        return ax
        
    def _removeAllArtist(self, ax):
        self.xArtist.remove()
        self.yArtist.remove()
        self.zArtist.remove()
        # Artist.remove(ax)