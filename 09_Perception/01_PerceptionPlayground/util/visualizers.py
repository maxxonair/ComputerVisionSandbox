import matplotlib.pyplot as plt
GRID_LINE_THICKNESS = 0.1

def plotStereoImageRectification(imgl, imgr, rimgl, rimgr, enableSaveToFile):

    fig = plt.figure()

    ax1 = plt.subplot(2,2,1)
    ax1.set_title('Raw Left Image')
    ax1.imshow(imgl, cmap='gray', vmin=0, vmax=255)
    ax1.set_xlabel("x / px")
    ax1.set_ylabel("y / px")  
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()
    ax1.grid(linestyle='-', linewidth=GRID_LINE_THICKNESS)

    ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
    ax2.set_title('Raw Right Image')
    ax2.imshow(imgr, cmap='gray', vmin=0, vmax=255)
    ax2.set_xlabel("x / px")
    ax2.set_ylabel("y / px")  
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax2.grid(linestyle='-', linewidth=GRID_LINE_THICKNESS)

    ax3 = plt.subplot(2,2,3)
    ax3.set_title('Rectified Left Image')
    ax3.imshow(rimgl, cmap='gray', vmin=0, vmax=255)
    ax3.set_xlabel("x / px")
    ax3.set_ylabel("y / px")  
    ax3.xaxis.set_label_position('top')
    ax3.xaxis.tick_top()
    ax3.grid(linestyle='-', linewidth=GRID_LINE_THICKNESS)

    ax4 = plt.subplot(224, sharex=ax3, sharey=ax3)
    ax4.set_title('Rectified Right Image')
    ax4.imshow(rimgr, cmap='gray', vmin=0, vmax=255)
    ax4.set_xlabel("x / px")
    ax4.set_ylabel("y / px")  
    ax4.xaxis.set_label_position('top')
    ax4.xaxis.tick_top()
    ax4.grid(linestyle='-', linewidth=GRID_LINE_THICKNESS)
    
    plt.tight_layout()
    
    def on_resize(event):
        fig.tight_layout()
        fig.canvas.draw()

    cid = fig.canvas.mpl_connect('resize_event', on_resize)
    if enableSaveToFile:
        plt.savefig('./output/stereo_image_rect_and_raw.png')
    else:
        plt.show()

def plotStereoImage(imgl, imgr, enableSaveToFile):

    fig = plt.figure(figsize = (11, 5))

    ax1 = plt.subplot(1,2,1)
    ax1.imshow(imgl, cmap='gray', vmin=0, vmax=255)
    ax1.set_xlabel("x / px")
    ax1.set_ylabel("y / px")  
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()
    ax1.grid(linestyle='-', linewidth=GRID_LINE_THICKNESS)

    ax2 = plt.subplot(1,2,2)
    ax2.imshow(imgr, cmap='gray', vmin=0, vmax=255)
    ax2.set_xlabel("x / px")
    ax2.set_ylabel("y / px")  
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax2.grid(linestyle='-', linewidth=GRID_LINE_THICKNESS)

    fig.tight_layout()
    if enableSaveToFile:
        plt.savefig('./output/stereo_image.png')
    else:
        plt.show()

def plotDisparityMap(imgl, imgr, dispMap, depthMap, enableSaveToFile):
    fig = plt.figure(figsize = (11, 5))

    ax1 = plt.subplot(2,2,1)
    ax1.set_title('Rectified Left Image')
    ax1.imshow(imgl, cmap='gray', vmin=0, vmax=255)
    ax1.set_xlabel("x / px")
    ax1.set_ylabel("y / px")  
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()
    ax1.grid(linestyle='-', linewidth=GRID_LINE_THICKNESS)

    ax2 = plt.subplot(2,2,2)
    ax2.set_title('Rectified Right Image')
    ax2.imshow(imgr, cmap='gray', vmin=0, vmax=255)
    ax2.set_xlabel("x / px")
    ax2.set_ylabel("y / px")  
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax2.grid(linestyle='-', linewidth=GRID_LINE_THICKNESS)

    ax3 = plt.subplot(2,2,3)
    ax3.set_title('Disparity Map')
    dispPlot = ax3.imshow(dispMap, cmap='inferno', vmin=0, vmax=255)
    ax3.set_xlabel("x / px")
    ax3.set_ylabel("y / px")  
    bar = plt.colorbar(dispPlot)

    ax4 = plt.subplot(2,2,4)
    ax4.set_title('Depth Map')
    depthPlot = ax4.imshow(depthMap, cmap='inferno_r', vmax=5)
    ax4.set_xlabel("x / px")
    ax4.set_ylabel("y / px")  
    bar = plt.colorbar(depthPlot)

    fig.tight_layout()
    if enableSaveToFile:
        plt.savefig('./output/stereo_image.png')
    else:
        plt.show()