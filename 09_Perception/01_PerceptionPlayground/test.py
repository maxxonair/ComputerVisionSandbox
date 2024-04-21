import tkinter as tk

def resize_panes(event):
    # Set a fixed width for the right pane
    pane.paneconfig(right_frame, minsize=100)

root = tk.Tk()
root.title("PanedWindow Example")

# Create a PanedWindow
pane = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashwidth=5, sashrelief=tk.RAISED)
pane.pack(expand=True, fill=tk.BOTH)

# Left frame
left_frame = tk.Frame(pane, background="light blue")
pane.add(left_frame, minsize=100)

# Right frame
right_frame = tk.Frame(pane, background="light green")
pane.add(right_frame, minsize=100)

# Bind event to resize panes when the window is resized
root.bind("<Configure>", resize_panes)

root.mainloop()