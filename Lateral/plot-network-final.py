import numpy as np
import matplotlib.pyplot as plt

def plot_network(filename):
    nodes = {}
    bonds = []
    mode = "nodes"
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip blank lines
            if not line:
                continue
            # Switch mode if we hit the bonds header
            if line.startswith("#"):
                if "Bonds" in line:
                    mode = "bonds"
                continue
            parts = line.split()
            if mode == "nodes":
                idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                nodes[idx] = (x, y)
            elif mode == "bonds":
                # We assume bonds are given as: i j tag
                i = int(parts[0])
                j = int(parts[1])
                bonds.append((i, j))
    
    # Plot nodes
    xs = [pos[0] for pos in nodes.values()]
    ys = [pos[1] for pos in nodes.values()]
    
    plt.figure(figsize=(12,12))
    plt.scatter(xs, ys, c='r', s=0.4, zorder=2)
    
    # Plot bonds
    for (i, j) in bonds:
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]
        plt.plot([x1, x2], [y1, y2], 'b-', lw=0.8, zorder=1)
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Network from {filename}")
    #plt.axis("equal")
    plt.grid(True, which='both', ls="--", alpha=1.0)
    
    
    # Save the plot with the same basename as the data file and .png extension.
    #base, _ = os.path.splitext(filename)
    #outname = base + ".png"
    
    plt.savefig("final_network_px_0.9_py_1.png", dpi=300)
    plt.show()
    plt.close()

# Example usage:
if __name__ == '__main__':
    # Change the filename as needed.
    plot_network("final_network_px_0.9_py_1.dat")

