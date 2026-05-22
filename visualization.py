import matplotlib.pyplot as plt 
import numpy as np 

def visualize(agent, title="Gridworld"):
    env = agent.env
    V = agent.V
    P = agent.P

    fig, ax = plt.subplots(figsize = (env.cols, env.rows))

    im = ax.imshow(V, cmap='PiYG', origin='upper', alpha=0.5)
    plt.colorbar(im, ax=ax, label='V(s)')

    arrows = {
        0: (0, -0.3), #up
        1: (0, 0.3), #down
        2: (-0.3, 0), #left
        3: (0.3, 0) #right
    }

    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) == env.goal_state:
                ax.text(c, r, 'goal', ha='center', va='center', color='black', fontsize=10, fontweight='bold')
                continue
            if (r, c) == (env.start_row, env.start_col):
                ax.text(c, r - 0.35, 'start', ha='center', va='center', color='black', fontsize=10, fontweight='bold')

            for action, (dx, dy) in arrows.items():
                if P[action, r, c] > 0:
                    ax.arrow(c, r, dx, dy, head_width=0.05, head_length=0.05, fc='black', ec='black', length_includes_head=True)

            ax.text(c, r+0.35, f"{V[r, c]:.1f}", ha='center', va='center', color='black', fontsize=7)

    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
