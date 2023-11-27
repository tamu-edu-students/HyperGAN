import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse 

# # Read the image
# image = cv2.imread(r'/workspaces/HyperGAN/datasets/export_3/testB/16_rgb.png')


# def add_dirt(image):
#     # Create a mask for the dirt flecks
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     for i in range(10, 15):
#         # Generate random coordinates for the dirt flecks
#         x = np.random.randint(0, image.shape[1])
#         y = np.random.randint(0, image.shape[0])

#         # Draw a small circle on the mask at the random coordinates
#         cv2.circle(mask, (x, y), 2, 255, -1)

#     # Create a copy of the image
#     dirty_image = image.copy()

#     # Apply the mask to the image to add the dirt flecks
#     dirty_image[mask == 255] = [102, 51, 0]  # Brown color

#     return dirty_image


# # Add dirt to the image
# dirty_image = add_dirt(image)

# # Save the dirty image
# cv2.imwrite(r'/workspaces/HyperGAN/datasets/export_3/testB/16_dirt_rgb.png', dirty_image)

def add_dirt(ax, width, height):
    num_shapes = np.random.randint(10, 16)
    for _ in range(num_shapes):
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        shape_type = np.random.choice(['ellipse', 'rectangle'])
        
        if shape_type == 'ellipse':
            width_ellipse = np.random.uniform(5, 15)
            height_ellipse = np.random.uniform(5, 15)
            angle = np.random.uniform(0, 360)
            ellipse = Ellipse((x, y), width_ellipse, height_ellipse, angle=angle, color='brown')
            ax.add_patch(ellipse)
        elif shape_type == 'rectangle':
            width_rect = np.random.uniform(5, 15)
            height_rect = np.random.uniform(5, 15)
            angle = np.random.uniform(0, 360)
            rectangle = Rectangle((x - width_rect / 2, y - height_rect / 2), width_rect, height_rect, angle=angle, color='brown')
            ax.add_patch(rectangle)

def generate_dirt_image(width=100, height=100):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal', 'box')
    ax.axis('off')

    add_dirt(ax, width, height)

    plt.savefig('dirt_on_lens.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    generate_dirt_image()