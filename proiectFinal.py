import cv2
import math
import maxflow
import numpy as np
import matplotlib.pyplot as plt


sigma = 50.0
SEED_CAPACITY = 100000



img = cv2.imread("banana.png")

if img is None:
    raise Exception("Image not found!")
img = cv2.resize(img, (500, 500*img.shape[0]//img.shape[1]))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


rows, cols, _ = img.shape

print("Image shape:", img.shape)

#initializam mascile pentru foreground si background cu 0 
fg_mask = np.zeros((rows, cols), dtype=np.uint8)
bg_mask = np.zeros((rows, cols), dtype=np.uint8)

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Threshold the image to get a binary mask 
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV) #intre 0-240 = background, 240-255 = foreground

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(fg_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)

# image borders = background pp ca obiectul nu atinge marginile imaginii
border = 5

bg_mask[:border, :] = 1
bg_mask[-border:, :] = 1
bg_mask[:, :border] = 1
bg_mask[:, -border:] = 1

# background = 1, foreground = 0 
bg_mask[fg_mask == 1] = 0


# Calculate mean and variance for foreground and background - e +1 sa nu avem /0 in calculul costurilor
fg_pixels = img[fg_mask == 1].astype(np.float32)
bg_pixels = img[bg_mask == 1].astype(np.float32)

fg_mean = np.mean(fg_pixels, axis=0)
bg_mean = np.mean(bg_pixels, axis=0)

fg_var = np.var(fg_pixels, axis=0) + 1
bg_var = np.var(bg_pixels, axis=0) + 1

print("Foreground mean:", fg_mean)
print("Background mean:", bg_mean)


#distanta euclidiana intre doua culori (fara sqrt ca doar comparam distantele)
def color_distance(c1, c2): 
    return np.sum((c1 - c2) ** 2)

# costul gaussian pentru a determina cat de bine se potriveste un pixel cu modelul de foreground sau background
def gaussian_cost(color, mean, var):
    return np.sum(
        ((color - mean) ** 2) / (2 * var)
    )


g = maxflow.Graph[float]()

nodes = g.add_grid_nodes((rows, cols))


for i in range(rows):
    for j in range(cols):

        current = img[i, j].astype(np.float32) # culoarea pixelului curent convertita la float32 pentru calcule ulterioare

        # RIGHT
        if j + 1 < cols:

            neigh = img[i, j + 1].astype(np.float32)

            dist_sq = color_distance(current, neigh)

            weight = 500 * math.exp(
                -dist_sq / (2 * sigma * sigma)
            ) # cat de similare sunt cele doua culori, cu cat sunt mai similare, cu atat mai mare este greutatea muchiei

            weight = max(weight, 1)

            g.add_edge(
                nodes[i, j],
                nodes[i, j + 1],
                weight,
                weight
            )

        # DOWN
        if i + 1 < rows:

            neigh = img[i + 1, j].astype(np.float32)

            dist_sq = color_distance(current, neigh)

            weight = 500 * math.exp(
                -dist_sq / (2 * sigma * sigma)
            )

            weight = max(weight, 1)

            g.add_edge(
                nodes[i, j],
                nodes[i + 1, j],
                weight,
                weight
            )


#taietura se verifica ce noduri sunt in masca de foreground sau background si se adauga muchii catre sursa sau destinatie cu costuri corespunzatoare
for i in range(rows):
    for j in range(cols):

        color = img[i, j].astype(np.float32)

        # hard foreground
        if fg_mask[i, j]:

            g.add_tedge(
                nodes[i, j],
                SEED_CAPACITY,
                0
            )

            continue

        # hard background
        if bg_mask[i, j]:

            g.add_tedge(
                nodes[i, j],
                0,
                SEED_CAPACITY
            )

            continue

        fg_cost = gaussian_cost(
            color,
            fg_mean,
            fg_var
        )

        bg_cost = gaussian_cost(
            color,
            bg_mean,
            bg_var
        )

        g.add_tedge(
            nodes[i, j],
            bg_cost,
            fg_cost
        )



flow = g.maxflow()

print("Max Flow =", flow)

 

segments = g.get_grid_segments(nodes)
foreground = np.logical_not(segments)
mask = foreground.astype(np.uint8) * 255

kernel = np.ones((5, 5), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)

mask = cv2.GaussianBlur(mask, (5,5), 0)



result = img.copy()

result[mask == 0] = 0 # background pixels set to black


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")
plt.subplot(1, 2, 2)

plt.imshow(result)
plt.title("Extracted Object")
plt.axis("off")

plt.tight_layout()
plt.show()