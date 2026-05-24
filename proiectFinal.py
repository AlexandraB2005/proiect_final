# import cv2
# import math
# import maxflow
# import numpy as np
# import matplotlib.pyplot as plt


# sigma = 50.0
# SEED_CAPACITY = 100000



# img = cv2.imread("banana.png")

# if img is None:
#     raise Exception("Image not found!")
# img = cv2.resize(img, (500, 500*img.shape[0]//img.shape[1]))

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# rows, cols, _ = img.shape

# print("Image shape:", img.shape)

# #initializam mascile pentru foreground si background cu 0 
# fg_mask = np.zeros((rows, cols), dtype=np.uint8)
# bg_mask = np.zeros((rows, cols), dtype=np.uint8)

# # Convert image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# # Threshold the image to get a binary mask 
# _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV) #intre 0-240 = background, 240-255 = foreground

# # Find contours
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Find the largest contour
# if contours:
#     largest_contour = max(contours, key=cv2.contourArea)
#     cv2.drawContours(fg_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)

# # image borders = background pp ca obiectul nu atinge marginile imaginii
# border = 5

# bg_mask[:border, :] = 1
# bg_mask[-border:, :] = 1
# bg_mask[:, :border] = 1
# bg_mask[:, -border:] = 1

# # background = 1, foreground = 0 
# bg_mask[fg_mask == 1] = 0


# # Calculate mean and variance for foreground and background - e +1 sa nu avem /0 in calculul costurilor
# fg_pixels = img[fg_mask == 1].astype(np.float32)
# bg_pixels = img[bg_mask == 1].astype(np.float32)

# fg_mean = np.mean(fg_pixels, axis=0)
# bg_mean = np.mean(bg_pixels, axis=0)

# fg_var = np.var(fg_pixels, axis=0) + 1
# bg_var = np.var(bg_pixels, axis=0) + 1

# print("Foreground mean:", fg_mean)
# print("Background mean:", bg_mean)


# #distanta euclidiana intre doua culori (fara sqrt ca doar comparam distantele)
# def color_distance(c1, c2): 
#     return np.sum((c1 - c2) ** 2)

# # costul gaussian pentru a determina cat de bine se potriveste un pixel cu modelul de foreground sau background
# def gaussian_cost(color, mean, var):
#     return np.sum(
#         ((color - mean) ** 2) / (2 * var)
#     )


# g = maxflow.Graph[float]()

# nodes = g.add_grid_nodes((rows, cols))


# for i in range(rows):
#     for j in range(cols):

#         current = img[i, j].astype(np.float32) # culoarea pixelului curent convertita la float32 pentru calcule ulterioare

#         # RIGHT
#         if j + 1 < cols:

#             neigh = img[i, j + 1].astype(np.float32)

#             dist_sq = color_distance(current, neigh)

#             weight = 500 * math.exp(
#                 -dist_sq / (2 * sigma * sigma)
#             ) # cat de similare sunt cele doua culori, cu cat sunt mai similare, cu atat mai mare este greutatea muchiei

#             weight = max(weight, 1)

#             g.add_edge(
#                 nodes[i, j],
#                 nodes[i, j + 1],
#                 weight,
#                 weight
#             )

#         # DOWN
#         if i + 1 < rows:

#             neigh = img[i + 1, j].astype(np.float32)

#             dist_sq = color_distance(current, neigh)

#             weight = 500 * math.exp(
#                 -dist_sq / (2 * sigma * sigma)
#             )

#             weight = max(weight, 1)

#             g.add_edge(
#                 nodes[i, j],
#                 nodes[i + 1, j],
#                 weight,
#                 weight
#             )


# #taietura se verifica ce noduri sunt in masca de foreground sau background si se adauga muchii catre sursa sau destinatie cu costuri corespunzatoare
# for i in range(rows):
#     for j in range(cols):

#         color = img[i, j].astype(np.float32)

#         # hard foreground
#         if fg_mask[i, j]:

#             g.add_tedge(
#                 nodes[i, j],
#                 SEED_CAPACITY,
#                 0
#             )

#             continue

#         # hard background
#         if bg_mask[i, j]:

#             g.add_tedge(
#                 nodes[i, j],
#                 0,
#                 SEED_CAPACITY
#             )

#             continue

#         fg_cost = gaussian_cost(
#             color,
#             fg_mean,
#             fg_var
#         )

#         bg_cost = gaussian_cost(
#             color,
#             bg_mean,
#             bg_var
#         )

#         g.add_tedge(
#             nodes[i, j],
#             bg_cost,
#             fg_cost
#         )



# flow = g.maxflow()

# print("Max Flow =", flow)

 

# segments = g.get_grid_segments(nodes)
# foreground = np.logical_not(segments)
# mask = foreground.astype(np.uint8) * 255

# kernel = np.ones((5, 5), np.uint8)

# mask = cv2.morphologyEx(
#     mask,
#     cv2.MORPH_OPEN,
#     kernel
# )

# mask = cv2.GaussianBlur(mask, (5,5), 0)



# result = img.copy()

# result[mask == 0] = 0 # background pixels set to black


# plt.figure(figsize=(15, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.title("Original")
# plt.axis("off")
# plt.subplot(1, 2, 2)

# plt.imshow(result)
# plt.title("Extracted Object")
# plt.axis("off")

# plt.tight_layout()
# plt.show()

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque



sigma = 50.0
SEED_CAPACITY = 100000


class Edge:
    def __init__(self, to, capacity):
        self.to = to
        self.capacity = capacity
        self.rev = None


class Graph:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v, capacity):
        forward = Edge(v, capacity)
        backward = Edge(u, 0)

        forward.rev = backward
        backward.rev = forward

        self.adj[u].append(forward)
        self.adj[v].append(backward)


def bfs(graph, source, sink, parent):

    visited = [False] * graph.n

    queue = deque([source])

    visited[source] = True

    while queue:

        u = queue.popleft()

        for edge in graph.adj[u]:

            if edge.capacity > 0 and not visited[edge.to]:

                visited[edge.to] = True

                parent[edge.to] = (u, edge)

                queue.append(edge.to)

                if edge.to == sink:
                    return True

    return False


def ford_fulkerson(graph, source, sink):

    max_flow = 0

    while True:
# pentru a gasi un drum de la sursa la destinatie cu capacitate disponibila, folosim BFS pentru a parcurge graful si a inregistra parintii nodurilor vizitate, astfel incat sa putem reconstrui drumul gasit
        parent = [None] * graph.n
# daca nu mai exista un drum de la sursa la destinatie cu capacitate disponibila, inseamna ca am atins fluxul maxim si iesim din bucla
        if not bfs(graph, source, sink, parent):
            break
# daca am gasit un drum, determinam fluxul maxim care poate fi trimis pe acest drum, adica minimul capacitatii muchiilor de pe drum, parcurgand drumul de la destinatie la sursa folosind parintii inregistrati in BFS
        path_flow = float("inf")

        v = sink
        # parcurgem drumul de la destinatie la sursa folosind parintii inregistrati in BFS pentru a determina fluxul maxim care poate fi trimis pe acest drum, 
        # adica minimul capacitatii muchiilor de pe drum
        while v != source:

            u, edge = parent[v]
# pentru fiecare muchie de pe drum, actualizam path_flow cu minimul dintre path_flow curent si capacitatea muchiei, astfel incat la final sa avem fluxul maxim care poate fi trimis pe acest drum
            path_flow = min(
                path_flow,
                edge.capacity
            )

            v = u

        v = sink
# dupa ce am determinat fluxul maxim care poate fi trimis pe acest drum, actualizam capacitatile muchiilor de pe drum scazand fluxul trimis din capacitatea muchiei directe si adaugand fluxul trimis la capacitatea muchiei inverse 
# (pentru a permite posibile reveniri in iteratiile urmatoare)
        while v != source:

            u, edge = parent[v]
# actualizam capacitatea muchiei directe scazand fluxul trimis, astfel incat sa reflecte capacitatea ramasa dupa trimiterea fluxului pe acest drum
            edge.capacity -= path_flow
            edge.rev.capacity += path_flow

            v = u
# adaugam fluxul trimis pe acest drum la fluxul total, astfel incat la final sa avem fluxul maxim care poate fi trimis de la sursa la destinatie
        max_flow += path_flow

    return max_flow



img = cv2.imread("banana.png")

if img is None:
    raise Exception("Image not found!")


# img = cv2.resize(
#     img,
#     (
#         200,
#         int(200 * img.shape[0] / img.shape[1])
#     )
# )

img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
b_channel, g_channel, r_channel = cv2.split(img_bgr)


img = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)

rows, cols, _ = img.shape

print("Image shape:", img.shape)



fg_mask = np.zeros((rows, cols), dtype=np.uint8)
bg_mask = np.zeros((rows, cols), dtype=np.uint8)

# Convert image to grayscale - pentru a simplifica procesul de detectare a conturului obiectului, convertim imaginea in grayscale, astfel incat sa avem doar o singura valoare de intensitate pentru fiecare pixel, in loc de 3 canale de culoare (RGB)
gray = cv2.cvtColor(
    img,
    cv2.COLOR_RGB2GRAY
)


# Threshold the image to get a binary mask - presupunem ca obiectul are culori mai inchise, iar fundalul mai deschis, 
# deci folosim THRESH_BINARY_INV pentru a inversa masca astfel incat obiectul sa fie foreground (1) si fundalul sa fie background (0)
_, thresh = cv2.threshold(
    gray,
    240,
    255,
    cv2.THRESH_BINARY_INV #inversam culorile astfel incat obiectul sa fie foreground (1) si fundalul sa fie background (0)
) 

plt.imshow(thresh, cmap="gray")
plt.title("Inverted Image")
plt.axis("off")

# Find contours - contururile reprezinta marginile obiectelor din imagine, iar
# noi vrem sa gasim conturul obiectului pentru a-l marca ca foreground
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if contours:

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    cv2.drawContours(
        fg_mask,
        [largest_contour],
        -1,
        1,
        thickness=cv2.FILLED
    )

border = 5

bg_mask[:border, :] = 1
bg_mask[-border:, :] = 1
bg_mask[:, :border] = 1
bg_mask[:, -border:] = 1

bg_mask[fg_mask == 1] = 0



fg_pixels = img[fg_mask == 1].astype(np.float32)
bg_pixels = img[bg_mask == 1].astype(np.float32)

fg_mean = np.mean(fg_pixels, axis=0)
bg_mean = np.mean(bg_pixels, axis=0)

fg_var = np.var(fg_pixels, axis=0) + 1
bg_var = np.var(bg_pixels, axis=0) + 1

print("Foreground mean:", fg_mean)
print("Background mean:", bg_mean)


def color_distance(c1, c2):

    return np.sum(
        (c1 - c2) ** 2
    )


def gaussian_cost(color, mean, var):

    return np.sum(
        ((color - mean) ** 2)
        / (2 * var)
    )


def pixel_to_node(i, j):

    return i * cols + j 



num_pixels = rows * cols

source = num_pixels
sink = num_pixels + 1

V = num_pixels + 2

g = Graph(V)


#
for i in range(rows):
    for j in range(cols):

        current = img[i, j].astype(np.float32)

        u = pixel_to_node(i, j)

        # RIGHT

        if j + 1 < cols:

            neigh = img[i, j + 1].astype(np.float32)

            dist_sq = color_distance(
                current,
                neigh
            )

            weight = (
                500
                * math.exp(
                    -dist_sq
                    / (2 * sigma * sigma)
                )
            )

            weight = max(weight, 1)

            v = pixel_to_node(i, j + 1)

            g.add_edge(u, v, weight)
            g.add_edge(v, u, weight)

        # DOWN

        if i + 1 < rows:

            neigh = img[i + 1, j].astype(np.float32)

            dist_sq = color_distance(
                current,
                neigh
            )

            weight = (
                500
                * math.exp(
                    -dist_sq
                    / (2 * sigma * sigma)
                )
            )

            weight = max(weight, 1)

            v = pixel_to_node(i + 1, j)

            g.add_edge(u, v, weight)
            g.add_edge(v, u, weight)



for i in range(rows):
    for j in range(cols):

        node = pixel_to_node(i, j)

        color = img[i, j].astype(np.float32)

        if fg_mask[i, j]:

            g.add_edge(
                source,
                node,
                SEED_CAPACITY
            )

            continue

        if bg_mask[i, j]:

            g.add_edge(
                node,
                sink,
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

        g.add_edge(
            source,
            node,
            bg_cost
        )

        g.add_edge(
            node,
            sink,
            fg_cost
        )



flow = ford_fulkerson(
    g,
    source,
    sink
)

print("Max Flow =", flow)




visited = [False] * V

queue = deque([source])

visited[source] = True 
# marcam nodul sursa ca vizitat pentru a incepe BFS-ul din sursa, astfel incat sa putem determina care noduri sunt in componenta conexa a sursei (foreground) si care nu (background)
while queue:

    u = queue.popleft()

    for edge in g.adj[u]:

        if edge.capacity > 0 and not visited[edge.to]: 

            visited[edge.to] = True
            queue.append(edge.to)


# daca un nod este vizitat, inseamna ca face parte din componenta conexa a sursei, adica este considerat foreground, altfel este background. Astfel, construim o masca binara in care pixelii foreground sunt albi (255) si pixelii background sunt negri (0)
mask = np.zeros(
    (rows, cols),
    dtype=np.uint8
)

# parcurgem fiecare pixel si verificam daca nodul corespunzator este vizitat sau nu, pentru a construi masca finala care va fi folosita pentru a extrage obiectul din imagine
for i in range(rows):
    for j in range(cols):

        node = pixel_to_node(i, j)

        if visited[node]:
            mask[i, j] = 255 # foreground



kernel = np.ones((5, 5), np.uint8)



mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)


mask = cv2.GaussianBlur(
    mask,
    (5, 5),
    0
)


channels = [b_channel, g_channel, r_channel, mask]
result_transparent = cv2.merge(channels)
result_transparent = cv2.cvtColor(
    result_transparent,
    cv2.COLOR_BGRA2RGBA
)
cv2.imwrite("banana_transparent.png", result_transparent)
print("Imaginea transparenta a fost salvata ca 'banana_transparent.png'")
result_plt = cv2.cvtColor(result_transparent, cv2.COLOR_BGRA2RGBA)

result = img.copy()

result[mask == 0] = 0


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)

plt.imshow(result)
plt.title("Segmented")
plt.axis("off")

plt.tight_layout()
plt.show()