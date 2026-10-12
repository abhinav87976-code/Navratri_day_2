
import os
import cv2
import turtle
import numpy as np

# ---------------- CONFIGURATION ----------------
IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brahmacharini.jpg")
TARGET_WIDTH = 750
THRESHOLD_VALUE = 180
MIN_CONTOUR_AREA = 4

# ---------------- LOAD IMAGE ----------------
if not os.path.isfile(IMAGE_PATH):
    print(f"Error: '{IMAGE_PATH}' not found.")
    print("Keep brahmacharini.jpg in the same folder as this Python file.")
    raise SystemExit(1)

image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: OpenCV could not read the image.")
    raise SystemExit(1)

height, width = image.shape
target_height = max(1, int(TARGET_WIDTH * height / width))

resized = cv2.resize(
    image,
    (TARGET_WIDTH, target_height),
    interpolation=cv2.INTER_AREA
)

# Convert dark areas into foreground shapes
_, binary = cv2.threshold(
    resized,
    THRESHOLD_VALUE,
    255,
    cv2.THRESH_BINARY_INV
)

# RETR_TREE preserves nested contours
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

if hierarchy is None or not contours:
    print("Error: No contours detected. Try changing THRESHOLD_VALUE.")
    raise SystemExit(1)

hierarchy = hierarchy[0]

# ---------------- CONTOUR DEPTH ----------------
def get_depth(index):
    depth = 0
    parent = hierarchy[index][3]

    while parent != -1:
        depth += 1
        parent = hierarchy[parent][3]

    return depth


layers = {}

for index, contour in enumerate(contours):
    if abs(cv2.contourArea(contour)) < MIN_CONTOUR_AREA:
        continue

    depth = get_depth(index)
    layers.setdefault(depth, []).append(contour)

# ---------------- TURTLE WINDOW ----------------
screen = turtle.Screen()
screen.setup(
    width=min(TARGET_WIDTH + 80, 1200),
    height=min(target_height + 100, 1000)
)
screen.title("Mata Brahmacharini - Contour Drawing")
screen.bgcolor("white")
screen.tracer(2, 0)

pen = turtle.Turtle()
pen.hideturtle()
pen.speed(0)
pen.pensize(1)
pen.shape("classic")
pen.showturtle()

half_w = TARGET_WIDTH / 2
half_h = target_height / 2

def to_turtle_coords(x, y):
    return x - half_w, half_h - y


# ---------------- DRAW CONTOURS ----------------
def render_shape(contour, color):
    # Approximate contours to reduce excessive drawing operations
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(0.5, perimeter * 0.0008)

    points = cv2.approxPolyDP(
        contour, epsilon, True
    ).reshape(-1, 2)

    if len(points) < 3:
        return

    first_x, first_y = points[0]
    pen.penup()
    pen.goto(to_turtle_coords(first_x, first_y))
    pen.pencolor(color)
    pen.fillcolor(color)
    pen.pendown()
    pen.begin_fill()

    for x, y in points[1:]:
        pen.goto(to_turtle_coords(x, y))

    pen.goto(to_turtle_coords(first_x, first_y))
    pen.end_fill()
    pen.penup()


# ---------------- RENDER BY DEPTH ----------------
print(f"Image size: {TARGET_WIDTH} x {target_height}")
print(f"Contours detected: {len(contours)}")
print("Drawing started...")

for depth in sorted(layers):
    # Alternate dark foreground and white cutouts
    color = "black" if depth % 2 == 0 else "white"

    for contour in layers[depth]:
        render_shape(contour, color)

    screen.update()

pen.hideturtle()
screen.update()

print("Drawing complete!")
screen.mainloop()
