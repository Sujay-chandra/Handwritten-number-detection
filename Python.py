import pygame
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model you downloaded from Colab
model = load_model("bestmodel_manual_save.h5")

pygame.init()
window_size = 280  # 10x scale of MNIST (28px → 280px)
window = pygame.display.set_mode((window_size, window_size))
pygame.display.set_caption("Draw a Digit (Press Enter to Predict)")
clock = pygame.time.Clock()
window.fill((0, 0, 0))
drawing = False

def predict_digit(surface):
    # Convert pygame surface to numpy array
    canvas_str = pygame.surfarray.array3d(surface)
    canvas_gray = np.dot(canvas_str[..., :3], [0.2989, 0.5870, 0.1140])  # RGB → Gray
    # Resize to 28x28
    img_small = pygame.transform.scale(pygame.surfarray.make_surface(canvas_gray.T), (28, 28))
    img_array = pygame.surfarray.array3d(img_small)[..., 0]  # Take R channel
    img_array = 255 - img_array  # Invert colors (white digit on black)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)
    pred = model.predict(img_array)
    print("Predicted Digit:", np.argmax(pred))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        if event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        if event.type == pygame.MOUSEMOTION and drawing:
            pygame.draw.circle(window, (255, 255, 255), event.pos, 10)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:  # Enter key → Predict
                predict_digit(window)
            if event.key == pygame.K_c:  # C key → Clear canvas
                window.fill((0, 0, 0))

    pygame.display.update()
    clock.tick(60)

pygame.quit()
